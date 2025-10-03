import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as OpenAILLM
from llama_index.core import Settings
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# ==== Setup ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY)

# ==== FastAPI App ====
app = FastAPI()
# Permitir todas origens para teste rápido (substitua por origem específica em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # <--- durante testes usar "*" ; em produção coloque a URL do site Lovable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Load RAG ====
def load_rag_index():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()

    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    llm = OpenAILLM(model="gpt-4o-mini", temperature=1.0)

    Settings.llm = llm
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(docs)
    return index

index = load_rag_index()
rag_engine = index.as_query_engine(similarity_top_k=3)

# ==== Função: proximidade ====
def mais_proximos(user_lat, user_long, top_k=3):
    df = pd.read_csv("./data/pontos_coleta.csv")
    df["dist"] = ((df["lat"] - user_lat)**2 + (df["long"] - user_long)**2)**0.5
    df_sorted = df.sort_values("dist").head(top_k)
    return df_sorted

# ==== Webhook antigo (WhatsApp simulado) ====
class WebhookMessage(BaseModel):
    from_number: str
    text: str

@app.post("/webhook")
async def webhook(msg: WebhookMessage):
    from_number = msg.from_number
    text = msg.text.strip()

    if text.lower().startswith("perto"):
        parts = text.split()
        if len(parts) == 3:
            user_lat, user_long = float(parts[1]), float(parts[2])
            locais = mais_proximos(user_lat, user_long)
            return {"reply": "\n".join([f"{row['nome']} - {row['endereco']}" for _, row in locais.iterrows()])}
        else:
            return {"reply": "Envie no formato: perto LAT LONG"}
    else:
        query = f"""
        Você é um especialista em descarte de baterias elétricas e automotivas no Brasil.

        Responda à pergunta do usuário de forma clara, objetiva e útil.  
        Use **exclusivamente** as informações recuperadas dos documentos fornecidos pelo sistema RAG.  
        Sempre mencione explicitamente o(s) nome(s) do(s) arquivo(s) de onde a informação foi extraída.  
        Se nenhum documento recuperado for relevante, responda claramente que não encontrou informações.
        
        Pergunta do usuário: {text}
        """
        response = rag_engine.query(query)
        return {"reply": response.response}

# ==== Novo endpoint público para o Lovable ====
class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat(msg: ChatMessage):
    user_input = msg.message.strip()
    query = f"""
    Você é um especialista em descarte de baterias elétricas e automotivas no Brasil.

    Responda à pergunta do usuário de forma clara, objetiva e útil.  
    Use **exclusivamente** as informações recuperadas dos documentos fornecidos pelo sistema RAG.  
    Sempre mencione explicitamente o(s) nome(s) do(s) arquivo(s) de onde a informação foi extraída.  
    Se nenhum documento recuperado for relevante, responda claramente que não encontrou informações.

    Pergunta do usuário: {user_input}
    """
    response = rag_engine.query(query)
    return {"reply": response.response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

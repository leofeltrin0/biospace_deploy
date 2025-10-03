import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as OpenAILLM
from openai import OpenAI

# ==== Setup ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY)

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

# ==== Função: chat local ====
def chat_local(user_input):
    query = f"""
    Você é um especialista em biologia.

    Responda à pergunta do usuário de forma clara, objetiva e útil.  
    Use **exclusivamente** as informações recuperadas dos documentos fornecidos pelo sistema RAG.  
    Sempre mencione explicitamente o(s) nome(s) do(s) arquivo(s) de onde a informação foi extraída.  
    Se nenhum documento recuperado for relevante, responda claramente que não encontrou informações.

    Pergunta do usuário: {user_input}
    """
    response = rag_engine.query(query)
    return response.response

# ==== Teste interativo ====
if __name__ == "__main__":
    print("=== RAG Local (Biology Specialist) ===")
    while True:
        user_input = input("\nDigite sua pergunta (ou 'sair' para encerrar): ").strip()
        if user_input.lower() == "sair":
            break
        elif user_input.lower().startswith("perto"):
            parts = user_input.split()
            if len(parts) == 3:
                try:
                    lat, long = float(parts[1]), float(parts[2])
                    locais = mais_proximos(lat, long)
                    for _, row in locais.iterrows():
                        print(f"{row['nome']} - {row['endereco']}")
                except ValueError:
                    print("Formato inválido: perto LAT LONG")
            else:
                print("Envie no formato: perto LAT LONG")
        else:
            reply = chat_local(user_input)
            print("\nResposta:\n", reply)

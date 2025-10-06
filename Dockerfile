# Base moderna suportada
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Só o essencial; remova software-properties-common e build-essential
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
    ; \
    rm -rf /var/lib/apt/lists/*

# Cache melhor: requirements primeiro
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# (Opcional) Baixar modelo spaCy; se preferir, pinne via pip:
#   adicione "en-core-web-sm==3.7.1" no requirements.txt
RUN python -m spacy download en_core_web_sm

# Código
COPY . .

# Pastas necessárias
RUN mkdir -p logs data/processed data/vectorstore data/kg_store

# Render define $PORT em runtime; mantenha um default local
ENV PORT=8000

# Expose (apenas informativo no Render)
EXPOSE 8000

# Healthcheck usando $PORT
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Se você tem FastAPI com `app` em main.py, use uvicorn:
# CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

# Caso contrário (se o seu servidor é `python main.py --mode serve`), use:
CMD ["sh", "-c", "python main.py --mode serve"]

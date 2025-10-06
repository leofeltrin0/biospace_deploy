"""
NASA Space Apps Hackathon MVP - Space Mission Knowledge Engine
Main CLI orchestrator (S3-enabled) for the intelligent space biology research system
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from modules.pdf_processing import PDFProcessor
from modules.chunking import TextChunker
from modules.embeddings import EmbeddingGenerator
from modules.vectorstore import VectorStore
from modules.kg_extraction import KGExtractor
from modules.kg_store import KGStore
from modules.mission_engine import MissionEngine
from modules.adaptive_generator import AdaptiveGenerator
from modules.prompt_engineering import PromptEngineer

# ================= S3 helpers =================
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def s3_sync_down(bucket: str, prefix: str, dest_dir: str, logger: logging.Logger, only_extensions: Optional[tuple] = None) -> int:
    """
    Baixa todos os objetos de s3://bucket/prefix para dest_dir, preservando subpastas.
    Retorna a contagem de arquivos baixados/atualizados.
    """
    _ensure_dir(dest_dir)
    s3 = boto3.client("s3", config=BotoConfig(retries={"max_attempts": 5}))
    paginator = s3.get_paginator("list_objects_v2")
    count = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            # filtro por extensão (opcional)
            if only_extensions and not any(key.lower().endswith(ext) for ext in only_extensions):
                continue

            rel = key[len(prefix):] if key.startswith(prefix) else key
            local_path = os.path.join(dest_dir, rel)
            _ensure_dir(os.path.dirname(local_path))

            # pulo simples se arquivo existe com mesmo tamanho
            if os.path.exists(local_path):
                try:
                    head = s3.head_object(Bucket=bucket, Key=key)
                    size_remote = head.get("ContentLength", -1)
                    size_local = os.path.getsize(local_path)
                    if size_local == size_remote and size_remote >= 0:
                        continue  # já está ok
                except ClientError:
                    pass

            logger.info(f"S3 ↓ {key} -> {local_path}")
            s3.download_file(bucket, key, local_path)
            count += 1
    return count

def s3_upload_file(bucket: str, local_path: str, dest_key: str, logger: logging.Logger) -> None:
    s3 = boto3.client("s3", config=BotoConfig(retries={"max_attempts": 5}))
    logger.info(f"S3 ↑ {local_path} -> s3://{bucket}/{dest_key}")
    extra_args = {}
    # Se for PDF, define content-type p/ abrir no navegador (caso use público)
    if local_path.lower().endswith(".pdf"):
        extra_args["ContentType"] = "application/pdf"
    s3.upload_file(local_path, bucket, dest_key, ExtraArgs=extra_args)

def s3_upload_dir(bucket: str, local_dir: str, dest_prefix: str, logger: logging.Logger) -> int:
    """
    Sobe recursivamente um diretório local para um prefixo no S3.
    """
    uploaded = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            lpath = os.path.join(root, fname)
            rel = os.path.relpath(lpath, start=local_dir).replace("\\", "/")
            key = f"{dest_prefix.rstrip('/')}/{rel}"
            s3_upload_file(bucket, lpath, key, logger)
            uploaded += 1
    return uploaded

# ================ Load env & config ================
load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")            # ex.: biospacedata
S3_PREFIX = os.getenv("S3_PREFIX", "rag/")    # ex.: rag/
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-2")

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration"""
    log_config = config.get("logging", {})
    log_file = log_config.get("file", "./logs/space_mission_engine.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("main")
    logger.info("Logging configured successfully")

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

# ================== Pipelines ==================
def run_ingest_pipeline(config: Dict[str, Any]) -> None:
    """Run the complete document ingestion pipeline (S3 → local → S3)"""
    logger = logging.getLogger("main")
    logger.info("Starting document ingestion pipeline")

    # --- S3 → local (docs e meta) ---
    if S3_BUCKET:
        docs_prefix = f"{S3_PREFIX.rstrip('/')}/docs/"
        meta_prefix = f"{S3_PREFIX.rstrip('/')}/meta/"
        dl_docs = s3_sync_down(S3_BUCKET, docs_prefix, "./data/raw", logger, only_extensions=(".pdf", ".txt"))
        dl_meta = s3_sync_down(S3_BUCKET, meta_prefix, "./data/meta", logger)
        logger.info(f"S3 sync down: docs={dl_docs} novos/atualizados, meta={dl_meta} novos/atualizados")
    else:
        logger.warning("S3_BUCKET not set — using local ./data for artifacts. For production set S3_BUCKET and S3_PREFIX.")

    try:
        # Step 1: PDF Processing
        logger.info("Step 1: Processing PDFs (from ./data/raw)")
        pdf_processor = PDFProcessor(config.get("pdf_processing", {}))
        pdf_results = pdf_processor.process_all_pdfs()  # deve ler ./data/raw e escrever ./data/processed
        logger.info(f"PDF processing completed: {pdf_results.get('successful',0)} successful, {pdf_results.get('failed',0)} failed")

        # Step 2: Text Chunking
        logger.info("Step 2: Chunking documents")
        text_chunker = TextChunker(config.get("chunking", {}))
        chunk_results = text_chunker.chunk_all_documents("./data/processed", "./data/chunks.json")
        logger.info(f"Chunking completed: {chunk_results.get('total_chunks',0)} chunks created")

        # Step 3: Generate Embeddings
        logger.info("Step 3: Generating embeddings")
        embedding_generator = EmbeddingGenerator(config.get("embeddings", {}))
        embedding_results = embedding_generator.batch_process_chunks("./data/chunks.json", "./data/embeddings.json")
        logger.info(f"Embeddings completed: {embedding_results.get('embedded_chunks', 0)} embedded")

        # Step 4: Store in Vector Store
        logger.info("Step 4: Storing in vector database")
        vectorstore = VectorStore(**config.get("vectorstore", {}))
        import json
        with open("./data/embeddings.json", 'r', encoding='utf-8') as f:
            embeddings_data = json.load(f)
        vectorstore.add_embeddings(embeddings_data)
        vectorstore.save()
        logger.info("Embeddings stored in vector database")

        # Step 5: Knowledge Graph Extraction
        logger.info("Step 5: Extracting knowledge graph")
        kg_extractor = KGExtractor(config.get("kg_extraction", {}))
        kg_store = KGStore(config.get("kg_store", {}))
        with open("./data/chunks.json", 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        processed_chunks = kg_extractor.process_chunks_for_kg(chunks_data)

        all_triples = []
        for chunk in processed_chunks:
            for triple in chunk.get("extracted_triples", []):
                all_triples.append((triple["subject"], triple["relation"], triple["object"], triple["confidence"]))
        kg_store.add_triples(all_triples)
        kg_store.save_graph()
        logger.info(f"Knowledge graph updated with {len(all_triples)} triples")

        # --- Upload artefatos p/ S3 (opcional, mas útil) ---
        if S3_BUCKET:
            out_prefix = f"{S3_PREFIX.rstrip('/')}/output"
            s3_upload_file(S3_BUCKET, "./data/chunks.json", f"{out_prefix}/chunks.json", logger)
            s3_upload_file(S3_BUCKET, "./data/embeddings.json", f"{out_prefix}/embeddings.json", logger)
            # Se o KG gera um arquivo (ajuste conforme sua implementação):
            kg_file = "./data/kg.graph"
            if os.path.exists(kg_file):
                s3_upload_file(S3_BUCKET, kg_file, f"{out_prefix}/kg.graph", logger)
            # Se o VectorStore salva uma pasta (ex.: ./data/vectorstore), suba também:
            vecdir = "./data/vectorstore"
            if os.path.isdir(vecdir):
                uploaded = s3_upload_dir(S3_BUCKET, vecdir, f"{out_prefix}/vectorstore", logger)
                logger.info(f"Vectorstore uploaded files: {uploaded}")
        else:
            logger.info("S3_BUCKET not set — artifacts saved locally only")

        logger.info("Document ingestion pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in ingestion pipeline: {str(e)}")
        raise

def run_embed_pipeline(config: Dict[str, Any]) -> None:
    """Run embedding generation pipeline (S3 → local → S3)"""
    logger = logging.getLogger("main")
    logger.info("Starting embedding generation pipeline")

    if S3_BUCKET:
        # Certifica que chunks.json está local (baixa de output caso exista)
        out_prefix = f"{S3_PREFIX.rstrip('/')}/output/"
        s3_sync_down(S3_BUCKET, out_prefix, "./data", logger)  # traz chunks.json/embeddings.json prévias se existirem
    else:
        logger.warning("S3_BUCKET not set — using local ./data for artifacts.")

    try:
        import json
        with open("./data/chunks.json", 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        embedding_generator = EmbeddingGenerator(config.get("embeddings", {}))
        embedding_results = embedding_generator.batch_process_chunks("./data/chunks.json", "./data/embeddings.json")
        logger.info(f"Embedding generation completed: {embedding_results.get('embedded_chunks', 0)} embedded")

        # sobe embeddings atualizados
        if S3_BUCKET:
            s3_upload_file(S3_BUCKET, "./data/embeddings.json", f"{out_prefix}embeddings.json", logger)
        else:
            logger.info("Embeddings saved locally only")

    except Exception as e:
        logger.error(f"Error in embedding pipeline: {str(e)}")
        raise

def run_kg_pipeline(config: Dict[str, Any]) -> None:
    """Run knowledge graph building pipeline (S3 → local → S3)"""
    logger = logging.getLogger("main")
    logger.info("Starting knowledge graph building pipeline")

    if S3_BUCKET:
        out_prefix = f"{S3_PREFIX.rstrip('/')}/output/"
        s3_sync_down(S3_BUCKET, out_prefix, "./data", logger)  # baixa chunks.json, etc.
    else:
        logger.warning("S3_BUCKET not set — using local ./data for artifacts.")

    try:
        import json
        with open("./data/chunks.json", 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        kg_extractor = KGExtractor(config.get("kg_extraction", {}))
        kg_store = KGStore(config.get("kg_store", {}))
        processed_chunks = kg_extractor.process_chunks_for_kg(chunks_data)

        all_triples = []
        for chunk in processed_chunks:
            for triple in chunk.get("extracted_triples", []):
                all_triples.append((triple["subject"], triple["relation"], triple["object"], triple["confidence"]))
        kg_store.add_triples(all_triples)
        kg_store.save_graph()
        logger.info(f"Knowledge graph building completed with {len(all_triples)} triples")

        # sobe grafo
        if S3_BUCKET:
            kg_file = "./data/kg.graph"
            if os.path.exists(kg_file):
                s3_upload_file(S3_BUCKET, kg_file, f"{out_prefix}kg.graph", logger)
        else:
            logger.info("Knowledge graph saved locally only")

    except Exception as e:
        logger.error(f"Error in KG pipeline: {str(e)}")
        raise

def run_serve(config: Dict[str, Any]) -> None:
    """Run the FastAPI server (faz um sync down antes)"""
    logger = logging.getLogger("main")
    logger.info("Starting FastAPI server")

    try:
        # trás artefatos necessários para servir (vectorstore/chunks/etc.)
        if S3_BUCKET:
            out_prefix = f"{S3_PREFIX.rstrip('/')}/output/"
            s3_sync_down(S3_BUCKET, out_prefix, "./data", logger)
        else:
            logger.warning("S3_BUCKET not set — running with local ./data artifacts. For production set S3_BUCKET and S3_PREFIX.")

        import uvicorn
        from modules.api.app import app

        api_config = config.get("api", {})
        host = api_config.get("host", "0.0.0.0")
        port = api_config.get("port", 8000)
        debug = api_config.get("debug", True)

        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run("modules.api.app:app", host=host, port=port, reload=debug, log_level="info")

    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

def run_interactive_chat(config: Dict[str, Any]) -> None:
    """Run interactive chat interface (faz um sync down antes)"""
    logger = logging.getLogger("main")
    logger.info("Starting interactive chat interface")
    try:
        if S3_BUCKET:
            out_prefix = f"{S3_PREFIX.rstrip('/')}/output/"
            s3_sync_down(S3_BUCKET, out_prefix, "./data", logger)
        else:
            logger.warning("S3_BUCKET not set — using local ./data for artifacts.")

        vectorstore = VectorStore(**config.get("vectorstore", {}))
        kg_store = KGStore(config.get("kg_store", {}))
        mission_engine = MissionEngine(config.get("mission_engine", {}), vectorstore, kg_store)
        adaptive_generator = AdaptiveGenerator(config.get("adaptive_generator", {}))

        print("=== NASA Space Apps Hackathon MVP - Space Mission Knowledge Engine ===")
        print("Ask questions about space biology research, missions, and experiments.")
        print("Type 'exit' to quit.\n")

        while True:
            try:
                user_input = input("Your question: ").strip()
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break
                if not user_input:
                    continue

                print("Processing your question...")
                relevant_chunks = mission_engine._retrieve_relevant_chunks(user_input)
                response = adaptive_generator.generate_adaptive_response(user_input, relevant_chunks)

                if response.get("success"):
                    print(f"\nAnswer:\n{response['answer']}")
                    if response.get("sources"):
                        print(f"\nSources: {len(response['sources'])} documents referenced")
                else:
                    print(f"\nError: {response.get('error', 'Unknown error')}")
                print("\n" + "="*50 + "\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in interactive chat: {str(e)}")
        raise

# ================== CLI ==================
def main():
    """Main CLI orchestrator"""
    parser = argparse.ArgumentParser(
        description="NASA Space Apps Hackathon MVP - Space Mission Knowledge Engine (S3-enabled)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode ingest          # S3→local, processa PDFs e atualiza base (e sobe artefatos p/ S3)
  python main.py --mode embed           # Gera embeddings a partir dos chunks (S3→local→S3)
  python main.py --mode build_kg        # Reconstrói o grafo (S3→local→S3)
  python main.py --mode serve           # Sobe FastAPI (faz sync down antes)
  python main.py --mode chat            # Interface de chat (faz sync down antes)
        """
    )

    parser.add_argument("--mode", choices=["ingest", "embed", "build_kg", "serve", "chat"], required=True,
                        help="Operation mode to run")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    config = load_config()
    # Defaults de logging se não existirem
    config.setdefault("logging", {"level": "INFO", "file": "./logs/space_mission_engine.log"})

    if args.verbose:
        config["logging"]["level"] = "DEBUG"
    setup_logging(config)

    logger = logging.getLogger("main")
    logger.info(f"Starting Space Mission Engine in {args.mode} mode")

    try:
        if args.mode == "ingest":
            run_ingest_pipeline(config)
        elif args.mode == "embed":
            run_embed_pipeline(config)
        elif args.mode == "build_kg":
            run_kg_pipeline(config)
        elif args.mode == "serve":
            run_serve(config)
        elif args.mode == "chat":
            run_interactive_chat(config)
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
FastAPI Application for NASA Space Apps Hackathon MVP
Main API endpoints for the Space Mission Knowledge Engine
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our modules
from modules.pdf_processing import PDFProcessor
from modules.chunking import TextChunker
from modules.embeddings import EmbeddingGenerator
from modules.vectorstore import VectorStore
from modules.kg_extraction import KGExtractor
from modules.kg_store import KGStore
from modules.mission_engine import MissionEngine
from modules.adaptive_generator import AdaptiveGenerator
from modules.prompt_engineering import PromptEngineer

logger = logging.getLogger("api")

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    user_profile: Optional[str] = None
    max_results: Optional[int] = 5

class IngestResponse(BaseModel):
    success: bool
    message: str
    processed_files: Optional[int] = None
    errors: Optional[List[str]] = None

class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    statistics: Dict[str, Any]

# Initialize FastAPI app
app = FastAPI(
    title="NASA Space Apps Hackathon MVP - Space Mission Knowledge Engine",
    description="Intelligent knowledge engine for space biology research",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - configurable via environment
import os
_allowed = os.getenv("CORS_ALLOW_ORIGINS", "*")
if _allowed == "*" or _allowed == "":
    origins = ["*"]
else:
    origins = [o.strip() for o in _allowed.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (will be initialized in startup)
pdf_processor = None
text_chunker = None
embedding_generator = None
vectorstore = None
kg_extractor = None
kg_store = None
mission_engine = None
adaptive_generator = None
prompt_engineer = None

@app.on_event("startup")
async def startup_event():
    """Lightweight startup - components will be loaded lazily"""
    logger.info("API server starting up - components will be loaded on demand")

def load_config() -> Dict:
    """Load configuration from YAML file"""
    try:
        import yaml
        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            logger.warning("Config file not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def load_vectorstore_if_present(logger=None):
    """Lazy load vectorstore from local data if available"""
    global vectorstore
    if vectorstore is not None:
        return vectorstore
    
    data_path = os.path.join(os.getcwd(), "data", "vectorstore")
    if os.path.exists(data_path):
        try:
            # Try to load existing vectorstore
            config = load_config()
            vectorstore = VectorStore(**config.get("vectorstore", {}))
            if logger:
                logger.info(f"Loaded vectorstore from {data_path}")
        except Exception as e:
            if logger:
                logger.exception("Failed to load vectorstore from disk")
            vectorstore = None
    else:
        if logger:
            logger.warning("No local vectorstore at ./data/vectorstore. API will run but retrieval will be disabled.")
    return vectorstore

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NASA Space Apps Hackathon MVP - Space Mission Knowledge Engine",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "ingest": "/ingest",
            "query": "/query",
            "graph": "/graph"
        }
    }

@app.get("/health")
async def health():
    """Lightweight health check endpoint"""
    return {"status": "ok", "ready": True}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdfs(background_tasks: BackgroundTasks):
    """Trigger PDF ingestion, chunking, and embedding process"""
    try:
        # Check if ingestion is disabled via environment
        if os.getenv("DISABLE_INGESTION", "false").lower() == "true":
            raise HTTPException(status_code=403, detail="Ingestion is disabled in this environment")
        
        # Initialize components if needed
        global pdf_processor, text_chunker, embedding_generator, vectorstore, kg_extractor, kg_store
        if not pdf_processor:
            config = load_config()
            pdf_processor = PDFProcessor(config.get("pdf_processing", {}))
            text_chunker = TextChunker(config.get("chunking", {}))
            embedding_generator = EmbeddingGenerator(config.get("embeddings", {}))
            vectorstore = VectorStore(**config.get("vectorstore", {}))
            kg_extractor = KGExtractor(config.get("kg_extraction", {}))
            kg_store = KGStore(config.get("kg_store", {}))
        
        # Start background processing
        background_tasks.add_task(process_documents)
        
        return IngestResponse(
            success=True,
            message="Document processing started in background",
            processed_files=0  # Will be updated when processing completes
        )
        
    except Exception as e:
        logger.error(f"Error starting ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_documents():
    """Background task to process documents"""
    try:
        logger.info("Starting document processing pipeline")
        
        # Step 1: Process PDFs
        if pdf_processor:
            pdf_results = pdf_processor.process_all_pdfs()
            logger.info(f"PDF processing completed: {pdf_results['successful']} successful")
        
        # Step 2: Chunk documents
        if text_chunker:
            chunk_results = text_chunker.chunk_all_documents(
                "./data/processed", 
                "./data/chunks.json"
            )
            logger.info(f"Chunking completed: {chunk_results['total_chunks']} chunks created")
        
        # Step 3: Generate embeddings
        if embedding_generator:
            embedding_results = embedding_generator.batch_process_chunks(
                "./data/chunks.json",
                "./data/embeddings.json"
            )
            logger.info(f"Embeddings completed: {embedding_results.get('embedded_chunks', 0)} embedded")
        
        # Step 4: Store in vectorstore
        if vectorstore:
            # Load embeddings and add to vectorstore
            with open("./data/embeddings.json", 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            vectorstore.add_embeddings(embeddings_data)
            logger.info("Embeddings stored in vectorstore")
        
        # Step 5: Extract knowledge graph
        if kg_extractor and kg_store:
            # Load chunks and extract triples
            with open("./data/chunks.json", 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            processed_chunks = kg_extractor.process_chunks_for_kg(chunks_data)
            
            # Extract all triples
            all_triples = []
            for chunk in processed_chunks:
                for triple in chunk.get("extracted_triples", []):
                    all_triples.append((
                        triple["subject"],
                        triple["relation"],
                        triple["object"],
                        triple["confidence"]
                    ))
            
            # Add to knowledge graph
            kg_store.add_triples(all_triples)
            kg_store.save_graph()
            logger.info(f"Knowledge graph updated with {len(all_triples)} triples")
        
        logger.info("Document processing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")

@app.get("/query")
async def query_mission_insights(request: QueryRequest):
    """Query the mission insights using RAG + KG"""
    try:
        if not mission_engine:
            raise HTTPException(status_code=500, detail="Mission engine not initialized")
        
        # Process query
        response = mission_engine.mission_insight_query(
            request.query,
            {"user_profile": request.user_profile}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query-adaptive")
async def query_adaptive(request: QueryRequest):
    """Query with adaptive tone generation"""
    try:
        if not mission_engine or not adaptive_generator:
            raise HTTPException(status_code=500, detail="Required components not initialized")
        
        # Get relevant chunks
        relevant_chunks = mission_engine._retrieve_relevant_chunks(request.query)
        
        # Generate adaptive response
        response = adaptive_generator.generate_adaptive_response(
            request.query,
            relevant_chunks,
            request.user_profile
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing adaptive query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph", response_model=GraphResponse)
async def get_graph_data():
    """Get knowledge graph data for visualization"""
    try:
        if not kg_store:
            raise HTTPException(status_code=500, detail="Knowledge graph store not initialized")
        
        # Export graph for visualization
        viz_file = "./data/kg_visualization.json"
        kg_store.export_for_visualization(viz_file)
        
        # Load visualization data
        with open(viz_file, 'r', encoding='utf-8') as f:
            viz_data = json.load(f)
        
        # Get graph statistics
        stats = kg_store.get_graph_statistics()
        
        return GraphResponse(
            nodes=viz_data.get("nodes", []),
            edges=viz_data.get("links", []),
            statistics=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting graph data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/entities/{entity}")
async def get_entity_relations(entity: str, max_depth: int = 2):
    """Get relations for a specific entity"""
    try:
        if not kg_store:
            raise HTTPException(status_code=500, detail="Knowledge graph store not initialized")
        
        # Get entity relations
        relations = kg_store.query_graph(entity, max_depth)
        
        return {
            "entity": entity,
            "relations": relations,
            "count": len(relations)
        }
        
    except Exception as e:
        logger.error(f"Error getting entity relations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/central")
async def get_central_entities(top_k: int = 10):
    """Get most central entities in the knowledge graph"""
    try:
        if not kg_store:
            raise HTTPException(status_code=500, detail="Knowledge graph store not initialized")
        
        central_entities = kg_store.get_central_entities(top_k)
        
        return {
            "central_entities": central_entities,
            "count": len(central_entities)
        }
        
    except Exception as e:
        logger.error(f"Error getting central entities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_statistics():
    """Get system statistics"""
    try:
        stats = {
            "api": "running",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Add component statistics
        if vectorstore:
            stats["vectorstore"] = vectorstore.get_stats()
        
        if kg_store:
            stats["knowledge_graph"] = kg_store.get_graph_statistics()
        
        if mission_engine:
            stats["mission_engine"] = mission_engine.get_mission_statistics()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profiles")
async def get_user_profiles():
    """Get available user profiles for adaptive generation"""
    try:
        if not adaptive_generator:
            raise HTTPException(status_code=500, detail="Adaptive generator not initialized")
        
        profiles = adaptive_generator.get_available_profiles()
        
        return {
            "available_profiles": profiles,
            "count": len(profiles)
        }
        
    except Exception as e:
        logger.error(f"Error getting user profiles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/profiles/detect")
async def detect_user_profile(request: QueryRequest):
    """Detect user profile from query"""
    try:
        if not adaptive_generator:
            raise HTTPException(status_code=500, detail="Adaptive generator not initialized")
        
        profile = adaptive_generator.detect_user_profile(request.query)
        
        return {
            "query": request.query,
            "detected_profile": profile,
            "confidence": "high"  # Could be enhanced with confidence scoring
        }
        
    except Exception as e:
        logger.error(f"Error detecting user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Prompt Engineering Endpoints

@app.get("/query-structured")
async def query_structured(request: QueryRequest):
    """Query with structured response including references and theme"""
    try:
        if not mission_engine or not adaptive_generator:
            raise HTTPException(status_code=500, detail="Required components not initialized")
        
        # Get relevant chunks
        relevant_chunks = mission_engine._retrieve_relevant_chunks(request.query)
        
        # Generate structured response with prompt engineering
        response = adaptive_generator.generate_adaptive_response(
            request.query,
            relevant_chunks,
            request.user_profile
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing structured query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/themes")
async def get_available_themes():
    """Get available scientific themes for classification"""
    try:
        if not prompt_engineer:
            raise HTTPException(status_code=500, detail="Prompt engineer not initialized")
        
        return {
            "themes": prompt_engineer.themes,
            "count": len(prompt_engineer.themes)
        }
        
    except Exception as e:
        logger.error(f"Error getting themes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-theme")
async def classify_theme(request: QueryRequest):
    """Classify the theme of a query"""
    try:
        if not prompt_engineer:
            raise HTTPException(status_code=500, detail="Prompt engineer not initialized")
        
        # Simple theme classification based on keywords
        theme = prompt_engineer._classify_theme_from_content(request.query)
        
        return {
            "query": request.query,
            "classified_theme": theme,
            "available_themes": prompt_engineer.themes
        }
        
    except Exception as e:
        logger.error(f"Error classifying theme: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompt-engineering/stats")
async def get_prompt_engineering_stats():
    """Get statistics about prompt engineering responses"""
    try:
        if not prompt_engineer:
            raise HTTPException(status_code=500, detail="Prompt engineer not initialized")
        
        # This would typically come from a database or cache
        # For now, return basic configuration info
        return {
            "reference_extraction": prompt_engineer.reference_extraction,
            "theme_classification": prompt_engineer.theme_classification,
            "structured_output": prompt_engineer.structured_output,
            "available_themes": prompt_engineer.themes
        }
        
    except Exception as e:
        logger.error(f"Error getting prompt engineering stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the application
    uvicorn.run(
        "modules.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

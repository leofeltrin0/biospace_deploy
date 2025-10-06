"""
Vector Store Module for NASA Space Apps Hackathon MVP
Abstraction layer for storing and retrieving embeddings with multiple backend support
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

# FAISS imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# ChromaDB imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

logger = logging.getLogger("vectorstore")


class VectorStore:
    """
    Unified interface for vector storage and retrieval with multiple backend support
    """
    
    def __init__(self, backend: str = "faiss", persist_dir: str = "./data/vectorstore", **kwargs):
        self.backend = backend.lower()
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Backend-specific initialization
        if self.backend == "faiss":
            self._init_faiss(**kwargs)
        elif self.backend == "chroma":
            self._init_chroma(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        logger.info(f"VectorStore initialized with backend: {self.backend}")
    
    def _init_faiss(self, **kwargs):
        """Initialize FAISS backend"""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.index = None
        self.index_file = self.persist_dir / "faiss_index.bin"
        self.metadata_file = self.persist_dir / "metadata.json"
        self.embeddings_file = self.persist_dir / "embeddings.npy"
        
        # Load existing index if available
        if self.index_file.exists():
            self._load_faiss_index()
    
    def _init_chroma(self, **kwargs):
        """Initialize ChromaDB backend"""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.collection_name = kwargs.get("collection_name", "space_biology_docs")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Space biology research documents"}
            )
    
    def add_embeddings(self, embeddings: List[Dict]) -> bool:
        """
        Add embeddings to the vector store
        
        Args:
            embeddings: List of chunk dictionaries with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.backend == "faiss":
                return self._add_embeddings_faiss(embeddings)
            elif self.backend == "chroma":
                return self._add_embeddings_chroma(embeddings)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
                
        except Exception as e:
            logger.error(f"Error adding embeddings: {str(e)}")
            return False
    
    def _add_embeddings_faiss(self, embeddings: List[Dict]) -> bool:
        """Add embeddings to FAISS index"""
        try:
            # Extract embeddings and metadata
            embedding_vectors = []
            metadata_list = []
            
            for chunk in embeddings:
                if "embedding" in chunk:
                    embedding_vectors.append(chunk["embedding"])
                    metadata_list.append({
                        "chunk_id": chunk.get("chunk_id", ""),
                        "document_id": chunk.get("document_id", ""),
                        "text": chunk.get("text", ""),
                        "metadata": chunk.get("metadata", {})
                    })
            
            if not embedding_vectors:
                logger.warning("No embeddings found in data")
                return False
            
            # Convert to numpy array
            embeddings_array = np.array(embedding_vectors, dtype=np.float32)
            
            # Create or update FAISS index
            if self.index is None:
                # Create new index
                dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Add embeddings to index
            self.index.add(embeddings_array)
            
            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            np.save(self.embeddings_file, embeddings_array)
            
            # Save index
            faiss.write_index(self.index, str(self.index_file))
            
            logger.info(f"Added {len(embedding_vectors)} embeddings to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS: {str(e)}")
            return False
    
    def _add_embeddings_chroma(self, embeddings: List[Dict]) -> bool:
        """Add embeddings to ChromaDB"""
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings_list = []
            metadatas = []
            documents = []
            
            for chunk in embeddings:
                if "embedding" in chunk:
                    ids.append(chunk.get("chunk_id", f"chunk_{len(ids)}"))
                    embeddings_list.append(chunk["embedding"])
                    documents.append(chunk.get("text", ""))
                    metadatas.append({
                        "document_id": chunk.get("document_id", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                        **chunk.get("metadata", {})
                    })
            
            if not embeddings_list:
                logger.warning("No embeddings found in data")
                return False
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Added {len(embeddings_list)} embeddings to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings to ChromaDB: {str(e)}")
            return False
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> List[Dict]:
        """
        Query the vector store for similar documents
        
        Args:
            query_text: Query text
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            if self.backend == "faiss":
                return self._query_faiss(query_text, top_k, **kwargs)
            elif self.backend == "chroma":
                return self._query_chroma(query_text, top_k, **kwargs)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
                
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            return []
    
    def _query_faiss(self, query_text: str, top_k: int, query_embedding: np.ndarray = None) -> List[Dict]:
        """Query FAISS index"""
        try:
            if self.index is None:
                logger.warning("FAISS index not loaded")
                return []
            
            if query_embedding is None:
                logger.error("Query embedding required for FAISS backend")
                return []
            
            # Ensure query embedding is normalized
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
            else:
                metadata_list = []
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(metadata_list):
                    result = {
                        "chunk_id": metadata_list[idx].get("chunk_id", ""),
                        "document_id": metadata_list[idx].get("document_id", ""),
                        "text": metadata_list[idx].get("text", ""),
                        "similarity_score": float(score),
                        "metadata": metadata_list[idx].get("metadata", {})
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying FAISS: {str(e)}")
            return []
    
    def _query_chroma(self, query_text: str, top_k: int, query_embedding: List[float] = None) -> List[Dict]:
        """Query ChromaDB collection"""
        try:
            if query_embedding is not None:
                # Use provided embedding
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
            else:
                # Use text query (ChromaDB will generate embedding)
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=top_k
                )
            
            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    result = {
                        "chunk_id": results["ids"][0][i],
                        "document_id": results["metadatas"][0][i].get("document_id", ""),
                        "text": results["documents"][0][i],
                        "similarity_score": results["distances"][0][i] if "distances" in results else 0.0,
                        "metadata": results["metadatas"][0][i]
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {str(e)}")
            return []
    
    def save(self) -> bool:
        """
        Save the vector store to disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.backend == "faiss":
                return self._save_faiss()
            elif self.backend == "chroma":
                return self._save_chroma()
            else:
                logger.warning(f"Save not implemented for backend: {self.backend}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def _save_faiss(self) -> bool:
        """Save FAISS index"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_file))
                logger.info("FAISS index saved successfully")
                return True
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            return False
    
    def _save_chroma(self) -> bool:
        """Save ChromaDB collection (auto-saved)"""
        try:
            # ChromaDB auto-saves, just verify collection exists
            if self.collection is not None:
                logger.info("ChromaDB collection auto-saved")
                return True
            return True
        except Exception as e:
            logger.error(f"Error saving ChromaDB: {str(e)}")
            return False
    
    def load(self) -> bool:
        """
        Load the vector store from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.backend == "faiss":
                return self._load_faiss_index()
            elif self.backend == "chroma":
                return self._load_chroma_collection()
            else:
                logger.warning(f"Load not implemented for backend: {self.backend}")
                return True
                
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def _load_faiss_index(self) -> bool:
        """Load FAISS index from disk"""
        try:
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                logger.info("FAISS index loaded successfully")
                return True
            else:
                logger.info("No existing FAISS index found")
                return True
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            return False
    
    def _load_chroma_collection(self) -> bool:
        """Load ChromaDB collection"""
        try:
            # ChromaDB collections are auto-loaded
            if self.collection is not None:
                logger.info("ChromaDB collection loaded successfully")
                return True
            return True
        except Exception as e:
            logger.error(f"Error loading ChromaDB collection: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store
        
        Returns:
            Statistics dictionary
        """
        try:
            if self.backend == "faiss":
                return self._get_faiss_stats()
            elif self.backend == "chroma":
                return self._get_chroma_stats()
            else:
                return {"backend": self.backend, "status": "unknown"}
                
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}
    
    def _get_faiss_stats(self) -> Dict:
        """Get FAISS statistics"""
        stats = {
            "backend": "faiss",
            "total_vectors": 0,
            "dimension": 0,
            "index_type": "unknown"
        }
        
        try:
            if self.index is not None:
                stats["total_vectors"] = self.index.ntotal
                stats["dimension"] = self.index.d
                stats["index_type"] = type(self.index).__name__
            
            # Check metadata file
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
                stats["metadata_entries"] = len(metadata_list)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting FAISS stats: {str(e)}")
            return stats
    
    def _get_chroma_stats(self) -> Dict:
        """Get ChromaDB statistics"""
        stats = {
            "backend": "chroma",
            "collection_name": self.collection_name,
            "total_documents": 0
        }
        
        try:
            if self.collection is not None:
                count = self.collection.count()
                stats["total_documents"] = count
                stats["collection_exists"] = True
            else:
                stats["collection_exists"] = False
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting ChromaDB stats: {str(e)}")
            return stats
    
    def clear(self) -> bool:
        """
        Clear all data from the vector store
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.backend == "faiss":
                return self._clear_faiss()
            elif self.backend == "chroma":
                return self._clear_chroma()
            else:
                logger.warning(f"Clear not implemented for backend: {self.backend}")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def _clear_faiss(self) -> bool:
        """Clear FAISS index"""
        try:
            self.index = None
            if self.index_file.exists():
                self.index_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            if self.embeddings_file.exists():
                self.embeddings_file.unlink()
            
            logger.info("FAISS index cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing FAISS: {str(e)}")
            return False
    
    def _clear_chroma(self) -> bool:
        """Clear ChromaDB collection"""
        try:
            if self.collection is not None:
                # Delete and recreate collection
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Space biology research documents"}
                )
                logger.info("ChromaDB collection cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing ChromaDB: {str(e)}")
            return False

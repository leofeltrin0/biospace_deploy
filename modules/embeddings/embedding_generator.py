"""
Embeddings Module for NASA Space Apps Hackathon MVP
Handles dense vector representations of text chunks using HuggingFace models
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

logger = logging.getLogger("embeddings")


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using HuggingFace models
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get("model_name", "BAAI/bge-small-en-v1.5")
        self.cache_dir = config.get("cache_dir", "./data/embeddings_cache")
        self.batch_size = config.get("batch_size", 32)
        self.max_length = config.get("max_length", 512)
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"EmbeddingGenerator initialized with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_embedding_model(self, model_name: str = None) -> None:
        """
        Load the embedding model and tokenizer
        
        Args:
            model_name: Name of the model to load (optional)
        """
        if model_name:
            self.model_name = model_name
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Try to load as SentenceTransformer first (preferred for embeddings)
            try:
                self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
                self.model.to(self.device)
                logger.info("Loaded as SentenceTransformer model")
            except Exception as e:
                logger.warning(f"Failed to load as SentenceTransformer: {e}")
                
                # Fallback to transformers library
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    cache_dir=self.cache_dir
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name, 
                    cache_dir=self.cache_dir
                )
                self.model.to(self.device)
                logger.info("Loaded as transformers model")
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def embed_texts_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using SentenceTransformer
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            # Generate embeddings in batches
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Generate embeddings for batch
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                
                # Convert to numpy and normalize
                batch_embeddings = batch_embeddings.cpu().numpy()
                batch_embeddings = self._normalize_embeddings(batch_embeddings)
                
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with SentenceTransformer: {str(e)}")
            raise
    
    def embed_texts_transformers(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using transformers library
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling of last hidden states
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings = embeddings.cpu().numpy()
                    embeddings = self._normalize_embeddings(embeddings)
                    all_embeddings.append(embeddings)
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with transformers: {str(e)}")
            raise
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit vectors
        
        Args:
            embeddings: Raw embeddings array
            
        Returns:
            Normalized embeddings
        """
        # L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for a list of text chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            return []
        
        # Load model if not already loaded
        if self.model is None:
            self.load_embedding_model()
        
        try:
            # Extract texts from chunks
            texts = [chunk.get("text", "") for chunk in chunks]
            
            # Filter out empty texts
            valid_indices = [i for i, text in enumerate(texts) if text.strip()]
            valid_texts = [texts[i] for i in valid_indices]
            
            if not valid_texts:
                logger.warning("No valid texts found in chunks")
                return chunks
            
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            
            # Generate embeddings
            if isinstance(self.model, SentenceTransformer):
                embeddings = self.embed_texts_sentence_transformer(valid_texts)
            else:
                embeddings = self.embed_texts_transformers(valid_texts)
            
            # Add embeddings to chunks
            result_chunks = []
            embedding_idx = 0
            
            for i, chunk in enumerate(chunks):
                if i in valid_indices:
                    chunk_copy = chunk.copy()
                    chunk_copy["embedding"] = embeddings[embedding_idx].tolist()
                    chunk_copy["embedding_dim"] = len(embeddings[embedding_idx])
                    result_chunks.append(chunk_copy)
                    embedding_idx += 1
                else:
                    # Keep chunk without embedding
                    result_chunks.append(chunk)
            
            logger.info(f"Generated embeddings for {len(result_chunks)} chunks")
            return result_chunks
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return chunks
    
    def embed_single_text(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            return None
        
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_embedding_model()
            
            if isinstance(self.model, SentenceTransformer):
                embedding = self.model.encode([text], convert_to_tensor=True)
                embedding = embedding.cpu().numpy()[0]
            else:
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1)
                    embedding = embedding.cpu().numpy()[0]
            
            # Normalize embedding
            embedding = self._normalize_embeddings(embedding.reshape(1, -1))[0]
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for single text: {str(e)}")
            return None
    
    def save_embeddings(self, embeddings_data: List[Dict], path: str) -> bool:
        """
        Save embeddings to file
        
        Args:
            embeddings_data: List of chunks with embeddings
            path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON (embeddings as lists)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved embeddings to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            return False
    
    def load_embeddings(self, path: str) -> List[Dict]:
        """
        Load embeddings from file
        
        Args:
            path: Path to embeddings file
            
        Returns:
            List of chunks with embeddings
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            
            logger.info(f"Loaded embeddings from {path}")
            return embeddings_data
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            return []
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Ensure embeddings are normalized
            emb1 = embedding1 / np.linalg.norm(embedding1)
            emb2 = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(emb1, emb2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def find_similar_chunks(self, query_embedding: np.ndarray, 
                          chunk_embeddings: List[Dict], 
                          top_k: int = 5) -> List[Dict]:
        """
        Find most similar chunks to a query embedding
        
        Args:
            query_embedding: Query embedding vector
            chunk_embeddings: List of chunks with embeddings
            top_k: Number of top similar chunks to return
            
        Returns:
            List of similar chunks with similarity scores
        """
        try:
            similarities = []
            
            for chunk in chunk_embeddings:
                if "embedding" in chunk:
                    chunk_emb = np.array(chunk["embedding"])
                    similarity = self.compute_similarity(query_embedding, chunk_emb)
                    similarities.append((chunk, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results
            results = []
            for chunk, similarity in similarities[:top_k]:
                result_chunk = chunk.copy()
                result_chunk["similarity_score"] = similarity
                results.append(result_chunk)
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {str(e)}")
            return []
    
    def get_embedding_statistics(self, embeddings_data: List[Dict]) -> Dict:
        """
        Get statistics about embeddings
        
        Args:
            embeddings_data: List of chunks with embeddings
            
        Returns:
            Statistics dictionary
        """
        if not embeddings_data:
            return {"total_embeddings": 0}
        
        # Count embeddings
        embeddings_with_vectors = [chunk for chunk in embeddings_data if "embedding" in chunk]
        
        if not embeddings_with_vectors:
            return {"total_embeddings": 0}
        
        # Get embedding dimensions
        embedding_dims = [len(chunk["embedding"]) for chunk in embeddings_with_vectors]
        
        stats = {
            "total_chunks": len(embeddings_data),
            "total_embeddings": len(embeddings_with_vectors),
            "embedding_dimension": embedding_dims[0] if embedding_dims else 0,
            "coverage_percentage": (len(embeddings_with_vectors) / len(embeddings_data)) * 100
        }
        
        return stats
    
    def batch_process_chunks(self, chunks_file: str, output_file: str) -> Dict:
        """
        Process chunks file and generate embeddings in batches
        
        Args:
            chunks_file: Path to chunks JSON file
            output_file: Path to save embeddings
            
        Returns:
            Processing results
        """
        try:
            # Load chunks
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            logger.info(f"Processing {len(chunks)} chunks for embeddings")
            
            # Generate embeddings
            embedded_chunks = self.embed_chunks(chunks)
            
            # Save embeddings
            success = self.save_embeddings(embedded_chunks, output_file)
            
            if success:
                stats = self.get_embedding_statistics(embedded_chunks)
                logger.info(f"Batch processing completed successfully")
                return {
                    "success": True,
                    "total_chunks": len(chunks),
                    "embedded_chunks": len([c for c in embedded_chunks if "embedding" in c]),
                    "output_file": output_file,
                    "statistics": stats
                }
            else:
                return {"success": False, "error": "Failed to save embeddings"}
                
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return {"success": False, "error": str(e)}

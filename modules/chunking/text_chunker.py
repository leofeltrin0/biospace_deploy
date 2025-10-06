"""
Text Chunking Module for NASA Space Apps Hackathon MVP
Handles semantic text splitting for embedding and retrieval
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken

logger = logging.getLogger("chunking")


class TextChunker:
    """
    Handles text chunking for space biology documents with semantic awareness
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.chunk_size = config.get("chunk_size", 800)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        self.separators = config.get("separators", ["\n\n", "\n", ".", "!", "?", ";", " "])
        self.metadata_fields = config.get("metadata_fields", ["document_id", "chunk_index", "source"])
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except Exception as e:
            logger.warning(f"Could not load tiktoken tokenizer: {e}. Using character count.")
            self.tokenizer = None
        
        logger.info(f"TextChunker initialized with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken or fallback to character count
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed: {e}. Using character count.")
        
        # Fallback to character count (rough approximation)
        return len(text) // 4
    
    def chunk_document(self, text: str, document_id: str, metadata: Dict = None) -> List[Dict]:
        """
        Split a document into semantic chunks
        
        Args:
            text: Input text to chunk
            document_id: Unique document identifier
            metadata: Additional document metadata
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for document {document_id}")
            return []
        
        try:
            # Create LangChain document
            doc = Document(
                page_content=text,
                metadata=metadata or {}
            )
            
            # Split the document
            chunks = self.text_splitter.split_documents([doc])
            
            # Convert to our format and add metadata
            processed_chunks = []
            
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk.page_content,
                    "text_length": len(chunk.page_content),
                    "token_count": self.count_tokens(chunk.page_content),
                    "metadata": {
                        **chunk.metadata,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "total_chunks": len(chunks)
                    }
                }
                
                processed_chunks.append(chunk_data)
            
            logger.info(f"Created {len(processed_chunks)} chunks for document {document_id}")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error chunking document {document_id}: {str(e)}")
            return []
    
    def chunk_text_with_semantic_boundaries(self, text: str, document_id: str) -> List[Dict]:
        """
        Advanced chunking that respects semantic boundaries for space biology content
        
        Args:
            text: Input text
            document_id: Document identifier
            
        Returns:
            List of semantically-aware chunks
        """
        if not text or not text.strip():
            return []
        
        # Define semantic boundaries for scientific papers
        semantic_boundaries = [
            r'\n\nAbstract\s*:',  # Abstract sections
            r'\n\nIntroduction\s*:',  # Introduction
            r'\n\nMethods?\s*:',  # Methods
            r'\n\nResults?\s*:',  # Results
            r'\n\nDiscussion\s*:',  # Discussion
            r'\n\nConclusion\s*:',  # Conclusion
            r'\n\nReferences\s*:',  # References
            r'\n\n[A-Z][a-z]+\s+[A-Z][a-z]+:',  # Author names
            r'\n\n\d+\.\s+[A-Z]',  # Numbered sections
        ]
        
        try:
            # First, try to split by semantic boundaries
            sections = [text]
            for boundary in semantic_boundaries:
                new_sections = []
                for section in sections:
                    parts = re.split(boundary, section, flags=re.IGNORECASE)
                    new_sections.extend([part.strip() for part in parts if part.strip()])
                sections = new_sections
            
            # If sections are still too large, chunk them further
            all_chunks = []
            for section_idx, section in enumerate(sections):
                if len(section) <= self.chunk_size:
                    # Section is small enough, use as-is
                    chunk_data = {
                        "chunk_id": f"{document_id}_section_{section_idx}",
                        "document_id": document_id,
                        "chunk_index": section_idx,
                        "text": section,
                        "text_length": len(section),
                        "token_count": self.count_tokens(section),
                        "metadata": {
                            "section_index": section_idx,
                            "semantic_boundary": True,
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap
                        }
                    }
                    all_chunks.append(chunk_data)
                else:
                    # Section is too large, chunk it further
                    section_chunks = self.chunk_document(section, f"{document_id}_section_{section_idx}")
                    all_chunks.extend(section_chunks)
            
            logger.info(f"Created {len(all_chunks)} semantic chunks for document {document_id}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking for {document_id}: {str(e)}")
            # Fallback to regular chunking
            return self.chunk_document(text, document_id)
    
    def chunk_all_documents(self, input_dir: str, output_path: str) -> Dict:
        """
        Process all JSON documents in input directory and create chunks
        
        Args:
            input_dir: Directory containing processed JSON documents
            output_path: Path to save chunked data
            
        Returns:
            Processing summary
        """
        input_path = Path(input_dir)
        output_path = Path(output_path)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all JSON files
        json_files = list(input_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to chunk")
        
        results = {
            "total_documents": len(json_files),
            "total_chunks": 0,
            "successful": 0,
            "failed": 0,
            "errors": [],
            "chunk_stats": {
                "min_chunk_size": float('inf'),
                "max_chunk_size": 0,
                "avg_chunk_size": 0
            }
        }
        
        all_chunks = []
        chunk_sizes = []
        
        for json_file in json_files:
            try:
                logger.info(f"Processing {json_file.name}")
                
                # Load document
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                
                doc_id = doc_data.get("doc_id", json_file.stem)
                text = doc_data.get("text", "")
                metadata = doc_data.get("metadata", {})
                
                if not text:
                    logger.warning(f"No text content in {json_file.name}")
                    results["failed"] += 1
                    results["errors"].append({
                        "file": str(json_file),
                        "error": "No text content"
                    })
                    continue
                
                # Create chunks
                chunks = self.chunk_document(text, doc_id, metadata)
                
                if chunks:
                    results["successful"] += 1
                    results["total_chunks"] += len(chunks)
                    all_chunks.extend(chunks)
                    
                    # Collect chunk size statistics
                    for chunk in chunks:
                        chunk_size = chunk.get("text_length", 0)
                        chunk_sizes.append(chunk_size)
                        results["chunk_stats"]["min_chunk_size"] = min(
                            results["chunk_stats"]["min_chunk_size"], chunk_size
                        )
                        results["chunk_stats"]["max_chunk_size"] = max(
                            results["chunk_stats"]["max_chunk_size"], chunk_size
                        )
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "file": str(json_file),
                        "error": "No chunks created"
                    })
                
            except Exception as e:
                logger.error(f"Error processing {json_file}: {str(e)}")
                results["failed"] += 1
                results["errors"].append({
                    "file": str(json_file),
                    "error": str(e)
                })
        
        # Calculate average chunk size
        if chunk_sizes:
            results["chunk_stats"]["avg_chunk_size"] = sum(chunk_sizes) / len(chunk_sizes)
        
        # Save all chunks to output file
        if all_chunks:
            output_file = output_path / "all_chunks.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(all_chunks)} chunks to {output_file}")
            results["output_file"] = str(output_file)
        
        logger.info(f"Chunking completed: {results['successful']} successful, {results['failed']} failed")
        return results
    
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Calculate statistics for a list of chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {"total_chunks": 0}
        
        chunk_sizes = [chunk.get("text_length", 0) for chunk in chunks]
        token_counts = [chunk.get("token_count", 0) for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "total_text_length": sum(chunk_sizes),
            "total_tokens": sum(token_counts),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "avg_token_count": sum(token_counts) / len(token_counts),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts)
        }
        
        return stats
    
    def optimize_chunk_size(self, sample_texts: List[str], target_tokens: int = 800) -> Tuple[int, int]:
        """
        Optimize chunk size based on sample texts
        
        Args:
            sample_texts: List of sample text documents
            target_tokens: Target number of tokens per chunk
            
        Returns:
            Tuple of (optimal_chunk_size, optimal_overlap)
        """
        logger.info("Optimizing chunk size based on sample texts")
        
        best_size = self.chunk_size
        best_overlap = self.chunk_overlap
        best_score = float('inf')
        
        # Test different chunk sizes around the target
        test_sizes = [target_tokens - 200, target_tokens - 100, target_tokens, target_tokens + 100, target_tokens + 200]
        test_overlaps = [50, 100, 150, 200]
        
        for size in test_sizes:
            for overlap in test_overlaps:
                if overlap >= size:
                    continue
                
                # Create temporary splitter
                temp_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=size,
                    chunk_overlap=overlap,
                    separators=self.separators
                )
                
                # Test on sample texts
                total_variance = 0
                total_chunks = 0
                
                for text in sample_texts[:5]:  # Test on first 5 samples
                    try:
                        chunks = temp_splitter.split_text(text)
                        chunk_tokens = [self.count_tokens(chunk) for chunk in chunks]
                        
                        if chunk_tokens:
                            avg_tokens = sum(chunk_tokens) / len(chunk_tokens)
                            variance = sum((tokens - avg_tokens) ** 2 for tokens in chunk_tokens) / len(chunk_tokens)
                            total_variance += variance
                            total_chunks += len(chunk_tokens)
                    except Exception:
                        continue
                
                if total_chunks > 0:
                    avg_variance = total_variance / total_chunks
                    if avg_variance < best_score:
                        best_score = avg_variance
                        best_size = size
                        best_overlap = overlap
        
        logger.info(f"Optimized chunk size: {best_size}, overlap: {best_overlap}")
        return best_size, best_overlap

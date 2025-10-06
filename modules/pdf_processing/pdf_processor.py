"""
PDF Processing Module for NASA Space Apps Hackathon MVP
Handles extraction and cleaning of text from PDF documents
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF
import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger("pdf_processing")


class PDFProcessor:
    """
    Handles PDF text extraction and cleaning for space biology documents
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.input_dir = Path(config.get("input_dir", "./data"))
        self.output_dir = Path(config.get("output_dir", "./data/processed"))
        self.supported_formats = config.get("supported_formats", [".pdf"])
        self.text_cleaning_config = config.get("text_cleaning", {})
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PDFProcessor initialized with input_dir: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def extract_text_from_pdf(self, file_path: str, method: str = "pymupdf") -> str:
        """
        Extract text from PDF using specified method
        
        Args:
            file_path: Path to PDF file
            method: Extraction method ('pymupdf', 'pdfplumber', 'pypdf')
            
        Returns:
            Extracted text content
        """
        try:
            if method == "pymupdf":
                return self._extract_with_pymupdf(file_path)
            elif method == "pdfplumber":
                return self._extract_with_pdfplumber(file_path)
            elif method == "pypdf":
                return self._extract_with_pypdf(file_path)
            else:
                raise ValueError(f"Unsupported extraction method: {method}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""
    
    def _extract_with_pymupdf(self, file_path: str) -> str:
        """Extract text using PyMuPDF (fitz)"""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {file_path}: {str(e)}")
        return text
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {file_path}: {str(e)}")
        return text
    
    def _extract_with_pypdf(self, file_path: str) -> str:
        """Extract text using pypdf"""
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"pypdf extraction failed for {file_path}: {str(e)}")
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing headers, footers, and other noise
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Remove excessive whitespace
        if self.text_cleaning_config.get("normalize_whitespace", True):
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove headers and footers (common patterns in academic papers)
        if self.text_cleaning_config.get("remove_headers_footers", True):
            # Remove page numbers
            text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
            
            # Remove common header patterns
            header_patterns = [
                r'^.*\d{4}.*$',  # Years in headers
                r'^.*doi:.*$',   # DOI references
                r'^.*http.*$',   # URLs
            ]
            for pattern in header_patterns:
                text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # Remove figure captions
        if self.text_cleaning_config.get("remove_figure_captions", True):
            # Remove "Figure X:" patterns
            text = re.sub(r'Figure\s+\d+[:\-].*?(?=\n\n|\n[A-Z]|$)', '', text, flags=re.DOTALL)
            # Remove "Table X:" patterns
            text = re.sub(r'Table\s+\d+[:\-].*?(?=\n\n|\n[A-Z]|$)', '', text, flags=re.DOTALL)
        
        # Remove references section (common at end of papers)
        text = re.sub(r'References.*$', '', text, flags=re.DOTALL)
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def save_cleaned_text(self, doc_id: str, text: str, metadata: Dict = None) -> str:
        """
        Save cleaned text to output directory
        
        Args:
            doc_id: Unique document identifier
            text: Cleaned text content
            metadata: Additional document metadata
            
        Returns:
            Path to saved file
        """
        try:
            # Prepare document data
            doc_data = {
                "doc_id": doc_id,
                "text": text,
                "metadata": metadata or {},
                "text_length": len(text),
                "word_count": len(text.split())
            }
            
            # Save as JSON
            output_file = self.output_dir / f"{doc_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved cleaned text for {doc_id} to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error saving cleaned text for {doc_id}: {str(e)}")
            return ""
    
    def process_single_pdf(self, file_path: str, doc_id: str = None) -> Dict:
        """
        Process a single PDF file
        
        Args:
            file_path: Path to PDF file
            doc_id: Optional document ID (defaults to filename)
            
        Returns:
            Processing result dictionary
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return {"success": False, "error": "File not found"}
        
        if file_path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return {"success": False, "error": "Unsupported format"}
        
        if doc_id is None:
            doc_id = file_path.stem
        
        logger.info(f"Processing PDF: {file_path}")
        
        try:
            # Extract text using multiple methods for robustness
            methods = ["pymupdf", "pdfplumber", "pypdf"]
            best_text = ""
            best_length = 0
            
            for method in methods:
                try:
                    text = self.extract_text_from_pdf(str(file_path), method)
                    if len(text) > best_length:
                        best_text = text
                        best_length = len(text)
                        logger.debug(f"Method {method} extracted {len(text)} characters")
                except Exception as e:
                    logger.warning(f"Method {method} failed: {str(e)}")
                    continue
            
            if not best_text:
                return {"success": False, "error": "Failed to extract text with any method"}
            
            # Clean the text
            cleaned_text = self.clean_text(best_text)
            
            # Check minimum text length
            min_length = self.text_cleaning_config.get("min_text_length", 50)
            if len(cleaned_text) < min_length:
                logger.warning(f"Text too short for {doc_id}: {len(cleaned_text)} characters")
                return {"success": False, "error": "Text too short after cleaning"}
            
            # Prepare metadata
            metadata = {
                "source_file": str(file_path),
                "original_length": len(best_text),
                "cleaned_length": len(cleaned_text),
                "extraction_method": "multi_method",
                "processing_timestamp": str(Path(file_path).stat().st_mtime)
            }
            
            # Save cleaned text
            output_file = self.save_cleaned_text(doc_id, cleaned_text, metadata)
            
            if output_file:
                return {
                    "success": True,
                    "doc_id": doc_id,
                    "output_file": output_file,
                    "text_length": len(cleaned_text),
                    "metadata": metadata
                }
            else:
                return {"success": False, "error": "Failed to save processed text"}
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def process_all_pdfs(self) -> Dict:
        """
        Process all PDF files in the input directory
        
        Returns:
            Summary of processing results
        """
        logger.info(f"Starting batch processing of PDFs in {self.input_dir}")
        
        # Find all PDF files
        pdf_files = []
        for ext in self.supported_formats:
            pdf_files.extend(self.input_dir.glob(f"*{ext}"))
            pdf_files.extend(self.input_dir.glob(f"**/*{ext}"))  # Recursive search
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {
            "total_files": len(pdf_files),
            "successful": 0,
            "failed": 0,
            "errors": [],
            "processed_files": []
        }
        
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_file.name}")
            
            result = self.process_single_pdf(str(pdf_file))
            
            if result["success"]:
                results["successful"] += 1
                results["processed_files"].append({
                    "file": str(pdf_file),
                    "doc_id": result["doc_id"],
                    "output_file": result["output_file"],
                    "text_length": result["text_length"]
                })
            else:
                results["failed"] += 1
                results["errors"].append({
                    "file": str(pdf_file),
                    "error": result["error"]
                })
                logger.error(f"Failed to process {pdf_file}: {result['error']}")
        
        logger.info(f"Batch processing completed: {results['successful']} successful, {results['failed']} failed")
        return results
    
    def get_processing_stats(self) -> Dict:
        """
        Get statistics about processed documents
        
        Returns:
            Processing statistics
        """
        processed_files = list(self.output_dir.glob("*.json"))
        
        stats = {
            "total_processed": len(processed_files),
            "total_size_mb": 0,
            "total_text_length": 0,
            "average_text_length": 0
        }
        
        for file_path in processed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    stats["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
                    stats["total_text_length"] += data.get("text_length", 0)
            except Exception as e:
                logger.warning(f"Error reading stats from {file_path}: {str(e)}")
        
        if stats["total_processed"] > 0:
            stats["average_text_length"] = stats["total_text_length"] / stats["total_processed"]
        
        return stats

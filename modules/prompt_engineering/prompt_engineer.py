"""
Prompt Engineering Module for NASA Space Apps Hackathon MVP
Ensures every model response includes reference metadata and theme categorization
"""

import json
import logging
import re
from typing import List, Dict, Optional, Tuple, Any
from openai import OpenAI
import os

logger = logging.getLogger("prompt_engineering")


class PromptEngineer:
    """
    Advanced prompt engineering for structured responses with metadata and categorization
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.themes = [
            "biotechnology", "neuroscience", "biochemistry", 
            "ecology", "microbiology", "genetics"
        ]
        self.reference_extraction = config.get("reference_extraction", True)
        self.theme_classification = config.get("theme_classification", True)
        self.structured_output = config.get("structured_output", True)
        
        # Initialize OpenAI client
        self.openai_client = None
        self._init_openai_client()
        
        # Load system prompt template
        self.system_prompt_template = self._load_system_prompt_template()
        
        logger.info("PromptEngineer initialized")
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for prompt engineering")
            else:
                logger.warning("OpenAI API key not found for prompt engineering")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _load_system_prompt_template(self) -> str:
        """Load the system prompt template"""
        return """
You are a NASA expert assistant specialized in space biology research.
Your tasks:
1. Provide a clear and accurate answer based on retrieved scientific documents.
2. Always include the reference of the information source:
   - File name (extract from document metadata)
   - Author(s) (extract from document metadata or content)
   - Publication date (if available in document metadata or content)
3. Internally categorize both the user question and retrieved content
   into one of the following scientific domains:
   [biotechnology, neuroscience, biochemistry, ecology, microbiology, genetics].
4. Return the category as 'theme' in your final structured output.

IMPORTANT: You must respond in valid JSON format with the following structure:
{
  "answer": "<your comprehensive answer based on the retrieved documents>",
  "references": [
    {
      "file": "<filename>",
      "authors": "<author names>",
      "date": "<publication date>",
      "relevance_score": <0.0-1.0>
    }
  ],
  "theme": "<one of the 6 categories>",
  "confidence": <0.0-1.0>,
  "key_findings": ["<finding 1>", "<finding 2>", ...]
}

Guidelines:
- Extract author names from document metadata or content
- Extract publication dates from document metadata or content
- If metadata is not available, indicate "Not available"
- Assign relevance scores based on how well each reference supports the answer
- Choose the most appropriate theme based on the content focus
- Provide confidence score for your answer (0.0-1.0)
- Include 2-5 key findings from the retrieved documents
"""
    
    def build_prompt(self, user_query: str, retrieved_chunks: List[Dict], 
                    user_type: str = "scientist") -> str:
        """
        Compose the complete LLM prompt by combining:
        - SYSTEM_PROMPT_TEMPLATE
        - Audience-specific tone (from Adaptive Generator)
        - Retrieved context chunks
        
        Args:
            user_query: User's question
            retrieved_chunks: List of retrieved document chunks with metadata
            user_type: User profile type (scientist, manager, layperson)
            
        Returns:
            Complete prompt string
        """
        try:
            # Get audience-specific tone instructions
            tone_instructions = self._get_audience_tone_instructions(user_type)
            
            # Prepare context from retrieved chunks
            context = self._prepare_context_with_metadata(retrieved_chunks)
            
            # Build the complete prompt
            prompt_parts = [
                self.system_prompt_template,
                "",
                f"AUDIENCE ADAPTATION: {tone_instructions}",
                "",
                "RETRIEVED DOCUMENTS:",
                context,
                "",
                f"USER QUERY: {user_query}",
                "",
                "Please provide your response in the specified JSON format."
            ]
            
            complete_prompt = "\n".join(prompt_parts)
            
            logger.info(f"Built prompt for {user_type} user with {len(retrieved_chunks)} chunks")
            return complete_prompt
            
        except Exception as e:
            logger.error(f"Error building prompt: {str(e)}")
            return self._build_fallback_prompt(user_query, retrieved_chunks)
    
    def _get_audience_tone_instructions(self, user_type: str) -> str:
        """Get audience-specific tone instructions"""
        tone_instructions = {
            "scientist": """
            TONE: Use precise, technical language with:
            - Specific terminology and jargon appropriate for space biology
            - Detailed methodology and experimental design information
            - Statistical significance and confidence intervals where relevant
            - Citations to specific studies and data points
            - Technical depth appropriate for peer review
            - Focus on mechanisms, pathways, and biological processes
            """,
            "manager": """
            TONE: Use strategic, high-level language with:
            - Executive summaries and key findings
            - Strategic implications and research opportunities
            - Resource requirements and timeline considerations
            - Risk assessment and mitigation strategies
            - Competitive advantages and innovation potential
            - ROI and impact metrics where applicable
            """,
            "layperson": """
            TONE: Use clear, accessible language with:
            - Simple explanations of complex concepts
            - Analogies and real-world examples
            - Visual descriptions and metaphors
            - Step-by-step explanations
            - Avoid technical jargon or explain it clearly
            - Focus on practical implications and benefits
            """
        }
        
        return tone_instructions.get(user_type, tone_instructions["layperson"])
    
    def _prepare_context_with_metadata(self, retrieved_chunks: List[Dict]) -> str:
        """Prepare context with metadata for each chunk"""
        try:
            context_parts = []
            
            for i, chunk in enumerate(retrieved_chunks[:5], 1):  # Limit to top 5 chunks
                # Extract metadata
                metadata = chunk.get("metadata", {})
                document_id = chunk.get("document_id", "Unknown")
                chunk_id = chunk.get("chunk_id", f"chunk_{i}")
                
                # Try to extract author and date from metadata
                authors = metadata.get("authors", "Not available")
                date = metadata.get("publication_date", metadata.get("date", "Not available"))
                
                # Build context entry
                context_entry = f"""
Document {i}:
- File: {document_id}
- Authors: {authors}
- Date: {date}
- Chunk ID: {chunk_id}
- Content: {chunk.get('text', '')[:500]}...
- Similarity Score: {chunk.get('similarity_score', 0.0):.3f}
"""
                context_parts.append(context_entry)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error preparing context: {str(e)}")
            return "Context preparation failed."
    
    def _build_fallback_prompt(self, user_query: str, retrieved_chunks: List[Dict]) -> str:
        """Build fallback prompt if main prompt building fails"""
        context = "\n\n".join([
            f"Source: {chunk.get('document_id', 'Unknown')}\n{chunk.get('text', '')[:300]}..."
            for chunk in retrieved_chunks[:3]
        ])
        
        return f"""
You are a NASA space biology expert. Answer the user's question based on the provided research documents.

User Question: {user_query}

Research Documents:
{context}

Please provide a comprehensive answer with references to the source documents.
"""
    
    def parse_model_response(self, response_text: str) -> Dict:
        """
        Parse and validate JSON output from the LLM response.
        Ensure fields 'answer', 'references', and 'theme' are present.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Structured response dictionary
        """
        try:
            # Clean the response text
            cleaned_response = self._clean_response_text(response_text)
            
            # Try to parse as JSON
            try:
                structured_response = json.loads(cleaned_response)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                structured_response = self._extract_json_from_text(cleaned_response)
            
            # Validate and fix the response structure
            validated_response = self._validate_response_structure(structured_response)
            
            logger.info(f"Parsed response with theme: {validated_response.get('theme', 'unknown')}")
            return validated_response
            
        except Exception as e:
            logger.error(f"Error parsing model response: {str(e)}")
            return self._create_fallback_response(response_text)
    
    def _clean_response_text(self, response_text: str) -> str:
        """Clean response text for JSON parsing"""
        try:
            # Remove markdown code blocks if present
            cleaned = re.sub(r'```json\s*', '', response_text)
            cleaned = re.sub(r'```\s*', '', cleaned)
            
            # Remove any leading/trailing whitespace
            cleaned = cleaned.strip()
            
            # Try to find JSON object boundaries
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned = cleaned[start_idx:end_idx + 1]
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning response text: {str(e)}")
            return response_text
    
    def _extract_json_from_text(self, text: str) -> Dict:
        """Extract JSON from text that might contain other content"""
        try:
            # Find JSON object in the text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                raise ValueError("No JSON object found in response")
                
        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            return {}
    
    def _validate_response_structure(self, response: Dict) -> Dict:
        """Validate and fix response structure"""
        try:
            # Ensure required fields exist
            validated = {
                "answer": response.get("answer", "No answer provided"),
                "references": response.get("references", []),
                "theme": response.get("theme", "unknown"),
                "confidence": response.get("confidence", 0.5),
                "key_findings": response.get("key_findings", [])
            }
            
            # Validate theme
            if validated["theme"] not in self.themes:
                validated["theme"] = self._classify_theme_from_content(validated["answer"])
            
            # Validate references format
            if not isinstance(validated["references"], list):
                validated["references"] = []
            
            # Ensure each reference has required fields
            validated_references = []
            for ref in validated["references"]:
                if isinstance(ref, dict):
                    validated_ref = {
                        "file": ref.get("file", "Unknown"),
                        "authors": ref.get("authors", "Not available"),
                        "date": ref.get("date", "Not available"),
                        "relevance_score": ref.get("relevance_score", 0.5)
                    }
                    validated_references.append(validated_ref)
            
            validated["references"] = validated_references
            
            # Validate confidence score
            if not isinstance(validated["confidence"], (int, float)) or not 0 <= validated["confidence"] <= 1:
                validated["confidence"] = 0.5
            
            return validated
            
        except Exception as e:
            logger.error(f"Error validating response structure: {str(e)}")
            return {
                "answer": "Error processing response",
                "references": [],
                "theme": "unknown",
                "confidence": 0.0,
                "key_findings": []
            }
    
    def _classify_theme_from_content(self, content: str) -> str:
        """Classify theme from content when not provided by model"""
        try:
            content_lower = content.lower()
            
            # Theme keywords
            theme_keywords = {
                "biotechnology": ["biotech", "genetic engineering", "recombinant", "transgenic", "cloning"],
                "neuroscience": ["neural", "brain", "neuron", "synapse", "cognitive", "behavioral"],
                "biochemistry": ["protein", "enzyme", "metabolism", "biochemical", "molecular"],
                "ecology": ["ecosystem", "environment", "habitat", "population", "community"],
                "microbiology": ["microbe", "bacteria", "virus", "fungi", "microbial", "pathogen"],
                "genetics": ["gene", "genome", "dna", "rna", "chromosome", "mutation", "inheritance"]
            }
            
            # Count keyword matches for each theme
            theme_scores = {}
            for theme, keywords in theme_keywords.items():
                score = sum(1 for keyword in keywords if keyword in content_lower)
                theme_scores[theme] = score
            
            # Return theme with highest score
            if theme_scores:
                return max(theme_scores, key=theme_scores.get)
            else:
                return "biotechnology"  # Default theme
                
        except Exception as e:
            logger.error(f"Error classifying theme: {str(e)}")
            return "biotechnology"
    
    def _create_fallback_response(self, response_text: str) -> Dict:
        """Create fallback response when parsing fails"""
        return {
            "answer": response_text,
            "references": [],
            "theme": "unknown",
            "confidence": 0.3,
            "key_findings": [],
            "parsing_error": True
        }
    
    def adaptive_rag_pipeline(self, query: str, retrieved_chunks: List[Dict], 
                            user_type: str = "scientist") -> Dict:
        """
        Complete adaptive RAG pipeline with prompt engineering
        
        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks
            user_type: User profile type
            
        Returns:
            Structured response with metadata and theme
        """
        try:
            # Build the complete prompt
            prompt = self.build_prompt(query, retrieved_chunks, user_type)
            
            # Call LLM if available
            if self.openai_client:
                response = self._call_llm(prompt)
            else:
                response = self._generate_fallback_response(query, retrieved_chunks)
            
            # Parse and structure the response
            structured_output = self.parse_model_response(response)
            
            # Add metadata
            structured_output["user_type"] = user_type
            structured_output["chunks_used"] = len(retrieved_chunks)
            structured_output["processing_timestamp"] = self._get_timestamp()
            
            logger.info(f"Generated structured response for {user_type} user")
            return structured_output
            
        except Exception as e:
            logger.error(f"Error in adaptive RAG pipeline: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "references": [],
                "theme": "unknown",
                "confidence": 0.0,
                "key_findings": [],
                "error": True
            }
    
    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI LLM with the prompt"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a NASA space biology expert assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise
    
    def _generate_fallback_response(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Generate fallback response without LLM"""
        try:
            # Simple response generation
            context = "\n\n".join([
                f"Source: {chunk.get('document_id', 'Unknown')}\n{chunk.get('text', '')[:200]}..."
                for chunk in retrieved_chunks[:3]
            ])
            
            return f"""
{{
  "answer": "Based on the retrieved documents: {context[:500]}...",
  "references": [
    {{
      "file": "{retrieved_chunks[0].get('document_id', 'Unknown') if retrieved_chunks else 'Unknown'}",
      "authors": "Not available",
      "date": "Not available",
      "relevance_score": 0.7
    }}
  ],
  "theme": "biotechnology",
  "confidence": 0.6,
  "key_findings": ["Research findings from space biology studies"]
}}
"""
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {str(e)}")
            return '{"answer": "Error generating response", "references": [], "theme": "unknown", "confidence": 0.0, "key_findings": []}'
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def extract_metadata_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Extract metadata from chunks for reference building"""
        try:
            metadata_list = []
            
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                document_id = chunk.get("document_id", "Unknown")
                
                # Try to extract author and date information
                authors = self._extract_authors_from_metadata(metadata, chunk.get("text", ""))
                date = self._extract_date_from_metadata(metadata, chunk.get("text", ""))
                
                metadata_entry = {
                    "file": document_id,
                    "authors": authors,
                    "date": date,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "similarity_score": chunk.get("similarity_score", 0.0)
                }
                
                metadata_list.append(metadata_entry)
            
            return metadata_list
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return []
    
    def _extract_authors_from_metadata(self, metadata: Dict, text: str) -> str:
        """Extract author names from metadata or text"""
        try:
            # Try metadata first
            if "authors" in metadata:
                return metadata["authors"]
            
            # Try to extract from text using patterns
            author_patterns = [
                r"Authors?:\s*([^\n]+)",
                r"By\s+([^\n]+)",
                r"^([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            ]
            
            for pattern in author_patterns:
                match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            return "Not available"
            
        except Exception as e:
            logger.error(f"Error extracting authors: {str(e)}")
            return "Not available"
    
    def _extract_date_from_metadata(self, metadata: Dict, text: str) -> str:
        """Extract publication date from metadata or text"""
        try:
            # Try metadata first
            if "date" in metadata:
                return metadata["date"]
            if "publication_date" in metadata:
                return metadata["publication_date"]
            
            # Try to extract from text using patterns
            date_patterns = [
                r"(\d{4})",
                r"(\d{1,2}/\d{1,2}/\d{4})",
                r"(\d{4}-\d{2}-\d{2})",
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1)
            
            return "Not available"
            
        except Exception as e:
            logger.error(f"Error extracting date: {str(e)}")
            return "Not available"
    
    def get_theme_statistics(self, responses: List[Dict]) -> Dict:
        """Get statistics about theme classifications"""
        try:
            theme_counts = {theme: 0 for theme in self.themes}
            theme_counts["unknown"] = 0
            
            for response in responses:
                theme = response.get("theme", "unknown")
                if theme in theme_counts:
                    theme_counts[theme] += 1
                else:
                    theme_counts["unknown"] += 1
            
            total = len(responses)
            percentages = {
                theme: (count / total * 100) if total > 0 else 0
                for theme, count in theme_counts.items()
            }
            
            return {
                "total_responses": total,
                "theme_counts": theme_counts,
                "theme_percentages": percentages,
                "most_common_theme": max(theme_counts, key=theme_counts.get)
            }
            
        except Exception as e:
            logger.error(f"Error calculating theme statistics: {str(e)}")
            return {"error": str(e)}

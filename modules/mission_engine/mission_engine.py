"""
Mission Engine Module for NASA Space Apps Hackathon MVP
Combines vectorstore and knowledge graph to generate intelligent insights
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from openai import OpenAI
import os

logger = logging.getLogger("mission_engine")


class MissionEngine:
    """
    Mission intelligence engine that combines RAG and knowledge graph insights
    """
    
    def __init__(self, config: Dict, vectorstore=None, kg_store=None):
        self.config = config
        self.vectorstore = vectorstore
        self.kg_store = kg_store
        
        # RAG configuration
        self.rag_config = config.get("rag", {})
        self.similarity_top_k = self.rag_config.get("similarity_top_k", 5)
        self.rerank = self.rag_config.get("rerank", True)
        
        # Synthesis configuration
        self.synthesis_config = config.get("synthesis", {})
        self.max_context_length = self.synthesis_config.get("max_context_length", 4000)
        self.include_sources = self.synthesis_config.get("include_sources", True)
        
        # Initialize OpenAI client
        self.openai_client = None
        self._init_openai_client()
        
        logger.info("MissionEngine initialized")
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for mission engine")
            else:
                logger.warning("OpenAI API key not found for mission engine")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def mission_insight_query(self, query: str, user_context: Dict = None) -> Dict:
        """
        Process a mission insight query using RAG + KG
        
        Args:
            query: User query
            user_context: Optional user context information
            
        Returns:
            Structured response with insights and sources
        """
        try:
            logger.info(f"Processing mission insight query: {query[:100]}...")
            
            # Step 1: Retrieve relevant chunks from vectorstore
            relevant_chunks = self._retrieve_relevant_chunks(query)
            
            # Step 2: Extract entities from query for KG lookup
            query_entities = self._extract_query_entities(query)
            
            # Step 3: Get KG insights for entities
            kg_insights = self._get_kg_insights(query_entities)
            
            # Step 4: Generate embedding for query (if vectorstore supports it)
            query_embedding = None
            if self.vectorstore and hasattr(self.vectorstore, 'embed_single_text'):
                query_embedding = self.vectorstore.embed_single_text(query)
            
            # Step 5: Synthesize response
            response = self._synthesize_response(
                query, relevant_chunks, kg_insights, user_context
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing mission insight query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "insights": [],
                "sources": []
            }
    
    def _retrieve_relevant_chunks(self, query: str) -> List[Dict]:
        """Retrieve relevant chunks from vectorstore"""
        try:
            if not self.vectorstore:
                logger.warning("No vectorstore available for retrieval")
                return []
            
            # Query vectorstore
            results = self.vectorstore.query(query, top_k=self.similarity_top_k)
            
            # Format results
            chunks = []
            for result in results:
                chunk = {
                    "text": result.get("text", ""),
                    "document_id": result.get("document_id", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "similarity_score": result.get("similarity_score", 0.0),
                    "metadata": result.get("metadata", {})
                }
                chunks.append(chunk)
            
            logger.info(f"Retrieved {len(chunks)} relevant chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from query for KG lookup"""
        try:
            # Simple entity extraction (can be enhanced with NER)
            entities = []
            
            # Look for common space biology terms
            space_terms = [
                "microgravity", "spaceflight", "mission", "astronaut", "space",
                "gravity", "radiation", "experiment", "study", "research",
                "organism", "mouse", "rat", "C. elegans", "Drosophila", "Arabidopsis"
            ]
            
            query_lower = query.lower()
            for term in space_terms:
                if term in query_lower:
                    entities.append(term)
            
            # Extract potential entity names (capitalized words)
            import re
            potential_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            entities.extend(potential_entities)
            
            # Remove duplicates
            entities = list(set(entities))
            
            logger.info(f"Extracted entities from query: {entities}")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting query entities: {str(e)}")
            return []
    
    def _get_kg_insights(self, entities: List[str]) -> Dict:
        """Get knowledge graph insights for entities"""
        try:
            if not self.kg_store or not entities:
                return {"entities": {}, "relations": [], "paths": []}
            
            insights = {
                "entities": {},
                "relations": [],
                "paths": []
            }
            
            # Get insights for each entity
            for entity in entities:
                # Get entity neighbors
                neighbors = self.kg_store.get_entity_neighbors(entity)
                if neighbors:
                    insights["entities"][entity] = neighbors
                
                # Get related entities
                related = self.kg_store.query_graph(entity, max_depth=2)
                insights["relations"].extend(related)
            
            # Find paths between entities
            if len(entities) >= 2:
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        paths = self.kg_store.find_paths(entities[i], entities[j])
                        insights["paths"].extend(paths)
            
            logger.info(f"Retrieved KG insights for {len(entities)} entities")
            return insights
            
        except Exception as e:
            logger.error(f"Error getting KG insights: {str(e)}")
            return {"entities": {}, "relations": [], "paths": []}
    
    def _synthesize_response(self, query: str, chunks: List[Dict], 
                           kg_insights: Dict, user_context: Dict = None) -> Dict:
        """Synthesize final response using LLM"""
        try:
            if not self.openai_client:
                # Fallback to simple concatenation
                return self._simple_synthesis(query, chunks, kg_insights)
            
            # Prepare context for LLM
            context = self._prepare_llm_context(query, chunks, kg_insights)
            
            # Generate response using OpenAI
            response = self._generate_llm_response(query, context, user_context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error synthesizing response: {str(e)}")
            return self._simple_synthesis(query, chunks, kg_insights)
    
    def _prepare_llm_context(self, query: str, chunks: List[Dict], kg_insights: Dict) -> str:
        """Prepare context for LLM"""
        try:
            context_parts = []
            
            # Add relevant text chunks
            if chunks:
                context_parts.append("## Relevant Research Documents:")
                for i, chunk in enumerate(chunks[:3], 1):  # Limit to top 3
                    context_parts.append(f"### Document {i}:")
                    context_parts.append(f"Source: {chunk.get('document_id', 'Unknown')}")
                    context_parts.append(f"Content: {chunk.get('text', '')[:500]}...")
                    context_parts.append("")
            
            # Add KG insights
            if kg_insights.get("relations"):
                context_parts.append("## Knowledge Graph Insights:")
                for relation in kg_insights["relations"][:5]:  # Limit to top 5
                    context_parts.append(
                        f"- {relation.get('source', '')} {relation.get('relation', '')} {relation.get('target', '')}"
                    )
                context_parts.append("")
            
            # Add entity connections
            if kg_insights.get("entities"):
                context_parts.append("## Related Entities:")
                for entity, neighbors in kg_insights["entities"].items():
                    if neighbors:
                        context_parts.append(f"**{entity}** is connected to:")
                        for neighbor in neighbors[:3]:  # Limit to top 3
                            context_parts.append(f"  - {neighbor.get('entity', '')} ({neighbor.get('relation', '')})")
                        context_parts.append("")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error preparing LLM context: {str(e)}")
            return ""
    
    def _generate_llm_response(self, query: str, context: str, user_context: Dict = None) -> Dict:
        """Generate response using OpenAI"""
        try:
            # Truncate context if too long
            max_context = self.max_context_length
            if len(context) > max_context:
                context = context[:max_context] + "\n\n[Context truncated...]"
            
            # Prepare system prompt
            system_prompt = """You are a space biology expert assistant. Your role is to provide comprehensive, 
            scientifically accurate answers about space biology research, missions, and experiments.
            
            Guidelines:
            1. Use only the information provided in the context
            2. Be specific about sources and document references
            3. Highlight key relationships between entities
            4. Provide actionable insights when possible
            5. If information is insufficient, clearly state limitations
            
            Always cite your sources and be transparent about the confidence level of your information."""
            
            # Prepare user message
            user_message = f"""Query: {query}

Context:
{context}

Please provide a comprehensive answer based on the provided research documents and knowledge graph insights."""
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            # Prepare sources
            sources = []
            if self.include_sources:
                # Extract sources from chunks
                for chunk in chunks:
                    if chunk.get("document_id"):
                        sources.append({
                            "document_id": chunk.get("document_id"),
                            "chunk_id": chunk.get("chunk_id"),
                            "similarity_score": chunk.get("similarity_score", 0.0)
                        })
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "sources": sources,
                "context_used": len(context),
                "model_used": "gpt-4o-mini"
            }
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def _simple_synthesis(self, query: str, chunks: List[Dict], kg_insights: Dict) -> Dict:
        """Simple synthesis without LLM"""
        try:
            # Combine relevant chunks
            combined_text = "\n\n".join([
                f"Source: {chunk.get('document_id', 'Unknown')}\n{chunk.get('text', '')}"
                for chunk in chunks[:3]
            ])
            
            # Add KG insights
            kg_text = ""
            if kg_insights.get("relations"):
                kg_text = "\n\nKnowledge Graph Insights:\n"
                for relation in kg_insights["relations"][:3]:
                    kg_text += f"- {relation.get('source', '')} {relation.get('relation', '')} {relation.get('target', '')}\n"
            
            # Create simple response
            answer = f"Based on the research documents and knowledge graph:\n\n{combined_text}{kg_text}"
            
            # Prepare sources
            sources = []
            for chunk in chunks:
                if chunk.get("document_id"):
                    sources.append({
                        "document_id": chunk.get("document_id"),
                        "chunk_id": chunk.get("chunk_id"),
                        "similarity_score": chunk.get("similarity_score", 0.0)
                    })
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "sources": sources,
                "synthesis_method": "simple"
            }
            
        except Exception as e:
            logger.error(f"Error in simple synthesis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def generate_summary_from_chunks(self, chunks: List[Dict]) -> str:
        """Generate a summary from chunks"""
        try:
            if not chunks:
                return "No relevant information found."
            
            # Combine chunk texts
            combined_text = "\n\n".join([chunk.get("text", "") for chunk in chunks])
            
            # Truncate if too long
            max_length = 2000
            if len(combined_text) > max_length:
                combined_text = combined_text[:max_length] + "..."
            
            # Generate summary using LLM if available
            if self.openai_client:
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Summarize the following space biology research content concisely and accurately."},
                            {"role": "user", "content": f"Summarize this research content:\n\n{combined_text}"}
                        ],
                        max_tokens=300,
                        temperature=0.3
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logger.warning(f"LLM summary failed: {str(e)}")
            
            # Fallback to simple summary
            sentences = combined_text.split('.')
            summary_sentences = sentences[:3]  # Take first 3 sentences
            return '. '.join(summary_sentences) + '.'
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating summary."
    
    def get_mission_statistics(self) -> Dict:
        """Get statistics about the mission engine"""
        try:
            stats = {
                "vectorstore_available": self.vectorstore is not None,
                "kg_store_available": self.kg_store is not None,
                "openai_available": self.openai_client is not None
            }
            
            # Add vectorstore stats
            if self.vectorstore:
                vs_stats = self.vectorstore.get_stats()
                stats["vectorstore_stats"] = vs_stats
            
            # Add KG stats
            if self.kg_store:
                kg_stats = self.kg_store.get_graph_statistics()
                stats["kg_stats"] = kg_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting mission statistics: {str(e)}")
            return {"error": str(e)}

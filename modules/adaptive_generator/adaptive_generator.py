"""
Adaptive Generator Module for NASA Space Apps Hackathon MVP
Dynamically adapts response tone and depth based on user profile detection
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from openai import OpenAI
import os
from modules.prompt_engineering import PromptEngineer

logger = logging.getLogger("adaptive_generator")


class AdaptiveGenerator:
    """
    Adaptive tone generator that adjusts responses based on detected user profile
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.user_profiles = config.get("user_profiles", ["scientist", "manager", "layperson"])
        self.tone_classification = config.get("tone_classification", {})
        self.prompt_templates = config.get("prompt_templates", {})
        
        # Initialize OpenAI client
        self.openai_client = None
        self._init_openai_client()
        
        # Initialize prompt engineer
        self.prompt_engineer = PromptEngineer(config.get("prompt_engineering", {}))
        
        # Load prompt templates
        self.templates = self._load_prompt_templates()
        
        logger.info("AdaptiveGenerator initialized")
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for adaptive generator")
            else:
                logger.warning("OpenAI API key not found for adaptive generator")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates for different user profiles"""
        templates = {}
        
        # Default templates
        templates["scientist"] = """You are responding to a scientific researcher. Use precise, technical language with:
- Specific terminology and jargon appropriate for space biology
- Detailed methodology and experimental design information
- Statistical significance and confidence intervals where relevant
- Citations to specific studies and data points
- Technical depth appropriate for peer review
- Focus on mechanisms, pathways, and biological processes"""
        
        templates["manager"] = """You are responding to a research manager or administrator. Use strategic, high-level language with:
- Executive summaries and key findings
- Strategic implications and research opportunities
- Resource requirements and timeline considerations
- Risk assessment and mitigation strategies
- Competitive advantages and innovation potential
- ROI and impact metrics where applicable"""
        
        templates["layperson"] = """You are responding to a general audience. Use clear, accessible language with:
- Simple explanations of complex concepts
- Analogies and real-world examples
- Visual descriptions and metaphors
- Step-by-step explanations
- Avoid technical jargon or explain it clearly
- Focus on practical implications and benefits"""
        
        # Try to load from files if specified
        for profile, template_path in self.prompt_templates.items():
            try:
                if Path(template_path).exists():
                    with open(template_path, 'r', encoding='utf-8') as f:
                        templates[profile] = f.read()
                    logger.info(f"Loaded template for {profile} from {template_path}")
            except Exception as e:
                logger.warning(f"Could not load template for {profile}: {str(e)}")
        
        return templates
    
    def detect_user_profile(self, query: str) -> str:
        """
        Infer user type based on lexical, syntactic, and semantic cues
        
        Args:
            query: User query text
            
        Returns:
            Detected user profile ('scientist', 'manager', 'layperson')
        """
        try:
            # Rule-based detection
            profile_scores = {
                "scientist": 0,
                "manager": 0,
                "layperson": 0
            }
            
            query_lower = query.lower()
            
            # Scientist indicators
            scientist_indicators = [
                # Technical terms
                r'\b(mechanism|pathway|regulation|expression|transcription|translation)\b',
                r'\b(statistical|significance|confidence|interval|p-value|correlation)\b',
                r'\b(methodology|experimental|design|hypothesis|control|variable)\b',
                r'\b(microgravity|spaceflight|radiation|stress|response|adaptation)\b',
                # Academic language
                r'\b(study|research|analysis|investigation|examination|assessment)\b',
                r'\b(results|findings|conclusions|implications|significance)\b',
                # Specific organisms/techniques
                r'\b(C\. elegans|Drosophila|Arabidopsis|PCR|Western blot|sequencing)\b'
            ]
            
            for pattern in scientist_indicators:
                if re.search(pattern, query_lower):
                    profile_scores["scientist"] += 1
            
            # Manager indicators
            manager_indicators = [
                # Strategic terms
                r'\b(strategy|planning|budget|resource|timeline|milestone)\b',
                r'\b(impact|outcome|deliverable|objective|goal|target)\b',
                r'\b(team|collaboration|partnership|stakeholder|client)\b',
                r'\b(risk|opportunity|challenge|solution|recommendation)\b',
                r'\b(ROI|efficiency|productivity|performance|metrics)\b',
                # Business language
                r'\b(overview|summary|executive|brief|report|presentation)\b'
            ]
            
            for pattern in manager_indicators:
                if re.search(pattern, query_lower):
                    profile_scores["manager"] += 1
            
            # Layperson indicators
            layperson_indicators = [
                # Simple questions
                r'\b(what is|how does|why|when|where|can you explain)\b',
                r'\b(simple|easy|basic|understand|learn|teach)\b',
                r'\b(example|analogy|like|similar to|imagine)\b',
                # General interest
                r'\b(interesting|fascinating|amazing|cool|wow)\b',
                r'\b(space|astronaut|moon|mars|gravity|floating)\b'
            ]
            
            for pattern in layperson_indicators:
                if re.search(pattern, query_lower):
                    profile_scores["layperson"] += 1
            
            # Use OpenAI for more sophisticated detection if available
            if self.openai_client and len(query) > 50:
                try:
                    ai_profile = self._detect_profile_with_ai(query)
                    if ai_profile in profile_scores:
                        profile_scores[ai_profile] += 2  # Boost AI-detected profile
                except Exception as e:
                    logger.warning(f"AI profile detection failed: {str(e)}")
            
            # Determine profile with highest score
            if max(profile_scores.values()) == 0:
                # Default to layperson if no clear indicators
                detected_profile = "layperson"
            else:
                detected_profile = max(profile_scores, key=profile_scores.get)
            
            logger.info(f"Detected user profile: {detected_profile} (scores: {profile_scores})")
            return detected_profile
            
        except Exception as e:
            logger.error(f"Error detecting user profile: {str(e)}")
            return "layperson"  # Safe default
    
    def _detect_profile_with_ai(self, query: str) -> str:
        """Use OpenAI to detect user profile"""
        try:
            prompt = f"""Analyze this query and determine the user's likely background:

Query: "{query}"

Classify the user as one of these profiles:
- scientist: Uses technical language, asks about mechanisms, methods, data
- manager: Focuses on strategy, outcomes, resources, timelines, impact
- layperson: Asks simple questions, wants explanations, uses everyday language

Respond with only the profile name (scientist, manager, or layperson)."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().lower()
            if result in ["scientist", "manager", "layperson"]:
                return result
            else:
                return "layperson"
                
        except Exception as e:
            logger.error(f"Error in AI profile detection: {str(e)}")
            return "layperson"
    
    def select_prompt_template(self, user_type: str) -> str:
        """
        Load the generation prompt template based on detected user profile
        
        Args:
            user_type: Detected user profile
            
        Returns:
            Prompt template string
        """
        try:
            if user_type in self.templates:
                return self.templates[user_type]
            else:
                logger.warning(f"Unknown user type: {user_type}, using layperson template")
                return self.templates.get("layperson", self.templates["scientist"])
                
        except Exception as e:
            logger.error(f"Error selecting prompt template: {str(e)}")
            return self.templates["layperson"]
    
    def generate_adaptive_response(self, query: str, retrieved_chunks: List[Dict], 
                                 user_profile: str = None) -> Dict:
        """
        Combine the selected prompt and RAG context to produce the final answer
        Now uses the prompt engineering module for structured responses
        
        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks
            user_profile: Optional pre-detected user profile
            
        Returns:
            Generated response with metadata, references, and theme
        """
        try:
            # Detect user profile if not provided
            if user_profile is None:
                user_profile = self.detect_user_profile(query)
            
            # Use prompt engineering for structured response
            structured_response = self.prompt_engineer.adaptive_rag_pipeline(
                query, retrieved_chunks, user_profile
            )
            
            # Add additional metadata
            structured_response["user_profile"] = user_profile
            structured_response["template_used"] = user_profile
            structured_response["chunks_used"] = len(retrieved_chunks)
            structured_response["generation_method"] = "prompt_engineering"
            
            return structured_response
            
        except Exception as e:
            logger.error(f"Error generating adaptive response: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "user_profile": user_profile or "unknown",
                "answer": f"Error processing query: {str(e)}",
                "references": [],
                "theme": "unknown",
                "confidence": 0.0,
                "key_findings": []
            }
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context from retrieved chunks"""
        try:
            if not chunks:
                return "No relevant information found."
            
            context_parts = []
            for i, chunk in enumerate(chunks[:3], 1):  # Limit to top 3 chunks
                context_parts.append(f"Source {i}: {chunk.get('text', '')[:300]}...")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error preparing context: {str(e)}")
            return "Error preparing context."
    
    def _generate_with_llm(self, query: str, template: str, context: str, user_profile: str) -> Dict:
        """Generate response using OpenAI"""
        try:
            # Prepare system message
            system_message = f"""{template}

Context from research documents:
{context}

Guidelines:
1. Use only the information provided in the context
2. Adapt your language and depth to the user profile: {user_profile}
3. Be accurate and cite sources when possible
4. If information is insufficient, clearly state limitations
5. Provide actionable insights when appropriate"""

            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "user_profile": user_profile,
                "model_used": "gpt-4o-mini",
                "generation_method": "llm"
            }
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return self._generate_simple_response(query, context, user_profile)
    
    def _generate_simple_response(self, query: str, context: str, user_profile: str) -> Dict:
        """Generate simple response without LLM"""
        try:
            # Create profile-specific response
            if user_profile == "scientist":
                answer = f"Based on the research data: {context[:500]}..."
            elif user_profile == "manager":
                answer = f"Key findings from the research: {context[:400]}..."
            else:  # layperson
                answer = f"Here's what the research tells us: {context[:300]}..."
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "user_profile": user_profile,
                "generation_method": "simple"
            }
            
        except Exception as e:
            logger.error(f"Error generating simple response: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "user_profile": user_profile
            }
    
    def get_profile_statistics(self, queries: List[str]) -> Dict:
        """Get statistics about detected user profiles"""
        try:
            profile_counts = {"scientist": 0, "manager": 0, "layperson": 0}
            
            for query in queries:
                profile = self.detect_user_profile(query)
                profile_counts[profile] += 1
            
            total = len(queries)
            percentages = {
                profile: (count / total * 100) if total > 0 else 0
                for profile, count in profile_counts.items()
            }
            
            return {
                "total_queries": total,
                "profile_counts": profile_counts,
                "profile_percentages": percentages,
                "most_common_profile": max(profile_counts, key=profile_counts.get)
            }
            
        except Exception as e:
            logger.error(f"Error calculating profile statistics: {str(e)}")
            return {"error": str(e)}
    
    def update_template(self, user_profile: str, template: str) -> bool:
        """Update prompt template for a user profile"""
        try:
            if user_profile in self.templates:
                self.templates[user_profile] = template
                logger.info(f"Updated template for {user_profile}")
                return True
            else:
                logger.warning(f"Unknown user profile: {user_profile}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating template: {str(e)}")
            return False
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available user profiles"""
        return list(self.templates.keys())
    
    def get_template_preview(self, user_profile: str) -> str:
        """Get preview of template for a user profile"""
        try:
            template = self.templates.get(user_profile, "")
            return template[:200] + "..." if len(template) > 200 else template
        except Exception as e:
            logger.error(f"Error getting template preview: {str(e)}")
            return ""

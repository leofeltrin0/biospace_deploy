"""
Knowledge Graph Extraction Module for NASA Space Apps Hackathon MVP
Extracts entities and relations from space biology documents to build knowledge graphs
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import spacy
from openai import OpenAI
import os

logger = logging.getLogger("kg_extraction")


class KGExtractor:
    """
    Extracts entities and relations from text to build knowledge graphs
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = config.get("models", {})
        self.entity_types = config.get("entity_types", ["ORGANISM", "EXPERIMENT", "MISSION", "OUTCOME", "TECHNIQUE", "CONDITION"])
        self.relation_types = config.get("relation_types", ["AFFECTS", "CAUSES", "MEASURES", "OCCURS_IN", "RESULTS_IN"])
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        
        # Initialize models
        self.nlp = None
        self.openai_client = None
        
        # Load spaCy model
        self._load_spacy_model()
        
        # Initialize OpenAI client
        self._init_openai_client()
        
        logger.info("KGExtractor initialized")
    
    def _load_spacy_model(self):
        """Load spaCy NER model"""
        try:
            model_name = self.models.get("ner", "en_core_web_sm")
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            # Try to download the model
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Downloaded and loaded spaCy model")
            except Exception as e2:
                logger.error(f"Failed to download spaCy model: {str(e2)}")
                self.nlp = None
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not found")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def extract_entities_relations(self, text: str) -> List[Tuple]:
        """
        Extract entities and relations from text using multiple methods
        
        Args:
            text: Input text to process
            
        Returns:
            List of (subject, relation, object) tuples
        """
        if not text or not text.strip():
            return []
        
        try:
            # Extract using spaCy
            spacy_triples = self._extract_with_spacy(text)
            
            # Extract using OpenAI (if available)
            openai_triples = []
            if self.openai_client:
                openai_triples = self._extract_with_openai(text)
            
            # Combine and deduplicate results
            all_triples = spacy_triples + openai_triples
            unique_triples = self._deduplicate_triples(all_triples)
            
            # Filter by confidence
            filtered_triples = [
                triple for triple in unique_triples 
                if triple[3] >= self.confidence_threshold  # confidence score
            ]
            
            logger.info(f"Extracted {len(filtered_triples)} triples from text")
            return filtered_triples
            
        except Exception as e:
            logger.error(f"Error extracting entities and relations: {str(e)}")
            return []
    
    def _extract_with_spacy(self, text: str) -> List[Tuple]:
        """Extract entities using spaCy NER"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            triples = []
            
            # Extract named entities
            entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
            
            # Map spaCy labels to our entity types
            label_mapping = {
                "PERSON": "PERSON",
                "ORG": "ORGANIZATION", 
                "GPE": "LOCATION",
                "EVENT": "EVENT",
                "WORK_OF_ART": "TECHNIQUE",
                "LAW": "CONDITION"
            }
            
            # Create entity-relation patterns
            for i, (subj_text, subj_label, subj_start, subj_end) in enumerate(entities):
                for j, (obj_text, obj_label, obj_start, obj_end) in enumerate(entities):
                    if i != j and subj_start < obj_start:  # Subject before object
                        # Determine relation based on context
                        relation = self._determine_relation_spacy(
                            text, subj_text, obj_text, subj_start, obj_start, obj_end
                        )
                        
                        if relation:
                            confidence = self._calculate_confidence_spacy(
                                subj_text, obj_text, relation, text
                            )
                            
                            triples.append((
                                subj_text.strip(),
                                relation,
                                obj_text.strip(),
                                confidence
                            ))
            
            return triples
            
        except Exception as e:
            logger.error(f"Error in spaCy extraction: {str(e)}")
            return []
    
    def _extract_with_openai(self, text: str) -> List[Tuple]:
        """Extract entities and relations using OpenAI"""
        if not self.openai_client:
            return []
        
        try:
            # Truncate text if too long
            max_length = 3000
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            prompt = f"""
            Extract entities and relations from the following space biology research text.
            Focus on scientific entities like organisms, experiments, missions, outcomes, techniques, and conditions.
            
            Text: {text}
            
            Extract triples in the format: (subject, relation, object)
            Relations should be one of: {', '.join(self.relation_types)}
            Entity types should be one of: {', '.join(self.entity_types)}
            
            Return only the triples, one per line, in the format:
            subject | relation | object
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Parse response
            triples = []
            for line in response.choices[0].message.content.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        subject = parts[0].strip()
                        relation = parts[1].strip()
                        obj = parts[2].strip()
                        
                        # Validate relation type
                        if relation in self.relation_types:
                            confidence = 0.8  # High confidence for OpenAI
                            triples.append((subject, relation, obj, confidence))
            
            return triples
            
        except Exception as e:
            logger.error(f"Error in OpenAI extraction: {str(e)}")
            return []
    
    def _determine_relation_spacy(self, text: str, subj: str, obj: str, subj_start: int, obj_start: int, obj_end: int) -> Optional[str]:
        """Determine relation between entities using context"""
        try:
            # Extract context between entities
            context_start = max(0, subj_start - 50)
            context_end = min(len(text), obj_end + 50)
            context = text[context_start:context_end].lower()
            
            # Relation patterns
            relation_patterns = {
                "AFFECTS": [r"affects?", r"influences?", r"impacts?", r"changes?"],
                "CAUSES": [r"causes?", r"leads to", r"results in", r"induces?"],
                "MEASURES": [r"measures?", r"quantifies?", r"assesses?", r"evaluates?"],
                "OCCURS_IN": [r"occurs in", r"happens in", r"takes place in", r"found in"],
                "RESULTS_IN": [r"results in", r"produces?", r"generates?", r"creates?"]
            }
            
            for relation, patterns in relation_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, context):
                        return relation
            
            # Default relation based on entity types
            if "experiment" in subj.lower() and "outcome" in obj.lower():
                return "RESULTS_IN"
            elif "organism" in subj.lower() and "condition" in obj.lower():
                return "AFFECTS"
            
            return None
            
        except Exception as e:
            logger.error(f"Error determining relation: {str(e)}")
            return None
    
    def _calculate_confidence_spacy(self, subj: str, obj: str, relation: str, text: str) -> float:
        """Calculate confidence score for spaCy extraction"""
        try:
            confidence = 0.5  # Base confidence
            
            # Boost confidence for scientific terms
            scientific_terms = ["experiment", "study", "research", "analysis", "measurement", "observation"]
            for term in scientific_terms:
                if term in text.lower():
                    confidence += 0.1
            
            # Boost confidence for space biology terms
            space_terms = ["microgravity", "spaceflight", "mission", "astronaut", "space", "gravity"]
            for term in space_terms:
                if term in text.lower():
                    confidence += 0.1
            
            # Boost confidence for entity length (longer entities are more specific)
            confidence += min(0.2, (len(subj) + len(obj)) / 100)
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _deduplicate_triples(self, triples: List[Tuple]) -> List[Tuple]:
        """Remove duplicate triples"""
        try:
            seen = set()
            unique_triples = []
            
            for triple in triples:
                # Create a key for deduplication (subject, relation, object)
                key = (triple[0].lower(), triple[1], triple[2].lower())
                
                if key not in seen:
                    seen.add(key)
                    unique_triples.append(triple)
                else:
                    # Keep the one with higher confidence
                    for i, existing in enumerate(unique_triples):
                        existing_key = (existing[0].lower(), existing[1], existing[2].lower())
                        if key == existing_key and triple[3] > existing[3]:
                            unique_triples[i] = triple
                            break
            
            return unique_triples
            
        except Exception as e:
            logger.error(f"Error deduplicating triples: {str(e)}")
            return triples
    
    def process_chunks_for_kg(self, chunks: List[Dict]) -> List[Dict]:
        """
        Process chunks to extract knowledge graph triples
        
        Args:
            chunks: List of text chunks with embeddings
            
        Returns:
            List of chunks with extracted triples
        """
        try:
            processed_chunks = []
            
            for chunk in chunks:
                chunk_copy = chunk.copy()
                text = chunk.get("text", "")
                
                if text:
                    # Extract triples
                    triples = self.extract_entities_relations(text)
                    
                    # Add triples to chunk metadata
                    chunk_copy["extracted_triples"] = [
                        {
                            "subject": triple[0],
                            "relation": triple[1],
                            "object": triple[2],
                            "confidence": triple[3]
                        }
                        for triple in triples
                    ]
                    
                    chunk_copy["triple_count"] = len(triples)
                else:
                    chunk_copy["extracted_triples"] = []
                    chunk_copy["triple_count"] = 0
                
                processed_chunks.append(chunk_copy)
            
            logger.info(f"Processed {len(processed_chunks)} chunks for KG extraction")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing chunks for KG: {str(e)}")
            return chunks
    
    def extract_domain_specific_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract domain-specific entities for space biology
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        try:
            entities = {
                "organisms": [],
                "experiments": [],
                "missions": [],
                "techniques": [],
                "conditions": [],
                "outcomes": []
            }
            
            # Organism patterns
            organism_patterns = [
                r'\b[A-Z][a-z]+ [a-z]+\b',  # Genus species
                r'\b[A-Z][a-z]+ sp\.',  # Genus sp.
                r'\bmice?\b', r'\brats?\b', r'\bC\. elegans\b',
                r'\bDrosophila\b', r'\bArabidopsis\b'
            ]
            
            for pattern in organism_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["organisms"].extend(matches)
            
            # Experiment patterns
            experiment_patterns = [
                r'\b[A-Z][a-z]+ experiment\b',
                r'\bstudy\b', r'\btrial\b', r'\btest\b',
                r'\bexperiment [A-Z]\d+\b'
            ]
            
            for pattern in experiment_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["experiments"].extend(matches)
            
            # Mission patterns
            mission_patterns = [
                r'\b[A-Z]{2,}\d+\b',  # Mission codes
                r'\bInternational Space Station\b',
                r'\bISS\b', r'\bNASA\b', r'\bESA\b'
            ]
            
            for pattern in mission_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["missions"].extend(matches)
            
            # Technique patterns
            technique_patterns = [
                r'\b[A-Z][a-z]+ [a-z]+ analysis\b',
                r'\bsequencing\b', r'\bPCR\b', r'\bWestern blot\b',
                r'\bmicroscopy\b', r'\bimaging\b'
            ]
            
            for pattern in technique_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["techniques"].extend(matches)
            
            # Condition patterns
            condition_patterns = [
                r'\bmicrogravity\b', r'\bzero gravity\b',
                r'\bspaceflight\b', r'\bradiation\b',
                r'\bweightlessness\b'
            ]
            
            for pattern in condition_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities["conditions"].extend(matches)
            
            # Clean and deduplicate
            for entity_type in entities:
                entities[entity_type] = list(set(entities[entity_type]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting domain entities: {str(e)}")
            return {}
    
    def get_extraction_statistics(self, processed_chunks: List[Dict]) -> Dict:
        """
        Get statistics about KG extraction
        
        Args:
            processed_chunks: List of processed chunks with triples
            
        Returns:
            Statistics dictionary
        """
        try:
            total_chunks = len(processed_chunks)
            chunks_with_triples = sum(1 for chunk in processed_chunks if chunk.get("triple_count", 0) > 0)
            total_triples = sum(chunk.get("triple_count", 0) for chunk in processed_chunks)
            
            # Count relation types
            relation_counts = {}
            for chunk in processed_chunks:
                for triple in chunk.get("extracted_triples", []):
                    relation = triple.get("relation", "")
                    relation_counts[relation] = relation_counts.get(relation, 0) + 1
            
            stats = {
                "total_chunks": total_chunks,
                "chunks_with_triples": chunks_with_triples,
                "total_triples": total_triples,
                "avg_triples_per_chunk": total_triples / total_chunks if total_chunks > 0 else 0,
                "coverage_percentage": (chunks_with_triples / total_chunks * 100) if total_chunks > 0 else 0,
                "relation_counts": relation_counts
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating extraction statistics: {str(e)}")
            return {}

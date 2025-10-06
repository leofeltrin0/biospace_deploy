"""
Knowledge Graph Storage Module for NASA Space Apps Hackathon MVP
Stores and manages knowledge graph data for querying and visualization
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
import networkx as nx
from datetime import datetime

logger = logging.getLogger("kg_store")


class KGStore:
    """
    Manages knowledge graph storage and retrieval using NetworkX
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.backend = config.get("backend", "networkx")
        self.persist_dir = Path(config.get("persist_dir", "./data/kg_store"))
        self.formats = config.get("formats", ["graphml", "json"])
        
        # Create persist directory
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize graph
        self.graph = nx.DiGraph()
        self.graph_file = self.persist_dir / "knowledge_graph.graphml"
        self.metadata_file = self.persist_dir / "kg_metadata.json"
        
        # Load existing graph if available
        self._load_graph()
        
        logger.info(f"KGStore initialized with backend: {self.backend}")
    
    def build_graph(self, triples: List[Tuple]) -> nx.DiGraph:
        """
        Build a NetworkX graph from triples
        
        Args:
            triples: List of (subject, relation, object, confidence) tuples
            
        Returns:
            NetworkX directed graph
        """
        try:
            graph = nx.DiGraph()
            
            for subject, relation, obj, confidence in triples:
                # Add nodes with attributes
                if not graph.has_node(subject):
                    graph.add_node(subject, 
                                 node_type="entity",
                                 label=subject,
                                 created_at=datetime.now().isoformat())
                
                if not graph.has_node(obj):
                    graph.add_node(obj,
                                 node_type="entity", 
                                 label=obj,
                                 created_at=datetime.now().isoformat())
                
                # Add edge with relation and confidence
                graph.add_edge(subject, obj,
                             relation=relation,
                             confidence=confidence,
                             created_at=datetime.now().isoformat())
            
            logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph
            
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            return nx.DiGraph()
    
    def add_triples(self, triples: List[Tuple]) -> bool:
        """
        Add triples to the knowledge graph
        
        Args:
            triples: List of (subject, relation, object, confidence) tuples
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for subject, relation, obj, confidence in triples:
                # Add nodes if they don't exist
                if not self.graph.has_node(subject):
                    self.graph.add_node(subject,
                                      node_type="entity",
                                      label=subject,
                                      created_at=datetime.now().isoformat())
                
                if not self.graph.has_node(obj):
                    self.graph.add_node(obj,
                                      node_type="entity",
                                      label=obj,
                                      created_at=datetime.now().isoformat())
                
                # Add edge
                self.graph.add_edge(subject, obj,
                                  relation=relation,
                                  confidence=confidence,
                                  created_at=datetime.now().isoformat())
            
            logger.info(f"Added {len(triples)} triples to knowledge graph")
            return True
            
        except Exception as e:
            logger.error(f"Error adding triples: {str(e)}")
            return False
    
    def query_graph(self, entity: str, max_depth: int = 2) -> List[Dict]:
        """
        Query the graph for entities related to the given entity
        
        Args:
            entity: Entity to search for
            max_depth: Maximum depth for traversal
            
        Returns:
            List of related entities with their relationships
        """
        try:
            if not self.graph.has_node(entity):
                logger.warning(f"Entity '{entity}' not found in graph")
                return []
            
            # Find all nodes within max_depth
            related_entities = []
            visited = set()
            
            def traverse(node, depth):
                if depth > max_depth or node in visited:
                    return
                
                visited.add(node)
                
                # Get neighbors
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        edge_data = self.graph.get_edge_data(node, neighbor)
                        if edge_data:
                            related_entities.append({
                                "source": node,
                                "target": neighbor,
                                "relation": edge_data.get("relation", ""),
                                "confidence": edge_data.get("confidence", 0.0),
                                "depth": depth + 1
                            })
                        
                        if depth + 1 <= max_depth:
                            traverse(neighbor, depth + 1)
            
            traverse(entity, 0)
            
            # Sort by confidence
            related_entities.sort(key=lambda x: x["confidence"], reverse=True)
            
            logger.info(f"Found {len(related_entities)} related entities for '{entity}'")
            return related_entities
            
        except Exception as e:
            logger.error(f"Error querying graph: {str(e)}")
            return []
    
    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """
        Find paths between two entities
        
        Args:
            source: Source entity
            target: Target entity
            max_length: Maximum path length
            
        Returns:
            List of paths between entities
        """
        try:
            if not self.graph.has_node(source) or not self.graph.has_node(target):
                logger.warning(f"Source or target entity not found in graph")
                return []
            
            # Find all simple paths
            paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=max_length
            ))
            
            logger.info(f"Found {len(paths)} paths between '{source}' and '{target}'")
            return paths
            
        except Exception as e:
            logger.error(f"Error finding paths: {str(e)}")
            return []
    
    def get_entity_neighbors(self, entity: str, relation_type: str = None) -> List[Dict]:
        """
        Get neighbors of an entity, optionally filtered by relation type
        
        Args:
            entity: Entity to get neighbors for
            relation_type: Optional relation type filter
            
        Returns:
            List of neighbor entities with their relationships
        """
        try:
            if not self.graph.has_node(entity):
                return []
            
            neighbors = []
            
            # Get outgoing edges
            for target in self.graph.successors(entity):
                edge_data = self.graph.get_edge_data(entity, target)
                if edge_data:
                    relation = edge_data.get("relation", "")
                    if relation_type is None or relation == relation_type:
                        neighbors.append({
                            "entity": target,
                            "relation": relation,
                            "direction": "outgoing",
                            "confidence": edge_data.get("confidence", 0.0)
                        })
            
            # Get incoming edges
            for source in self.graph.predecessors(entity):
                edge_data = self.graph.get_edge_data(source, entity)
                if edge_data:
                    relation = edge_data.get("relation", "")
                    if relation_type is None or relation == relation_type:
                        neighbors.append({
                            "entity": source,
                            "relation": relation,
                            "direction": "incoming",
                            "confidence": edge_data.get("confidence", 0.0)
                        })
            
            # Sort by confidence
            neighbors.sort(key=lambda x: x["confidence"], reverse=True)
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Error getting entity neighbors: {str(e)}")
            return []
    
    def get_graph_statistics(self) -> Dict:
        """
        Get statistics about the knowledge graph
        
        Returns:
            Dictionary of graph statistics
        """
        try:
            stats = {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "is_connected": nx.is_weakly_connected(self.graph),
                "number_of_components": nx.number_weakly_connected_components(self.graph)
            }
            
            # Node degree statistics
            degrees = dict(self.graph.degree())
            if degrees:
                stats["avg_degree"] = sum(degrees.values()) / len(degrees)
                stats["max_degree"] = max(degrees.values())
                stats["min_degree"] = min(degrees.values())
            
            # Relation type counts
            relation_counts = {}
            for source, target, data in self.graph.edges(data=True):
                relation = data.get("relation", "")
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
            
            stats["relation_counts"] = relation_counts
            
            # Confidence statistics
            confidences = [data.get("confidence", 0.0) for _, _, data in self.graph.edges(data=True)]
            if confidences:
                stats["avg_confidence"] = sum(confidences) / len(confidences)
                stats["max_confidence"] = max(confidences)
                stats["min_confidence"] = min(confidences)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating graph statistics: {str(e)}")
            return {}
    
    def save_graph(self, output_path: str = None) -> bool:
        """
        Save the knowledge graph to file
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if output_path:
                output_file = Path(output_path)
            else:
                output_file = self.graph_file
            
            # Save in different formats
            success = True
            
            # GraphML format
            if "graphml" in self.formats:
                try:
                    nx.write_graphml(self.graph, str(output_file.with_suffix(".graphml")))
                    logger.info(f"Saved graph in GraphML format to {output_file.with_suffix('.graphml')}")
                except Exception as e:
                    logger.error(f"Error saving GraphML: {str(e)}")
                    success = False
            
            # JSON format
            if "json" in self.formats:
                try:
                    json_file = output_file.with_suffix(".json")
                    graph_data = {
                        "nodes": [
                            {
                                "id": node,
                                "label": data.get("label", node),
                                "node_type": data.get("node_type", "entity"),
                                "created_at": data.get("created_at", "")
                            }
                            for node, data in self.graph.nodes(data=True)
                        ],
                        "edges": [
                            {
                                "source": source,
                                "target": target,
                                "relation": data.get("relation", ""),
                                "confidence": data.get("confidence", 0.0),
                                "created_at": data.get("created_at", "")
                            }
                            for source, target, data in self.graph.edges(data=True)
                        ]
                    }
                    
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(graph_data, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Saved graph in JSON format to {json_file}")
                except Exception as e:
                    logger.error(f"Error saving JSON: {str(e)}")
                    success = False
            
            # Save metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "statistics": self.get_graph_statistics()
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
            return False
    
    def _load_graph(self) -> bool:
        """
        Load existing graph from file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.graph_file.exists():
                self.graph = nx.read_graphml(str(self.graph_file))
                logger.info(f"Loaded existing graph with {self.graph.number_of_nodes()} nodes")
                return True
            else:
                logger.info("No existing graph found, starting with empty graph")
                return True
                
        except Exception as e:
            logger.error(f"Error loading graph: {str(e)}")
            return False
    
    def export_for_visualization(self, output_path: str) -> bool:
        """
        Export graph data in a format suitable for visualization
        
        Args:
            output_path: Path to save visualization data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data for D3.js or similar visualization libraries
            viz_data = {
                "nodes": [
                    {
                        "id": node,
                        "label": data.get("label", node),
                        "group": data.get("node_type", "entity"),
                        "size": self.graph.degree(node)
                    }
                    for node, data in self.graph.nodes(data=True)
                ],
                "links": [
                    {
                        "source": source,
                        "target": target,
                        "relation": data.get("relation", ""),
                        "confidence": data.get("confidence", 0.0),
                        "weight": data.get("confidence", 0.0)
                    }
                    for source, target, data in self.graph.edges(data=True)
                ]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(viz_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported visualization data to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting for visualization: {str(e)}")
            return False
    
    def get_central_entities(self, top_k: int = 10) -> List[Dict]:
        """
        Get the most central entities in the graph
        
        Args:
            top_k: Number of top entities to return
            
        Returns:
            List of central entities with their centrality scores
        """
        try:
            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            closeness_centrality = nx.closeness_centrality(self.graph)
            
            # Combine scores
            entities = []
            for node in self.graph.nodes():
                score = (
                    degree_centrality.get(node, 0) * 0.4 +
                    betweenness_centrality.get(node, 0) * 0.3 +
                    closeness_centrality.get(node, 0) * 0.3
                )
                
                entities.append({
                    "entity": node,
                    "combined_score": score,
                    "degree_centrality": degree_centrality.get(node, 0),
                    "betweenness_centrality": betweenness_centrality.get(node, 0),
                    "closeness_centrality": closeness_centrality.get(node, 0),
                    "degree": self.graph.degree(node)
                })
            
            # Sort by combined score
            entities.sort(key=lambda x: x["combined_score"], reverse=True)
            
            return entities[:top_k]
            
        except Exception as e:
            logger.error(f"Error calculating central entities: {str(e)}")
            return []
    
    def find_communities(self) -> Dict:
        """
        Find communities in the knowledge graph
        
        Returns:
            Dictionary mapping community IDs to entity lists
        """
        try:
            # Convert to undirected graph for community detection
            undirected_graph = self.graph.to_undirected()
            
            # Use connected components as communities
            communities = {}
            for i, component in enumerate(nx.connected_components(undirected_graph)):
                communities[f"community_{i}"] = list(component)
            
            logger.info(f"Found {len(communities)} communities")
            return communities
            
        except Exception as e:
            logger.error(f"Error finding communities: {str(e)}")
            return {}
    
    def clear_graph(self) -> bool:
        """
        Clear all data from the knowledge graph
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.graph.clear()
            
            # Remove files
            if self.graph_file.exists():
                self.graph_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            
            logger.info("Knowledge graph cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing graph: {str(e)}")
            return False

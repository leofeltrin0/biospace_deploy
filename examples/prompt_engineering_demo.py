"""
Prompt Engineering Demo for NASA Space Apps Hackathon MVP
Demonstrates structured responses with references and theme classification
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.prompt_engineering import PromptEngineer


def demo_prompt_engineering():
    """Demonstrate prompt engineering features"""
    print("🚀 NASA Space Apps Hackathon MVP - Prompt Engineering Demo")
    print("=" * 60)
    
    # Initialize prompt engineer
    config = {
        "reference_extraction": True,
        "theme_classification": True,
        "structured_output": True,
        "themes": ["biotechnology", "neuroscience", "biochemistry", "ecology", "microbiology", "genetics"]
    }
    
    prompt_engineer = PromptEngineer(config)
    
    # Sample queries and retrieved chunks
    demo_cases = [
        {
            "query": "What are the effects of microgravity on C. elegans development?",
            "chunks": [
                {
                    "document_id": "space_biology_2023.pdf",
                    "chunk_id": "chunk_1",
                    "text": "Microgravity exposure significantly affects C. elegans development, leading to altered gene expression patterns and developmental delays. The nematode shows increased stress response and changes in muscle development under space conditions.",
                    "metadata": {
                        "authors": "Smith, J. et al.",
                        "publication_date": "2023-03-15",
                        "journal": "Space Biology Research"
                    },
                    "similarity_score": 0.95
                },
                {
                    "document_id": "nematode_studies_2022.pdf",
                    "chunk_id": "chunk_2", 
                    "text": "C. elegans exhibits behavioral changes in microgravity, including altered locomotion patterns and feeding behavior. These changes are associated with modifications in neural circuit development.",
                    "metadata": {
                        "authors": "Johnson, A. and Brown, B.",
                        "publication_date": "2022-11-20",
                        "journal": "Gravitational Biology"
                    },
                    "similarity_score": 0.87
                }
            ],
            "user_type": "scientist"
        },
        {
            "query": "How does spaceflight affect astronaut health?",
            "chunks": [
                {
                    "document_id": "astronaut_health_2023.pdf",
                    "chunk_id": "chunk_1",
                    "text": "Spaceflight induces significant physiological changes in astronauts, including muscle atrophy, bone density loss, and cardiovascular deconditioning. These effects are primarily due to microgravity exposure and radiation.",
                    "metadata": {
                        "authors": "Williams, M. et al.",
                        "publication_date": "2023-01-10",
                        "journal": "Space Medicine"
                    },
                    "similarity_score": 0.92
                }
            ],
            "user_type": "manager"
        },
        {
            "query": "What happens to plants in space?",
            "chunks": [
                {
                    "document_id": "plant_biology_space.pdf",
                    "chunk_id": "chunk_1",
                    "text": "Plants grown in space show altered growth patterns, including changes in root orientation and stem elongation. These adaptations help plants survive in the unique space environment.",
                    "metadata": {
                        "authors": "Garcia, L. and Chen, K.",
                        "publication_date": "2023-02-28",
                        "journal": "Plant Space Biology"
                    },
                    "similarity_score": 0.89
                }
            ],
            "user_type": "layperson"
        }
    ]
    
    print("\n📋 Available Scientific Themes:")
    for i, theme in enumerate(prompt_engineer.themes, 1):
        print(f"  {i}. {theme}")
    
    print(f"\n🔧 Prompt Engineering Configuration:")
    print(f"  - Reference Extraction: {prompt_engineer.reference_extraction}")
    print(f"  - Theme Classification: {prompt_engineer.theme_classification}")
    print(f"  - Structured Output: {prompt_engineer.structured_output}")
    
    # Process each demo case
    for i, case in enumerate(demo_cases, 1):
        print(f"\n{'='*60}")
        print(f"📝 Demo Case {i}: {case['user_type'].upper()} User")
        print(f"Query: {case['query']}")
        print(f"Retrieved Chunks: {len(case['chunks'])}")
        
        # Build prompt
        prompt = prompt_engineer.build_prompt(
            case['query'], 
            case['chunks'], 
            case['user_type']
        )
        
        print(f"\n🔧 Generated Prompt Length: {len(prompt)} characters")
        print(f"Prompt Preview: {prompt[:200]}...")
        
        # Simulate structured response (without actual LLM call)
        mock_response = {
            "answer": f"Based on the research documents, {case['query'].lower()} shows significant findings in space biology research.",
            "references": [
                {
                    "file": chunk["document_id"],
                    "authors": chunk["metadata"].get("authors", "Not available"),
                    "date": chunk["metadata"].get("publication_date", "Not available"),
                    "relevance_score": chunk["similarity_score"]
                }
                for chunk in case['chunks']
            ],
            "theme": prompt_engineer._classify_theme_from_content(case['query']),
            "confidence": 0.85,
            "key_findings": [
                "Significant effects observed in space environment",
                "Multiple physiological changes documented",
                "Research provides insights for future missions"
            ]
        }
        
        # Parse and validate response
        parsed_response = prompt_engineer.parse_model_response(json.dumps(mock_response))
        
        print(f"\n📊 Structured Response:")
        print(f"  Answer: {parsed_response['answer'][:100]}...")
        print(f"  Theme: {parsed_response['theme']}")
        print(f"  Confidence: {parsed_response['confidence']}")
        print(f"  References: {len(parsed_response['references'])}")
        print(f"  Key Findings: {len(parsed_response['key_findings'])}")
        
        # Show references
        print(f"\n📚 References:")
        for j, ref in enumerate(parsed_response['references'], 1):
            print(f"  {j}. {ref['file']} - {ref['authors']} ({ref['date']}) - Relevance: {ref['relevance_score']:.2f}")
    
    # Demonstrate theme classification
    print(f"\n{'='*60}")
    print("🎯 Theme Classification Examples:")
    
    test_queries = [
        "What are the genetic effects of space radiation?",
        "How does microgravity affect neural development?",
        "What is the role of proteins in space adaptation?",
        "How do microbial communities change in space?",
        "What are the ecological implications of space habitats?"
    ]
    
    for query in test_queries:
        theme = prompt_engineer._classify_theme_from_content(query)
        print(f"  Query: {query}")
        print(f"  Classified Theme: {theme}")
        print()
    
    # Demonstrate metadata extraction
    print(f"{'='*60}")
    print("📋 Metadata Extraction Examples:")
    
    sample_chunks = [
        {
            "document_id": "research_paper_1.pdf",
            "text": "Authors: Smith, J. et al. (2023) published this study on space biology.",
            "metadata": {"authors": "Smith, J. et al.", "date": "2023"}
        },
        {
            "document_id": "research_paper_2.pdf", 
            "text": "This research was conducted by Johnson, A. and Brown, B. in 2022.",
            "metadata": {}
        }
    ]
    
    metadata_list = prompt_engineer.extract_metadata_from_chunks(sample_chunks)
    
    for i, metadata in enumerate(metadata_list, 1):
        print(f"  Chunk {i}:")
        print(f"    File: {metadata['file']}")
        print(f"    Authors: {metadata['authors']}")
        print(f"    Date: {metadata['date']}")
        print()
    
    print("✅ Prompt Engineering Demo Completed!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Structured JSON responses with references")
    print("  ✓ Theme classification for scientific domains")
    print("  ✓ Metadata extraction from documents")
    print("  ✓ User profile adaptation")
    print("  ✓ Confidence scoring and key findings")
    print("  ✓ Reference relevance scoring")


if __name__ == "__main__":
    demo_prompt_engineering()

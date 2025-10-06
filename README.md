# NASA Space Apps Hackathon MVP - Space Mission Knowledge Engine

An intelligent knowledge engine for space biology research that processes +600 PDFs, extracts relevant knowledge through text chunking, embeddings, and knowledge graph generation, and exposes an API for front-end dashboards.

## 🚀 Features

- **PDF Processing**: Extract and clean text from +600 space biology research papers
- **Intelligent Chunking**: Semantic text splitting optimized for space biology content
- **Advanced Embeddings**: HuggingFace-based dense vector representations
- **Vector Storage**: FAISS/ChromaDB abstraction layer for scalable retrieval
- **Knowledge Graph**: Entity and relation extraction for structured insights
- **Mission Engine**: RAG + KG combination for intelligent query answering
- **Adaptive Generation**: Tone-aware responses for different user profiles
- **Prompt Engineering**: Structured responses with reference metadata and theme classification
- **RESTful API**: FastAPI-based endpoints for front-end integration
- **Front-end Integration**: React-based UI with real-time API integration
- **Modular Architecture**: Easily replaceable and scalable components

## 📁 Project Structure

```
mvp_chatbot/
├── modules/                    # Core modules
│   ├── pdf_processing/         # PDF text extraction and cleaning
│   ├── chunking/              # Semantic text splitting
│   ├── embeddings/            # HuggingFace embedding generation
│   ├── vectorstore/           # Vector database abstraction
│   ├── kg_extraction/         # Knowledge graph extraction
│   ├── kg_store/             # Knowledge graph storage
│   ├── mission_engine/        # RAG + KG intelligence
│   ├── adaptive_generator/    # Tone-aware response generation
│   ├── prompt_engineering/   # Structured responses with metadata
│   └── api/                  # FastAPI application
├── front-end/                 # React front-end application
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── context/          # React contexts (API, Paper)
│   │   ├── services/         # API service layer
│   │   ├── config/           # Configuration
│   │   └── pages/            # Application pages
│   ├── package.json          # Front-end dependencies
│   └── vite.config.ts        # Build configuration
├── config/                   # Configuration files
├── tests/                   # Test suite
├── data/                    # Data directories
├── logs/                    # Log files
├── main.py                  # CLI orchestrator
├── requirements.txt         # Dependencies
├── Dockerfile              # Container configuration
└── docker-compose.yml      # Multi-container setup
```

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- OpenAI API key
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mvp_chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

6. **Run the system**
   ```bash
   # Process documents and build knowledge base
   python main.py --mode ingest
   
   # Start the API server
   python main.py --mode serve
   
   # Interactive chat interface
   python main.py --mode chat
   ```

## 🌐 Front-end Integration

The React front-end is fully integrated with the back-end API, providing a rich user interface for the Space Mission Knowledge Engine.

### Front-end Setup

1. **Install front-end dependencies**
   ```bash
   cd front-end
   npm install
   ```

2. **Configure environment**
   ```bash
   # Create .env file in front-end directory
   echo "VITE_API_BASE_URL=http://localhost:8000" > .env
   ```

3. **Start the front-end development server**
   ```bash
   npm run dev
   ```

4. **Access the application**
   - Front-end: `http://localhost:8080`
   - Back-end API: `http://localhost:8000`

### Front-end Features

- **🔬 User Profile Adaptation**: Scientist, Manager, and Layperson modes
- **📊 Structured Responses**: Display API responses with references and themes
- **📈 System Monitoring**: Real-time API health and statistics
- **🔗 Knowledge Graph Visualization**: Interactive graph display
- **📚 Reference Management**: PDF document references with metadata
- **🎨 Modern UI**: Glass-morphism design with responsive layout

### Integration Components

- **API Service Layer**: Complete TypeScript API integration
- **React Context**: State management for API responses
- **Structured Response Display**: Rich response visualization
- **System Status Monitoring**: Real-time health checks
- **User Profile Management**: Adaptive interface based on user type

For detailed integration information, see [Front-end Integration Guide](front-end/INTEGRATION_GUIDE.md).

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build manually**
   ```bash
   docker build -t space-mission-engine .
   docker run -p 8000:8000 -e OPENAI_API_KEY=your_key space-mission-engine
   ```

## 🎯 Usage

### CLI Commands

```bash
# Process all PDFs and build complete knowledge base
python main.py --mode ingest

# Generate embeddings for existing chunks
python main.py --mode embed

# Build knowledge graph from chunks
python main.py --mode build_kg

# Start FastAPI server
python main.py --mode serve

# Interactive chat interface
python main.py --mode chat

# With verbose logging
python main.py --mode serve --verbose
```

### API Endpoints

Once the server is running (default: http://localhost:8000):

- **GET /** - API information
- **GET /health** - Health check
- **POST /ingest** - Trigger document processing
- **GET /query** - Query mission insights
- **GET /query-adaptive** - Adaptive tone query
- **GET /query-structured** - Structured response with references and theme
- **GET /graph** - Knowledge graph data
- **GET /stats** - System statistics
- **GET /profiles** - Available user profiles
- **GET /themes** - Available scientific themes
- **POST /classify-theme** - Classify query theme
- **GET /prompt-engineering/stats** - Prompt engineering statistics

### Example API Usage

```python
import requests

# Query the system
response = requests.get("http://localhost:8000/query", 
                       params={"query": "What are the effects of microgravity on C. elegans?"})
print(response.json())

# Get structured response with references and theme
structured_response = requests.get("http://localhost:8000/query-structured",
                                 params={"query": "How does spaceflight affect astronaut health?",
                                        "user_profile": "scientist"})
print(structured_response.json())

# Get available themes
themes = requests.get("http://localhost:8000/themes")
print(themes.json())

# Get knowledge graph data
graph_data = requests.get("http://localhost:8000/graph")
print(graph_data.json())
```

## 🎯 Prompt Engineering Features

The system includes advanced prompt engineering capabilities that ensure every response includes:

### 📚 Reference Metadata
- **File names** extracted from document metadata
- **Author information** parsed from documents or metadata
- **Publication dates** when available
- **Relevance scores** for each reference

### 🏷️ Theme Classification
Automatic categorization into scientific domains:
- **Biotechnology**: Genetic engineering, recombinant DNA, cloning
- **Neuroscience**: Neural networks, brain development, cognitive studies
- **Biochemistry**: Protein studies, enzyme research, molecular biology
- **Ecology**: Ecosystem studies, environmental research
- **Microbiology**: Bacterial studies, microbial communities
- **Genetics**: Gene studies, genome research, inheritance

### 📊 Structured Responses
Every response includes:
```json
{
  "answer": "Comprehensive answer based on retrieved documents",
  "references": [
    {
      "file": "research_paper.pdf",
      "authors": "Smith, J. et al.",
      "date": "2023-03-15",
      "relevance_score": 0.95
    }
  ],
  "theme": "biotechnology",
  "confidence": 0.89,
  "key_findings": ["Finding 1", "Finding 2"],
  "user_profile": "scientist"
}
```

### 🔧 Demo Script
Run the prompt engineering demo:
```bash
python examples/prompt_engineering_demo.py
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_pdf_processing.py
pytest tests/test_chunking.py
pytest tests/test_vectorstore.py
pytest tests/test_prompt_engineering.py
pytest tests/test_api.py

# Run with coverage
pytest --cov=modules

# Run integration tests
pytest -m integration
```

## 📊 Configuration

The system is configured via `config/config.yaml`. Key settings include:

- **API Configuration**: Host, port, CORS settings
- **PDF Processing**: Text cleaning options, supported formats
- **Chunking**: Chunk size, overlap, separators
- **Embeddings**: Model selection, batch size, caching
- **Vector Store**: Backend selection (FAISS/Chroma), persistence
- **Knowledge Graph**: Entity types, relation types, confidence thresholds
- **Logging**: Log levels, file rotation, formatting

## 🏗️ Architecture

### Data Flow

1. **PDF Processing**: Extract text from PDFs → Clean and normalize
2. **Chunking**: Split documents → Semantic chunks with metadata
3. **Embeddings**: Generate vectors → Store in vector database
4. **KG Extraction**: Extract entities/relations → Build knowledge graph
5. **Query Processing**: Retrieve relevant chunks → Generate insights
6. **Adaptive Response**: Detect user profile → Generate appropriate response

### Components

- **PDFProcessor**: Multi-method text extraction (PyMuPDF, pdfplumber, pypdf)
- **TextChunker**: Semantic-aware chunking with overlap
- **EmbeddingGenerator**: HuggingFace model integration with caching
- **VectorStore**: FAISS/ChromaDB abstraction layer
- **KGExtractor**: spaCy + OpenAI entity/relation extraction
- **KGStore**: NetworkX-based graph storage and querying
- **MissionEngine**: RAG + KG intelligence combination
- **AdaptiveGenerator**: User profile detection and tone adaptation
- **PromptEngineer**: Structured responses with reference metadata and theme classification

## 🚀 Deployment

### AWS Deployment (Recommended)

1. **EC2 Instance**
   ```bash
   # Launch t2.micro (free tier)
   # Install Docker
   sudo yum update -y
   sudo yum install -y docker
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   ```

2. **S3 Storage**
   ```bash
   # Create S3 bucket for data persistence
   aws s3 mb s3://space-mission-engine-data
   ```

3. **Deploy with Docker**
   ```bash
   # Clone and deploy
   git clone <repository-url>
   cd mvp_chatbot
   docker-compose up -d
   ```

### Local Production

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn modules.api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📈 Performance

- **PDF Processing**: ~10-50 PDFs/minute (depending on size)
- **Embedding Generation**: ~100-500 chunks/minute
- **Query Response**: <2 seconds for typical queries
- **Memory Usage**: ~2-4GB for full dataset
- **Storage**: ~1-2GB for embeddings + knowledge graph

## 🔧 Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

2. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Memory Issues**
   - Reduce batch sizes in config
   - Use smaller embedding models
   - Process documents in smaller batches

4. **PDF Processing Errors**
   - Check PDF file integrity
   - Try different extraction methods
   - Verify file permissions

### Logs

Check logs in `logs/space_mission_engine.log` for detailed error information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is part of the NASA Space Apps Hackathon 2024.

## 🙏 Acknowledgments

- NASA Space Apps Challenge
- OpenAI for GPT models
- HuggingFace for embedding models
- FastAPI for the web framework
- The space biology research community

---

**Built with ❤️ for NASA Space Apps Hackathon 2024**

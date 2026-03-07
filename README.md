# Nutrition RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for analyzing nutrition data with automatic visualization. Features a modern Streamlit web interface and Docker support.

## Features

- **Modern Web UI**: Streamlit interface with real-time chat
- **Semantic Search**: Query 3,995 nutrition documents using natural language
- **GPT-4 Powered**: Advanced question answering with Azure OpenAI
- **Auto-Visualization**: Live chart generation for ranking queries
- **Smart Comparisons**: Balanced multi-brand retrieval
- **Multiple Chart Types**: Bar, pie, and line charts in-browser
- **Docker Ready**: Full containerization support

## Quick Start

### Create Environment File

```bash
# Create .env file with your Azure OpenAI credentials
cat > .env << EOF
DIAL_API_KEY=your-api-key
AZURE_ENDPOINT=https://ai-proxy.lab.epam.com
API_VERSION=2023-08-01-preview
MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small-1
EOF
```

### Using Docker (Recommended)

```bash
docker-compose up -d
```

Access the application at **http://localhost:8501**

### Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

The system will automatically index documents on first run (~2-3 minutes).

## Usage Examples

### Ranking Queries (Generates Charts)

```
> top 10 burgers by protein
> top 5 pizzas by calories
> highest fat in chicken products
```

### Comparison Queries

```
> compare McDonald's vs Burger King
> difference between KFC and Subway chicken
```

### Regular Queries

```
> how many calories in a McDonald's Cheeseburger?
> what's the healthiest pizza?
```

## Project Structure

```
rag_task/
├── app/                           # Core application modules
│   ├── config.py                  # Configuration management
│   ├── constants.py               # Application constants
│   ├── document_loader.py         # Document loading
│   ├── text_chunker.py            # Text chunking
│   ├── embedder.py                # Embedding generation
│   ├── vector_store.py            # ChromaDB interface
│   ├── retriever.py               # RAG query engine
│   ├── comparison_retriever.py    # Multi-brand comparison
│   ├── chat_client.py             # OpenAI chat client
│   └── visual_rag_engine.py       # Visualization engine
├── resources/                     # Nutrition data (JSONL)
├── chroma_db/                     # Vector database (persisted)
├── app.py                         # Streamlit application
├── Dockerfile                     # Docker image definition
├── docker-compose.yml             # Docker orchestration
└── requirements.txt               # Python dependencies
```

## Technical Stack

- **UI**: Streamlit
- **LLM**: Azure OpenAI (GPT-4)
- **Embeddings**: text-embedding-3-small
- **Vector Database**: ChromaDB
- **Visualization**: Matplotlib
- **Framework**: LangChain (text splitting)

## Interface

The Streamlit interface provides:

- **Chat Interface**: Natural language question answering
- **Example Buttons**: One-click query execution for common questions
- **Live Visualizations**: Charts generated and displayed automatically
- **Database Stats**: Indexing status in sidebar
- **Clear History**: One-click chat reset

## Chart Customization

Specify chart type in your query:

```
top 10 burgers by protein as pie chart
top 5 pizzas by calories as line chart
```

Default chart type is bar chart.

## Dataset

- **Documents**: 3,995 nutrition profiles
- **Brands**: 186 fast-food chains
- **Product Types**: Burgers, pizzas, chicken, tacos, sandwiches, salads

## Architecture

The system follows a modular RAG architecture:

1. **Document Processing**: JSONL → Chunks (600 chars) → Embeddings (text-embedding-3-small)
2. **Storage**: ChromaDB persistent vector store with metadata filtering
3. **Retrieval**: Semantic search with comparison detection and balanced multi-brand retrieval
4. **Generation**: GPT-4 for context-aware answer synthesis
5. **Visualization**: On-the-fly matplotlib chart generation for ranking queries

## Docker Management

```bash
# Start application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop application
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Reset database (remove all indexed data)
docker-compose down -v
```

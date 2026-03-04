# Nutrition RAG System

A Retrieval-Augmented Generation (RAG) system for analyzing nutrition data with automatic visualization capabilities.

## Features

- **Semantic Search**: Query 3,995 nutrition documents using natural language
- **Conversational AI**: GPT-4 powered question answering
- **Auto-Visualization**: Automatic chart generation for ranking queries
- **Smart Comparisons**: Balanced multi-brand retrieval for comparison queries
- **Multiple Chart Types**: Bar, pie, and line charts

## Quick Start

### Installation

```bash
pip3 install -r requirements.txt
```

### Configuration

Create a `.env` file with your Azure OpenAI credentials:

```
DIAL_API_KEY=your-api-key
AZURE_ENDPOINT=https://ai-proxy.lab.epam.com
API_VERSION=2023-08-01-preview
MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small-1
```

### Run

```bash
python3 rag_main.py
```

The system will automatically index documents on first run (one-time operation). Subsequent runs start instantly.

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
├── app/
│   ├── config.py                 # Configuration management
│   ├── constants.py              # Application constants
│   ├── document_loader.py        # Document loading
│   ├── text_chunker.py           # Text chunking
│   ├── embedder.py               # Embedding generation
│   ├── vector_store.py           # ChromaDB interface
│   ├── retriever.py              # RAG query engine
│   ├── comparison_retriever.py   # Multi-brand comparison
│   ├── chat_client.py            # OpenAI chat client
│   └── visual_rag_engine.py      # Visualization engine
├── resources/                     # Data files
├── visualizations/                # Generated charts
├── chroma_db/                     # Vector database
└── rag_main.py                    # Main application
```

## Technical Stack

- **LLM**: Azure OpenAI (GPT-4)
- **Embeddings**: text-embedding-3-small
- **Vector Database**: ChromaDB
- **Charts**: Matplotlib
- **Framework**: LangChain (text splitting)

## Commands

| Command | Description |
|---------|-------------|
| `python3 rag_main.py` | Start the system |
| `python3 rag_main.py --reindex` | Force rebuild database |
| `stats` (in chat) | Show database statistics |
| `quit` or `exit` (in chat) | Exit application |

## Chart Customization

Specify chart type in your query:

```
> top 10 burgers by protein as pie chart
> top 5 pizzas by calories as line chart
```

Default chart type is bar chart.

## Dataset

- **Documents**: 3,995 nutrition profiles
- **Brands**: 186 fast-food chains
- **Product Types**: Burgers, pizzas, chicken, tacos, sandwiches, salads

## Architecture

The system follows a modular RAG architecture:

1. **Document Processing**: JSONL → Chunks → Embeddings
2. **Storage**: ChromaDB persistent vector store
3. **Retrieval**: Semantic search with metadata filtering
4. **Generation**: GPT-4 for answer synthesis
5. **Visualization**: Automated chart generation for ranking queries


"""
Streamlit App for Nutrition RAG System
"""

import logging
from pathlib import Path
from typing import Tuple

import streamlit as st

from app.config import ConfigurationLoader
from app.document_loader import JSONLDocumentLoader
from app.text_chunker import RecursiveTextChunker
from app.embedder import AzureEmbedder
from app.vector_store import ChromaVectorStore
from app.retriever import Retriever
from app.visual_rag_engine import VisualRAGEngine
from app.chat_client import AzureChatClient
from app.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, MIN_CHUNK_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
for lib in ['httpx', 'openai', 'chromadb']:
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Constants
DATA_PATH = Path("./resources/Nutrition_RAG_Dataset.jsonl")
DB_PATH = "./chroma_db"
COLLECTION_NAME = "nutrition_rag"

st.set_page_config(
    page_title="Nutrition RAG System",
    page_icon="🥗",
    layout="centered"
)

st.title("🥗 Nutrition RAG System")
st.markdown("Ask questions about nutrition data and get instant answers with visualizations!")


@st.cache_resource
def initialize_rag_pipeline() -> Tuple[VisualRAGEngine, ChromaVectorStore]:
    """Initialize and cache the RAG pipeline."""
    logger.info("Initializing RAG pipeline")

    config = ConfigurationLoader().load()
    embedder = AzureEmbedder(config)
    vector_store = ChromaVectorStore(COLLECTION_NAME, DB_PATH)
    retriever = Retriever(embedder, vector_store)
    chat_client = AzureChatClient(config)
    rag_engine = VisualRAGEngine(retriever, chat_client)

    # Index documents if needed
    if vector_store.get_stats()['total_chunks'] == 0:
        with st.spinner("Indexing documents (first run only)..."):
            documents = JSONLDocumentLoader(str(DATA_PATH)).load()
            chunker = RecursiveTextChunker(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                min_chunk_size=MIN_CHUNK_SIZE
            )
            chunks = chunker.chunk_documents(documents)
            embeddings = embedder.embed_chunks(chunks)
            vector_store.add_chunks(chunks, embeddings)
            logger.info("Document indexing complete")

    logger.info("RAG pipeline initialized successfully")
    return rag_engine, vector_store


# Initialize RAG pipeline
try:
    rag_engine, vector_store = initialize_rag_pipeline()
    stats = vector_store.get_stats()
except Exception as e:
    st.error(f"⚠️ Error initializing system: {e}")
    logger.exception("Initialization failed")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        f"""
        RAG system analyzing 3,995 fast-food nutrition profiles.

        **Database:** {stats['total_chunks']} chunks indexed

        **Features:**
        - Semantic search with GPT-4
        - Auto chart generation
        - Multi-brand comparisons
        """
    )

    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# Example questions
st.subheader("Try these:")

examples = {
    "Rankings *(auto-generates charts: bar, pie, or line)*": [
        "Top 10 burgers by protein",
        "Highest calories in pizza",
        "Top 5 chicken by fat",
        "Best pizza for protein"
    ],
    "Comparisons": [
        "Compare McDonald's vs Burger King",
        "KFC vs Wendy's chicken"
    ],
    "Specific": [
        "Calories in Big Mac",
        "Protein in Whopper"
    ]
}

for category, queries in examples.items():
    st.markdown(f"**{category}**")
    cols = st.columns(2)
    for i, query in enumerate(queries):
        if cols[i % 2].button(query, key=f"{category}_{i}", use_container_width=True):
            st.session_state.user_input = query

st.divider()

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chart" in message:
            st.pyplot(message["chart"])

# Chat input
prompt = st.chat_input("Ask about nutrition...")
if prompt:
    st.session_state.user_input = prompt

# Process user input
if "user_input" in st.session_state:
    user_message = st.session_state.user_input
    del st.session_state.user_input

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Get AI response
    with st.spinner("Searching and generating answer..."):
        try:
            response = rag_engine.query(user_message, top_k=20)

            # Prepare response message
            message_data = {"role": "assistant", "content": response['answer']}

            # Add chart if available
            if 'visualization' in response:
                message_data["chart"] = response['visualization']
                chart_type = response.get('chart_type', 'bar').capitalize()
                product_count = len(response.get('extracted_data', []))
                message_data["content"] += f"\n\n**Chart:** {chart_type} | **Products:** {product_count}"

            # Add sources info
            sources_info = f"\n\n📚 **Sources:** {response['num_sources']} chunks retrieved"
            message_data["content"] += sources_info

            st.session_state.messages.append(message_data)

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            logger.exception("Query processing failed")

    st.rerun()

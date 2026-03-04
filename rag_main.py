"""
RAG Pipeline with Auto-Visualization

Intelligent nutrition Q&A system with automatic chart generation.

Usage:
    python3 rag_main.py
    python3 rag_main.py --reindex  # Force rebuild database
"""

import sys
import logging

from app.config import ConfigurationLoader
from app.document_loader import JSONLDocumentLoader
from app.text_chunker import RecursiveTextChunker
from app.embedder import AzureEmbedder
from app.vector_store import ChromaVectorStore
from app.retriever import Retriever
from app.visual_rag_engine import VisualRAGEngine
from app.chat_client import AzureChatClient
from app.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, MIN_CHUNK_SIZE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress verbose logs from external libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline orchestrator.

    Manages the full lifecycle of document processing, embedding,
    storage, and retrieval for question answering.
    """

    def __init__(
        self,
        data_path: str = "./resources/Nutrition_RAG_Dataset.jsonl",
        collection_name: str = "nutrition_rag",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize RAG pipeline.

        Args:
            data_path: Path to JSONL data file
            collection_name: ChromaDB collection name
            persist_directory: Directory for ChromaDB persistence
        """
        self.data_path = data_path
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        logger.info("Loading configuration")
        config_loader = ConfigurationLoader()
        self.config = config_loader.load()

        logger.info("Initializing components")
        self.embedder = AzureEmbedder(self.config)
        self.vector_store = ChromaVectorStore(collection_name, persist_directory)
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.chat_client = AzureChatClient(self.config)

        self.rag_engine = VisualRAGEngine(self.retriever, self.chat_client, visualization_dir="visualizations")

        logger.info("RAG Pipeline initialized successfully")

    def index_documents(self, rebuild: bool = False) -> None:
        """
        Index documents into the vector database.

        Args:
            rebuild: If True, clear existing data before indexing
        """
        logger.info("=" * 60)
        logger.info("INDEXING DOCUMENTS")
        logger.info("=" * 60)

        stats = self.vector_store.get_stats()
        if stats['total_chunks'] > 0 and not rebuild:
            logger.info(f"Collection already contains {stats['total_chunks']} chunks")
            logger.info("Use --rebuild flag to rebuild the index")
            return

        if rebuild:
            logger.info("Rebuilding index (clearing existing data)")
            self.vector_store.clear()

        logger.info("Step 1: Loading documents")
        loader = JSONLDocumentLoader(self.data_path)
        documents = loader.load()

        logger.info("Step 2: Chunking documents")
        chunker = RecursiveTextChunker(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            min_chunk_size=MIN_CHUNK_SIZE
        )
        chunks = chunker.chunk_documents(documents)

        logger.info("Step 3: Creating embeddings")
        embeddings = self.embedder.embed_chunks(chunks)

        logger.info("Step 4: Storing in ChromaDB")
        self.vector_store.add_chunks(chunks, embeddings)

        stats = self.vector_store.get_stats()
        logger.info("=" * 60)
        logger.info("INDEXING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total chunks: {stats['total_chunks']}, Collection: {stats['collection_name']}")

    def interactive_query(self) -> None:
        """
        Interactive Q&A interface.

        Allows users to ask questions and get answers from the knowledge base.
        """
        print("=" * 60)
        print("INTERACTIVE Q&A MODE WITH AUTO-VISUALIZATION")
        print("=" * 60)
        print("\nAsk questions about nutrition and food products.")
        print("\n💡 Special feature: Ranking questions automatically generate charts!")
        print("   Examples:")
        print("   - 'top 10 burgers by protein'")
        print("   - 'top 5 pizzas by calories'")
        print("   - 'highest protein in chicken products'")
        print("\n📊 Chart types (specify in your question):")
        print("   - Default: bar chart")
        print("   - Add 'pie chart': 'top 10 burgers by protein as pie chart'")
        print("   - Add 'line chart': 'top 5 pizzas by calories as line chart'")
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - Type 'quit' or 'exit' to stop")
        print("  - Type 'stats' to see database statistics")
        print("=" * 60)

        while True:
            try:
                # Get user input
                print("\n" + "─" * 60)
                question = input("\nYour question: ").strip()

                if not question:
                    continue

                # Handle commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if question.lower() == 'stats':
                    stats = self.vector_store.get_stats()
                    print("\nDatabase Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                # Process question with RAG
                print("\nSearching knowledge base...")
                response = self.rag_engine.query(question, top_k=20)

                # Display answer
                print("\n" + "=" * 60)
                print("ANSWER")
                print("=" * 60)
                print(f"\n{response['answer']}\n")

                # Display visualization if generated
                if 'visualization' in response:
                    print("─" * 60)
                    print("📊 VISUALIZATION GENERATED")
                    print("─" * 60)
                    chart_type = response.get('chart_type', 'bar').capitalize()
                    print(f"Chart type: {chart_type} chart")
                    print(f"Chart saved to: {response['visualization']}")
                    print(f"Products analyzed: {len(response.get('extracted_data', []))}")
                    print("\n💡 Tip: Open the file to see the chart!")

                # Display sources
                print("\n" + "─" * 60)
                print(f"Sources ({response['num_sources']} chunks retrieved):")
                print("─" * 60)
                for i, source in enumerate(response['sources'][:3], 1):
                    print(f"\n[Source {i}] (relevance: {1 - source['distance']:.2f})")
                    # Show more content (300 chars instead of 150) to see protein info
                    content_preview = source['content'][:300] + "..." if len(source['content']) > 300 else source['content']
                    print(f"Content: {content_preview}")
                    metadata = source.get('metadata', {})
                    if 'brand' in metadata:
                        print(f"Brand: {metadata.get('brand')}, Product: {metadata.get('product', 'N/A')}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")

    def run(self, force_reindex: bool = False) -> None:
        """
        Run the RAG pipeline.

        Args:
            force_reindex: Force rebuild even if database exists
        """
        stats = self.vector_store.get_stats()
        needs_indexing = stats['total_chunks'] == 0 or force_reindex

        if needs_indexing:
            if force_reindex:
                logger.info("Force re-indexing requested")
            else:
                logger.info("Database is empty. Indexing documents")
            self.index_documents(rebuild=True)
        else:
            logger.info(f"Database ready: {stats['total_chunks']} chunks indexed")

        self.interactive_query()


def main():
    """Main entry point."""
    force_reindex = '--reindex' in sys.argv or '--rebuild' in sys.argv

    print("\n" + "=" * 70)
    print("NUTRITION RAG SYSTEM WITH AUTO-VISUALIZATION")
    print("=" * 70)
    print("\nInitializing system...")

    try:
        pipeline = RAGPipeline(
            data_path="./resources/Nutrition_RAG_Dataset.jsonl",
            collection_name="nutrition_rag",
            persist_directory="./chroma_db"
        )

        pipeline.run(force_reindex=force_reindex)

    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        logger.info("Please check your .env file")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File Error: {e}")
        logger.info("Please check that resources/Nutrition_RAG_Dataset.jsonl exists")
        return 1
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        return 0
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

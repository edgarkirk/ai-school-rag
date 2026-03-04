"""
Retriever Module - Step 5 of RAG Pipeline

Provides high-level interface for retrieving relevant chunks
from the vector store based on user queries.
"""

from typing import List, Dict, Any, Optional
from app.embedder import Embedder
from app.vector_store import VectorStore
from app.comparison_retriever import ComparisonRetriever


class Retriever:
    """
    High-level retrieval interface for RAG.

    Combines embedding and vector store querying to retrieve
    relevant document chunks for a given query.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        """
        Initialize retriever.

        Args:
            embedder: Embedder to convert queries to vectors
            vector_store: Vector store to search
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User's question or search query
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters

        Returns:
            List of relevant chunks with content and metadata
        """
        # Embed the query
        query_embedding = self.embedder.embed_text(query)

        # Search vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata
        )

        return results

    def retrieve_as_context(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Retrieve relevant chunks and format as context string.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters
            include_metadata: Whether to include metadata in context

        Returns:
            Formatted context string for prompt injection
        """
        results = self.retrieve(query, top_k, filter_metadata)

        if not results:
            return "No relevant information found."

        # Format results as context
        context_parts = []
        for i, result in enumerate(results, 1):
            context_part = f"[Source {i}]\n{result['content']}"

            if include_metadata:
                # Add relevant metadata
                metadata = result.get('metadata', {})
                if 'brand' in metadata:
                    context_part += f"\n(Brand: {metadata['brand']}"
                    if 'product' in metadata:
                        context_part += f", Product: {metadata['product']}"
                    context_part += ")"

            context_parts.append(context_part)

        return "\n\n".join(context_parts)


class RAGQueryEngine:
    """
    Complete RAG query engine.

    Combines retrieval and generation to answer questions
    using retrieved context.
    """

    def __init__(self, retriever: Retriever, chat_client):
        """
        Initialize RAG query engine.

        Args:
            retriever: Retriever for fetching relevant chunks
            chat_client: Chat client for generation
        """
        self.retriever = retriever
        self.chat_client = chat_client
        self.comparison_retriever = ComparisonRetriever(retriever)

    def _format_results_as_context(self, results: List[Dict[str, Any]], include_metadata: bool = True) -> str:
        """
        Format retrieved results into context string.

        Args:
            results: List of retrieved chunks with metadata
            include_metadata: Whether to include metadata in context

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."

        context_parts = []
        for i, result in enumerate(results, 1):
            context_part = f"[Source {i}]\n{result['content']}"

            if include_metadata:
                # Add relevant metadata
                metadata = result.get('metadata', {})
                if 'brand' in metadata:
                    context_part += f"\n(Brand: {metadata['brand']}"
                    if 'product' in metadata:
                        context_part += f", Product: {metadata['product']}"
                    context_part += ")"

            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        only_first_chunks: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters
            only_first_chunks: If True, only retrieve first chunks (chunk_index=0) which contain full nutritional info

        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Add chunk_index filter to get first chunks only (which contain all nutritional info)
        if only_first_chunks:
            if filter_metadata is None:
                filter_metadata = {}
            filter_metadata['chunk_index'] = 0

        # Check if this is a comparison query and use appropriate retrieval
        results, compared_brands = self.comparison_retriever.retrieve_for_comparison(
            question,
            top_k_per_brand=top_k,
            filter_metadata=filter_metadata
        )

        # If comparison query but no results, inform user
        if compared_brands and not results:
            return {
                "answer": f"I don't have enough information to compare {' vs '.join(compared_brands)}. Please check if these brands exist in the database.",
                "sources": [],
                "context_used": ""
            }

        if not results:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "context_used": ""
            }

        # Format context from the retrieved results (don't do another search!)
        context = self._format_results_as_context(results, include_metadata=True)

        # Create RAG prompt with comparison-aware instructions
        comparison_note = ""
        if compared_brands:
            comparison_note = f"\nNote: This is a comparison query between {' and '.join(compared_brands)}. Make sure to include information about BOTH brands in your answer.\n"

        rag_prompt = f"""You are a professional nutrition analyst. Answer the user's question based on the provided context.

Context from knowledge base:
{context}
{comparison_note}
Instructions:
- Answer based ONLY on the provided context
- Extract numerical values and ranges from the text (e.g., "Protein is commonly 25–50 g" means protein range is 25-50g)
- When comparing brands or products:
  * Compare specific nutritional values (calories, protein, fat, carbs)
  * Highlight key differences
  * Provide clear recommendations if appropriate
  * Use data from BOTH/ALL brands mentioned in the question
- When comparing items by nutrients (e.g., "top 3 by protein", "top 10 burgers by calories"):
  * Use the MAXIMUM value from each range for ranking
  * Present results as a NUMBERED LIST ranked from highest to lowest
  * Format each item as: "1. Brand Product (Size): X-Y g protein [Source N]"
  * Include the specific range values, not just generic text
- For non-ranking questions, provide clear concise answers with specific values
- Always cite sources using [Source N] notation
- Use professional, factual language

User Question: {question}

Answer:"""

        # Generate answer
        messages = [{"role": "user", "content": rag_prompt}]
        answer = self.chat_client.send_message(messages)

        return {
            "answer": answer,
            "sources": results,
            "context_used": context,
            "num_sources": len(results)
        }

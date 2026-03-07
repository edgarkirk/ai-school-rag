"""
Comparison Retriever - Smart retrieval for comparison queries

Detects comparison queries and ensures balanced retrieval from multiple brands.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from app.constants import BRAND_MAPPINGS

logger = logging.getLogger(__name__)


class ComparisonRetriever:
    """
    Enhances retrieval for comparison queries by detecting brands
    and ensuring balanced retrieval from multiple entities.
    """

    def __init__(self, retriever):
        """
        Initialize comparison retriever.

        Args:
            retriever: Base retriever instance
        """
        self.retriever = retriever

    def _detect_comparison_query(self, query: str) -> Optional[Tuple[List[str], str]]:
        """
        Detect if query is a comparison and extract brand names.

        Args:
            query: User's question

        Returns:
            Tuple of (brand_list, comparison_type) or None
        """
        query_lower = query.lower()

        patterns = [
            r'compare\s+(.+?)\s+vs\.?\s+(.+?)(?:\s+on|\s+in|\s+for|$)',
            r'compare\s+(.+?)\s+and\s+(.+?)(?:\s+on|\s+in|\s+for|$)',
            r'(.+?)\s+vs\.?\s+(.+?)(?:\s+comparison|\s+on|\s+in|\s+for|$)',
            r'difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+on|\s+in|$)',
            r'which\s+is\s+better[,:]?\s+(.+?)\s+or\s+(.+?)(?:\s+for|$)',
        ]

        for pattern in patterns:
            if match := re.search(pattern, query_lower, re.IGNORECASE):
                brands = [match.group(1).strip(), match.group(2).strip()]

                # Clean up common words
                cleanup_words = [' on ', ' in ', ' for ', ' burger', ' burgers', ' product', ' products']
                brands = [
                    next((b.replace(word, '').strip() for word in cleanup_words if word in b), b)
                    for b in brands
                ]

                # Determine comparison type
                comp_type = (
                    'vs' if 'vs' in query_lower or 'versus' in query_lower
                    else 'difference' if 'difference' in query_lower
                    else 'compare'
                )

                return (brands, comp_type)

        return None

    def _normalize_brand_name(self, brand: str) -> str:
        """
        Normalize brand name for better matching.

        Args:
            brand: Raw brand name (might include product names)

        Returns:
            Normalized brand name
        """
        brand = brand.strip()
        brand_lower = brand.lower()

        if brand_lower in BRAND_MAPPINGS:
            return BRAND_MAPPINGS[brand_lower]

        for known_brand_lower, normalized_name in BRAND_MAPPINGS.items():
            if brand_lower.startswith(known_brand_lower + ' '):
                return normalized_name

        product_words = ['burger', 'burgers', 'chicken', 'nuggets', 'nugget',
                         'pizza', 'pizzas', 'sandwich', 'sandwiches', 'fries',
                         'taco', 'tacos', 'salad', 'salads', 'wrap', 'wraps']

        words = [w for w in brand.split() if w.lower() not in product_words]

        if words:
            result = ' '.join(words)
            if result.lower() in BRAND_MAPPINGS:
                return BRAND_MAPPINGS[result.lower()]
            return result.title()

        return brand.title()

    def retrieve_for_comparison(
        self,
        query: str,
        top_k_per_brand: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[List[str]]]:
        """
        Retrieve chunks optimized for comparison queries.

        Args:
            query: User's question
            top_k_per_brand: Number of chunks to retrieve per brand
            filter_metadata: Optional metadata filters

        Returns:
            Tuple of (results, brands) where brands is list of compared brands or None
        """
        # Check if this is a comparison query
        comparison_info = self._detect_comparison_query(query)

        if not comparison_info:
            # Not a comparison - use regular retrieval
            results = self.retriever.retrieve(query, top_k_per_brand * 2, filter_metadata)
            return results, None

        brands, comp_type = comparison_info
        normalized_brands = [self._normalize_brand_name(b) for b in brands]

        logger.info(f"Detected comparison query: {comp_type} - {' vs '.join(normalized_brands)}")

        all_results = []

        for brand in normalized_brands:
            if filter_metadata and len(filter_metadata) > 0:
                conditions = [{"brand": brand}]
                for key, value in filter_metadata.items():
                    conditions.append({key: value})
                brand_filter = {"$and": conditions}
            else:
                brand_filter = {"brand": brand}

            brand_results = self.retriever.retrieve(
                query,
                top_k_per_brand,
                brand_filter
            )

            logger.debug(f"{brand}: {len(brand_results)} chunks retrieved")
            all_results.extend(brand_results)

        if len(all_results) < len(normalized_brands):
            logger.warning("Some brands had no results, trying fuzzy matching")

            for brand in normalized_brands:
                brand_query = f"{brand} {query}"
                brand_results = self.retriever.retrieve(brand_query, top_k_per_brand, filter_metadata)
                filtered = [r for r in brand_results if brand.lower() in r.get('metadata', {}).get('brand', '').lower()]

                if filtered:
                    logger.debug(f"{brand} (fuzzy): {len(filtered)} chunks retrieved")
                    all_results.extend(filtered)

        logger.info(f"Total chunks retrieved: {len(all_results)}")

        return all_results, normalized_brands

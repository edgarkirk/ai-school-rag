"""
Visual RAG Engine - Combines chat with automatic visualization

Automatically generates visualizations for ranking queries.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .retriever import RAGQueryEngine
from .constants import (
    VALID_NUTRIENTS,
    VALID_PRODUCTS,
    BAR_CHART_SIZE,
    PIE_CHART_SIZE,
    LINE_CHART_SIZE,
    CHART_TYPE_BAR,
    CHART_TYPE_PIE,
    CHART_TYPE_LINE
)

logger = logging.getLogger(__name__)


class VisualRAGEngine(RAGQueryEngine):
    """
    Enhanced RAG engine that automatically generates visualizations
    for ranking and comparison queries.
    """

    def __init__(self, retriever, chat_client):
        super().__init__(retriever, chat_client)

    def _detect_ranking_query(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Detect if the question is asking for rankings.

        Returns:
            Dict with query_type, nutrient, top_n, filter, and chart_type, or None
        """
        question_lower = question.lower()

        chart_type = CHART_TYPE_BAR
        if any(keyword in question_lower for keyword in ['pie chart', 'pie']):
            chart_type = CHART_TYPE_PIE
        elif any(keyword in question_lower for keyword in ['line graph', 'line chart', 'line', 'trend']):
            chart_type = CHART_TYPE_LINE

        ranking_patterns = [
            r'top\s+(\d+)\s+(\w+)?\s*(?:by|with|in)\s+(\w+)',
            r'highest\s+(\w+)\s+in\s+(\w+)',
            r'best\s+(\w+)\s+for\s+(\w+)',
            r'most\s+(\w+)\s+in\s+(\w+)',
            r'(\d+)\s+(\w+)?\s*with\s+(?:the\s+)?(?:most|highest|best)\s+(\w+)',
        ]

        for pattern in ranking_patterns:
            match = re.search(pattern, question_lower)
            if match:
                groups = match.groups()

                top_n = next((int(g) for g in groups if g and g.isdigit()), 10)

                nutrient = None
                for g in groups:
                    if g and g in VALID_NUTRIENTS:
                        nutrient = g.rstrip('s')
                        if nutrient == 'carb':
                            nutrient = 'carbohydrates'
                        break

                product_filter = next((g.rstrip('s') for g in groups if g and g in VALID_PRODUCTS), None)

                if nutrient:
                    return {
                        'query_type': 'ranking',
                        'nutrient': nutrient,
                        'top_n': top_n,
                        'filter': product_filter,
                        'chart_type': chart_type
                    }

        return None

    def _get_nutrient_variations(self, nutrient: str) -> List[str]:
        """Get all variations of a nutrient name."""
        variations_map = {
            'calorie': ['calorie', 'calories'],
            'calories': ['calorie', 'calories'],
            'protein': ['protein', 'proteins'],
            'proteins': ['protein', 'proteins'],
            'carbohydrate': ['carbohydrate', 'carbohydrates', 'carb', 'carbs'],
            'carbohydrates': ['carbohydrate', 'carbohydrates', 'carb', 'carbs'],
            'carb': ['carbohydrate', 'carbohydrates', 'carb', 'carbs'],
            'carbs': ['carbohydrate', 'carbohydrates', 'carb', 'carbs'],
        }
        return variations_map.get(nutrient.lower(), [nutrient.lower()])

    def _extract_nutrient_from_text(self, text: str, nutrient: str) -> Tuple[float, float]:
        """Extract min and max values for a nutrient from text."""
        variations = self._get_nutrient_variations(nutrient)
        variations_pattern = '|'.join(variations)

        # Range patterns
        range_patterns = [
            rf"(?:{variations_pattern})\s+is\s+commonly\s+(\d+(?:\.\d+)?)[–\-—](\d+(?:\.\d+)?)\s*g",
            rf"(?:{variations_pattern})\s+often\s+falls?\s+around\s+(\d+(?:\.\d+)?)[–\-—](\d+(?:\.\d+)?)\s*(?:kcal|g)",
            rf"(?:{variations_pattern})\s+is\s+often\s+(\d+(?:\.\d+)?)[–\-—](\d+(?:\.\d+)?)\s*g",
        ]

        for pattern in range_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                return (float(match.group(1)), float(match.group(2)))

        # Single value patterns
        single_patterns = [
            rf"contains\s+(\d+(?:\.\d+)?)\s+(?:{variations_pattern})",
            rf"provides\s+(\d+(?:\.\d+)?)\s*g\s+of\s+(?:{variations_pattern})",
            rf"(\d+(?:\.\d+)?)\s*g\s+of\s+(?:{variations_pattern})",
            rf"(?:{variations_pattern})[:\s]+(\d+(?:\.\d+)?)",
        ]

        for pattern in single_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                value = float(match.group(1))
                return (value, value)

        return (0.0, 0.0)

    def _matches_filter(self, metadata: Dict, content: str, product_filter: Optional[str]) -> bool:
        """Check if product matches the filter."""
        if not product_filter:
            return True

        filter_lower = product_filter.lower()
        return any([
            filter_lower in metadata.get('brand', '').lower(),
            filter_lower in metadata.get('product', '').lower(),
            filter_lower in content[:300].lower()
        ])

    def _extract_structured_data(
        self,
        sources: List[Dict],
        nutrient: str,
        product_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract structured data from sources.
        Deduplicates by brand+product, keeping highest value.
        """
        products_dict = {}

        for source in sources:
            metadata = source.get('metadata', {})
            content = source.get('content', '')

            # Filter by product type
            if not self._matches_filter(metadata, content, product_filter):
                continue

            # Extract nutrient values
            min_val, max_val = self._extract_nutrient_from_text(content, nutrient.capitalize())

            if max_val <= 0:
                continue

            # Deduplicate by brand+product (without size)
            brand = metadata.get('brand', 'Unknown')
            product = metadata.get('product', 'Unknown')
            size = metadata.get('size', '')
            key = f"{brand}|{product}"

            product_data = {
                'brand': brand,
                'product': product,
                'size': size,
                'min': min_val,
                'max': max_val,
                'avg': (min_val + max_val) / 2
            }

            # Keep product with higher max value
            if key not in products_dict or max_val > products_dict[key]['max']:
                products_dict[key] = product_data
            elif max_val == products_dict[key]['max'] and size and not products_dict[key].get('size'):
                products_dict[key] = product_data

        # Sort by max value descending
        return sorted(products_dict.values(), key=lambda x: x['max'], reverse=True)

    def _prepare_labels(self, products: List[Dict]) -> List[str]:
        """Prepare product labels for charts."""
        labels = []
        for p in products:
            label = f"{p['brand']}\n{p['product']}"
            if p.get('size') and p['size'].strip():
                label += f"\n({p['size']})"
            labels.append(label)
        return labels

    def _get_nutrient_unit(self, nutrient: str) -> str:
        """Get unit for nutrient."""
        return "kcal" if nutrient.lower() == "calories" else "g"

    def _generate_bar_chart(
        self,
        products: List[Dict],
        nutrient: str,
        top_n: int,
        product_filter: Optional[str] = None
    ):
        """Generate bar chart and return figure."""
        if not products:
            return None

        labels = self._prepare_labels(products)
        max_values = [p['max'] for p in products]
        min_values = [p['min'] for p in products]

        fig, ax = plt.subplots(figsize=BAR_CHART_SIZE)

        x = range(len(products))
        bars = ax.bar(x, max_values, color='steelblue', alpha=0.8, label='Max')
        ax.bar(x, min_values, color='lightblue', alpha=0.6, label='Min')

        filter_label = f"{product_filter.capitalize()} " if product_filter else ""
        unit = self._get_nutrient_unit(nutrient)

        ax.set_xlabel('Products', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{nutrient.capitalize()} ({unit})', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} {filter_label}Products by {nutrient.capitalize()}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        logger.info("Bar chart generated")
        return fig

    def _generate_pie_chart(
        self,
        products: List[Dict],
        nutrient: str,
        top_n: int,
        product_filter: Optional[str] = None
    ):
        """Generate pie chart and return figure."""
        if not products:
            return None

        labels = [f"{p['brand']} {p['product']}" + (f" ({p['size']})" if p.get('size', '').strip() else '')
                  for p in products]
        values = [p['max'] for p in products]

        fig, ax = plt.subplots(figsize=PIE_CHART_SIZE)

        colors = plt.cm.Set3(range(len(products)))
        ax.pie(values, labels=None, autopct='%1.1f%%', startangle=90,
               colors=colors, textprops={'fontsize': 9})

        filter_label = f"{product_filter.capitalize()} " if product_filter else ""
        unit = self._get_nutrient_unit(nutrient)

        ax.set_title(f'Top {top_n} {filter_label}Products by {nutrient.capitalize()}',
                     fontsize=14, fontweight='bold', pad=20)

        legend_labels = [f"{label}: {val:.1f}{unit}" for label, val in zip(labels, values)]
        ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)

        plt.tight_layout()
        logger.info("Pie chart generated")
        return fig

    def _generate_line_chart(
        self,
        products: List[Dict],
        nutrient: str,
        top_n: int,
        product_filter: Optional[str] = None
    ):
        """Generate line chart and return figure."""
        if not products:
            return None

        labels = self._prepare_labels(products)
        max_values = [p['max'] for p in products]
        min_values = [p['min'] for p in products]
        avg_values = [p['avg'] for p in products]

        fig, ax = plt.subplots(figsize=LINE_CHART_SIZE)

        x = range(len(products))
        ax.plot(x, max_values, marker='o', linewidth=2, markersize=8,
                color='steelblue', label='Max', alpha=0.8)
        ax.plot(x, avg_values, marker='s', linewidth=2, markersize=6,
                color='green', label='Avg', alpha=0.6)
        ax.plot(x, min_values, marker='^', linewidth=2, markersize=6,
                color='lightblue', label='Min', alpha=0.6)

        ax.fill_between(x, min_values, max_values, alpha=0.2, color='steelblue')

        filter_label = f"{product_filter.capitalize()} " if product_filter else ""
        unit = self._get_nutrient_unit(nutrient)

        ax.set_xlabel('Products', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{nutrient.capitalize()} ({unit})', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} {filter_label}Products by {nutrient.capitalize()}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3)

        for i, val in enumerate(max_values):
            ax.text(i, val, f'{int(val)}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        logger.info("Line chart generated")
        return fig

    def query(
        self,
        question: str,
        top_k: int = 20,
        filter_metadata: Optional[Dict] = None,
        only_first_chunks: bool = True
    ) -> Dict[str, Any]:
        """
        Enhanced query that generates visualizations for ranking questions.
        """
        # Detect if this is a ranking query
        ranking_info = self._detect_ranking_query(question)

        # Get standard RAG response
        response = super().query(question, top_k, filter_metadata, only_first_chunks)

        if ranking_info:
            logger.info(f"Detected ranking query: {ranking_info['nutrient']} "
                       f"(top {ranking_info['top_n']}, filter: {ranking_info['filter'] or 'None'}, "
                       f"chart: {ranking_info['chart_type']})")

            products = self._extract_structured_data(
                response['sources'],
                ranking_info['nutrient'],
                ranking_info['filter']
            )

            if products:
                logger.debug(f"Extracted {len(products)} unique products")
                products = products[:ranking_info['top_n']]

                logger.info(f"Generating {ranking_info['chart_type']} chart")
                chart_type = ranking_info['chart_type']

                if chart_type == CHART_TYPE_PIE:
                    chart_fig = self._generate_pie_chart(
                        products, ranking_info['nutrient'],
                        ranking_info['top_n'], ranking_info['filter']
                    )
                elif chart_type == CHART_TYPE_LINE:
                    chart_fig = self._generate_line_chart(
                        products, ranking_info['nutrient'],
                        ranking_info['top_n'], ranking_info['filter']
                    )
                else:
                    chart_fig = self._generate_bar_chart(
                        products, ranking_info['nutrient'],
                        ranking_info['top_n'], ranking_info['filter']
                    )

                if chart_fig:
                    response['visualization'] = chart_fig
                    response['extracted_data'] = products
                    response['chart_type'] = chart_type
            else:
                logger.warning("No data extracted for visualization")

        return response

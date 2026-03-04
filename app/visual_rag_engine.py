"""
Visual RAG Engine - Combines chat with automatic visualization

Automatically generates visualizations for ranking queries.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .retriever import RAGQueryEngine
from .constants import (
    VALID_NUTRIENTS,
    VALID_PRODUCTS,
    CHART_DPI,
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

    def __init__(self, retriever, chat_client, visualization_dir: str = "visualizations"):
        super().__init__(retriever, chat_client)
        self.visualization_dir = Path(visualization_dir)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

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

    def _extract_nutrient_from_text(self, text: str, nutrient: str) -> Tuple[float, float]:
        """
        Extract min and max values for a nutrient from text.
        """
        # Normalize nutrient name (handle plural/singular)
        nutrient_lower = nutrient.lower()
        if nutrient_lower in ['calorie', 'calories']:
            nutrient_variations = ['calorie', 'calories']
        elif nutrient_lower in ['protein', 'proteins']:
            nutrient_variations = ['protein', 'proteins']
        elif nutrient_lower in ['carbohydrate', 'carbohydrates', 'carb', 'carbs']:
            nutrient_variations = ['carbohydrate', 'carbohydrates', 'carb', 'carbs']
        else:
            nutrient_variations = [nutrient_lower]

        # Try to match ranges first
        range_patterns = [
            rf"(?:{'|'.join(nutrient_variations)})\s+is\s+commonly\s+(\d+(?:\.\d+)?)[–\-—](\d+(?:\.\d+)?)\s*g",
            rf"(?:{'|'.join(nutrient_variations)})\s+often\s+falls?\s+around\s+(\d+(?:\.\d+)?)[–\-—](\d+(?:\.\d+)?)\s*(?:kcal|g)",
            rf"(?:{'|'.join(nutrient_variations)})\s+is\s+often\s+(\d+(?:\.\d+)?)[–\-—](\d+(?:\.\d+)?)\s*g",
        ]

        for pattern in range_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return (float(match.group(1)), float(match.group(2)))

        # Try single values - most common format: "contains 1882.1 calories per serving"
        single_value_patterns = [
            rf"contains\s+(\d+(?:\.\d+)?)\s+(?:{'|'.join(nutrient_variations)})",  # "contains 1882.1 calories"
            rf"provides\s+(\d+(?:\.\d+)?)\s*g\s+of\s+(?:{'|'.join(nutrient_variations)})",  # "provides 51.87g of protein"
            rf"(\d+(?:\.\d+)?)\s*g\s+of\s+(?:{'|'.join(nutrient_variations)})",  # "51.87g of protein"
            rf"(?:{'|'.join(nutrient_variations)})[:\s]+(\d+(?:\.\d+)?)",  # "Protein: 51.87" or "Calories 1882"
        ]

        for pattern in single_value_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return (value, value)

        return (0.0, 0.0)

    def _extract_structured_data(
        self,
        sources: List[Dict],
        nutrient: str,
        product_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract structured data from sources.
        Deduplicates by brand+product, keeping only the highest value.
        """
        products_dict = {}  # Use dict to deduplicate by brand+product

        for source in sources:
            metadata = source.get('metadata', {})
            brand = metadata.get('brand', 'Unknown')
            product = metadata.get('product', 'Unknown')
            size = metadata.get('size', '')
            content = source.get('content', '')

            # Create key WITHOUT size for deduplication
            # This ensures "Double Cheeseburger (Large)" and "Double Cheeseburger (Regular)"
            # are treated as the same product
            key = f"{brand}|{product}"

            # Filter by product type if specified
            # Check: brand, product name, or content (more lenient matching)
            if product_filter:
                filter_lower = product_filter.lower()
                brand_lower = brand.lower()
                product_lower = product.lower()
                content_lower = content.lower()

                # Pass if filter appears in brand, product name, or beginning of content
                match_found = (
                    filter_lower in brand_lower or
                    filter_lower in product_lower or
                    filter_lower in content_lower[:300]
                )

                if not match_found:
                    continue

            # Extract nutrient values
            min_val, max_val = self._extract_nutrient_from_text(content, nutrient.capitalize())

            # Validation: Log extraction for debugging
            if max_val > 0:
                # Successful extraction
                pass
            else:
                # Failed extraction - skip this source
                continue

            if max_val > 0:
                # If this product already exists, keep the one with higher max value
                # or if same max, prefer the one with larger size info
                if key in products_dict:
                    existing = products_dict[key]
                    # Keep the one with higher max value, or if equal, keep existing
                    if max_val > existing['max']:
                        products_dict[key] = {
                            'brand': brand,
                            'product': product,
                            'size': size,
                            'min': min_val,
                            'max': max_val,
                            'avg': (min_val + max_val) / 2
                        }
                    elif max_val == existing['max'] and size and not existing.get('size'):
                        # Same value but this one has size info, use it
                        products_dict[key] = {
                            'brand': brand,
                            'product': product,
                            'size': size,
                            'min': min_val,
                            'max': max_val,
                            'avg': (min_val + max_val) / 2
                        }
                else:
                    products_dict[key] = {
                        'brand': brand,
                        'product': product,
                        'size': size,
                        'min': min_val,
                        'max': max_val,
                        'avg': (min_val + max_val) / 2
                    }

        # Convert dict to list and sort by max value
        products = list(products_dict.values())
        products.sort(key=lambda x: x['max'], reverse=True)
        return products

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
    ) -> str:
        """Generate bar chart and return filename."""
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

        filename = f"top_{top_n}_{product_filter or 'products'}_{nutrient}_bar.png"
        filepath = self.visualization_dir / filename

        plt.tight_layout()
        plt.savefig(filepath, dpi=CHART_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"Bar chart saved: {filepath}")
        return str(filepath)

    def _generate_pie_chart(
        self,
        products: List[Dict],
        nutrient: str,
        top_n: int,
        product_filter: Optional[str] = None
    ) -> str:
        """Generate pie chart and return filename."""
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

        filename = f"top_{top_n}_{product_filter or 'products'}_{nutrient}_pie.png"
        filepath = self.visualization_dir / filename

        plt.tight_layout()
        plt.savefig(filepath, dpi=CHART_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"Pie chart saved: {filepath}")
        return str(filepath)

    def _generate_line_chart(
        self,
        products: List[Dict],
        nutrient: str,
        top_n: int,
        product_filter: Optional[str] = None
    ) -> str:
        """Generate line chart and return filename."""
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

        filename = f"top_{top_n}_{product_filter or 'products'}_{nutrient}_line.png"
        filepath = self.visualization_dir / filename

        plt.tight_layout()
        plt.savefig(filepath, dpi=CHART_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"Line chart saved: {filepath}")
        return str(filepath)

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
                    chart_file = self._generate_pie_chart(
                        products, ranking_info['nutrient'],
                        ranking_info['top_n'], ranking_info['filter']
                    )
                elif chart_type == CHART_TYPE_LINE:
                    chart_file = self._generate_line_chart(
                        products, ranking_info['nutrient'],
                        ranking_info['top_n'], ranking_info['filter']
                    )
                else:
                    chart_file = self._generate_bar_chart(
                        products, ranking_info['nutrient'],
                        ranking_info['top_n'], ranking_info['filter']
                    )

                if chart_file:
                    response['visualization'] = chart_file
                    response['extracted_data'] = products
                    response['chart_type'] = chart_type
            else:
                logger.warning("No data extracted for visualization")

        return response

"""
Application-wide constants.
"""

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 30
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ": ", " ", ""]

# Embedding Configuration
EMBEDDING_BATCH_SIZE = 100

# Vector Store Configuration
CHROMA_BATCH_SIZE = 5000
DEFAULT_COLLECTION_NAME = "nutrition_rag"
DEFAULT_PERSIST_DIR = "./chroma_db"

# Retrieval Configuration
DEFAULT_TOP_K = 5
DEFAULT_TOP_K_RANKING = 20

# Chat Configuration
DEFAULT_TEMPERATURE = 0.7

# Visualization Configuration
CHART_DPI = 300
BAR_CHART_SIZE = (14, 8)
PIE_CHART_SIZE = (12, 10)
LINE_CHART_SIZE = (14, 8)

# Nutrients
VALID_NUTRIENTS = ['protein', 'proteins', 'calories', 'calorie',
                   'fat', 'fats', 'carbohydrate', 'carbohydrates', 'carbs']

# Product Types
VALID_PRODUCTS = ['burger', 'burgers', 'pizza', 'pizzas', 'chicken',
                  'taco', 'tacos', 'sandwich', 'sandwiches', 'nugget',
                  'nuggets', 'fries', 'salad', 'salads', 'product',
                  'products', 'food', 'foods', 'item', 'items']

# Brand Mappings
BRAND_MAPPINGS = {
    'mcdonalds': "McDonald's",
    "mcdonald's": "McDonald's",
    'mcdonald': "McDonald's",
    'burger king': 'Burger King',
    'burgerking': 'Burger King',
    'pizza hut': 'Pizza Hut',
    'pizzahut': 'Pizza Hut',
    'kfc': 'KFC',
    'wendys': "Wendy's",
    "wendy's": "Wendy's",
    'subway': 'Subway',
    'dominos': "Domino's",
    "domino's": "Domino's",
    'taco bell': 'Taco Bell',
    'tacobell': 'Taco Bell',
}

# Chart Types
CHART_TYPE_BAR = 'bar'
CHART_TYPE_PIE = 'pie'
CHART_TYPE_LINE = 'line'

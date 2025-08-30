from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Flask Configuration
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5001))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True") == "True"

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "openrouter/auto")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

# Vector Store Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
SIMILARITY_SEARCH_K = int(os.getenv("SIMILARITY_SEARCH_K", 3))

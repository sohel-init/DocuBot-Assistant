import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Tokens / keys
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not HF_TOKEN:
        print("Warning: HUGGINGFACEHUB_API_TOKEN is missing in .env")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY is missing in .env")
    
    # Model Configuration
    REPO_ID = "google/flan-t5-large"
    HF_ENDPOINT_URL="https://router.huggingface.co/hf-inference/models"
    MODEL_NAME = "llama-3.1-8b-instant"
    GROQ_WHISPER_MODEL_NAME = "whisper-large-v3"
    TEMPERATURE = 0.5
    MAX_NEW_TOKENS = 512
    RETURN_FULL_TEXT = False
    MODEL_KWARGS = {
        "max_length": 512
    }
    
    # Chunking Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
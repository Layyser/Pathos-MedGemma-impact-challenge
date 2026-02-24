""" Configuration settings for the Pathos application """
from dotenv import load_dotenv
import os
load_dotenv()

# Your config
EMAIL = "" or os.getenv("EMAIL")


# Paths
DATASET_REGISTRY = "datasets.json"
DATA_DIR = "data"
RAW_DATA_DIR = "raw_data"
PERSIST_DIR = "db"
ADAPTERS_DIR = "adapters"
PROFILES_DIR = "profiles"

# VectorDB Settings
VECTOR_DB_TYPE = "chroma"  # Options: "chroma", ("pinecone") <- for future expansion
VECTOR_DB_UPSERT_BATCH_SIZE = 100

# Local model Settings
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
MEDGEMMA_PROVIDER_TYPE = "local"  # Options: "local", ("hospital_api") <- for future expansion

# Embedding Settings
EMBEDDING_PROVIDER_TYPE = "local"  # Options: "local", ("hospital_api") <- for future expansion
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Cloud LLM settings (WE USE FREE TIER) we don't need almost any query. 
# This query will never contain sensitive info
CLOUD_PROVIDER_TYPE = "google"
GOOGLE_API_KEY = "" or os.getenv("GOOGLE_API_KEY") # Insert your free google API Key here
GOOGLE_MODEL_PRIORITY_LIST = [
    "gemini-2.5-flash",       # Stable
    "gemini-3-flash-preview",
    "gemini-1.5-flash",       
]

# RAG Settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# ChromaDB Settings
CHROMA_COLLECTION_NAME = "medical_documents"
CHROMA_TRUTH_COLLECTION_NAME = "profile_ground_truth"

# Timeline Logic Settings
MIN_LONG_SUMMARY_LENGTH = 3000
MIN_SHORT_SUMMARY_LENGTH = 1000
DATE_PREFERRED_ORDER = ['MDY', 'DMY', 'YMD'] # Supported date formats: 'YMD', 'DMY', 'MDY'

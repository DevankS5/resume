# vertex.py
import os
import json
import logging
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import vertexai
from typing import Optional

# ------------------------------
# Load Environment Variables
# ------------------------------
load_dotenv()
logger = logging.getLogger("uvicorn")

# ------------------------------
# 1. Load Configuration
# ------------------------------
PROJECT_ID: Optional[str] = os.getenv("PROJECT_ID")
LOCATION: str = os.getenv("LOCATION", "asia-south1")
ENDPOINT_ID: Optional[str] = os.getenv("ENDPOINT_ID")

# This var should contain the *full JSON content* as a string
GCP_SERVICE_ACCOUNT_JSON: Optional[str] = os.getenv("GCP_SERVICE_ACCOUNT")
# This var should be the *path* to the .json file
SERVICE_ACCOUNT_FILE: Optional[str] = os.getenv("SERVICE_ACCOUNT_FILE")


# ------------------------------
# 2. Helper to Get Credentials
# ------------------------------
def get_credentials() -> Optional[service_account.Credentials]:
    """
    Loads credentials from env var (JSON string) or file path.
    """
    if GCP_SERVICE_ACCOUNT_JSON:
        try:
            creds_info = json.loads(GCP_SERVICE_ACCOUNT_JSON)
            creds = service_account.Credentials.from_service_account_info(creds_info)
            logger.info("Loaded credentials from GCP_SERVICE_ACCOUNT env var.")
            return creds
        except Exception as e:
            logger.error(f"Failed to load creds from env var: {e}")
            return None
    
    if SERVICE_ACCOUNT_FILE:
        try:
            creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
            logger.info(f"Loaded credentials from file: {SERVICE_ACCOUNT_FILE}")
            return creds
        except Exception as e:
            logger.error(f"Failed to load creds from file: {e}")
            return None
            
    logger.warning("No service account provided. Using Application Default Credentials (ADC).")
    return None

# ------------------------------
# 3. Initialize All Services
# ------------------------------

# Create a global variable for credentials to be used by other files
credentials = get_credentials()

try:
    # --- Initialize Vertex AI SDK ---
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    logger.info("Vertex AI SDK initialized successfully.")

    # --- Initialize Generative Model (Gemini) ---
    generative_model = GenerativeModel("gemini-1.5-flash-001")
    logger.info("GenerativeModel (Gemini) initialized.")

    # --- Initialize Embedding Model ---
    embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    logger.info("TextEmbeddingModel initialized.")
    
    # --- Initialize Vector Search Endpoint ---
    vector_search_endpoint = None
    if ENDPOINT_ID:
        vector_search_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=ENDPOINT_ID,
            credentials=credentials
        )
        logger.info(f"Vector Search endpoint loaded: {ENDPOINT_ID}")
    
except Exception as e:
    logger.error(f"CRITICAL: Failed to initialize Vertex AI services: {e}", exc_info=True)
    # Stop the app from starting if Vertex is critical
    raise e
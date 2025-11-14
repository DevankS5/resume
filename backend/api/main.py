# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from typing import Optional

# --- Import your initialized Vertex AI clients ---
from vertex import (
    credentials,  # Import the credentials object
    generative_model,
    embedding_model,
    vector_search_endpoint,
    PROJECT_ID,
    LOCATION
)

# --- Initialize non-Vertex AI services here ---
from google.cloud import storage
from sentence_transformers.cross_encoder import CrossEncoder

logger = logging.getLogger("uvicorn")

# Init GCS Client using the same credentials
try:
    storage_client = storage.Client(project=PROJECT_ID, credentials=credentials)
    logger.info("Google Cloud Storage client initialized.")
except Exception as e:
    logger.error(f"Failed to initialize GCS client: {e}")
    storage_client = None # App can run but GCS features will fail

# Init Cross-Encoder (this is local, no creds needed)
try:
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-minilm-l-6-v2')
    logger.info("Cross-encoder model loaded.")
except Exception as e:
    logger.error(f"Failed to load cross-encoder: {e}")
    cross_encoder = None # App can run but reranking will fail


# --- FastAPI App Setup ---
app = FastAPI(title="RezumAI-backend", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    source: str

# --- Endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message:
        raise HTTPException(status_code=400, detail="message is required")

    try:
        # Now you can use the imported clients directly
        # Example:
        # 1. Get embeddings:
        #    query_embedding = embedding_model.get_embeddings([req.message])[0].values
        #
        # 2. Search vector store:
        #    if vector_search_endpoint:
        #        search_results = vector_search_endpoint.find_neighbors(...)
        #
        # 3. Retrieve from GCS:
        #    if storage_client:
        #        blob = storage_client.bucket(WIKI_BUCKET_NAME).blob(...)
        #
        # 4. Rerank:
        #    if cross_encoder:
        #        scores = cross_encoder.predict(...)
        #
        # 5. Call Gemini
        response = generative_model.generate_content(
            req.message,
        )
        
        reply = response.text
        return ChatResponse(reply=reply, source="gemini-1.5-flash")

    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")

# (Your Agora token endpoint can be added back here if needed)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from typing import Optional

app = FastAPI(title="RezumAI-backend", version="0.1")

# CORS - allow frontend dev; adjust in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn")

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    source: str


def call_vertex_ai(prompt: str) -> str:
    """Attempt to call Vertex AI (if library available). Falls back to a mock reply.

    Replace environment variables and adjust model usage to your setup.
    """
    project = os.environ.get("VERTEX_PROJECT")
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    model = os.environ.get("VERTEX_MODEL_ID")  # e.g. "text-bison@001" or your model resource

    try:
        from google.cloud import aiplatform
        # NOTE: This example uses the low-level PredictionServiceClient via aiplatform.gapic
        client = aiplatform.gapic.PredictionServiceClient()
        name = f"projects/{project}/locations/{location}/publishers/google/models/{model}"

        instance = {"content": prompt}
        request = {
            "endpoint": name,
            "instances": [instance],
            # "parameters": {...},
        }
        # The exact request/response shape depends on model and SDK versions.
        response = client.predict(request=request)
        # Try to extract a sensible text reply from response
        if response.predictions:
            # This will depend on the model. We conservatively join text fields.
            parts = []
            for p in response.predictions:
                if isinstance(p, dict):
                    # look for 'content' or 'text'
                    for k in ("content", "text", "output"):
                        if k in p:
                            parts.append(str(p[k]))
                else:
                    parts.append(str(p))
            return "\n".join(parts) if parts else str(response)
        return str(response)
    except Exception as e:
        # If Vertex SDK or credentials are not available, return a helpful mock reply
        logger.warning("Vertex AI call failed or not configured: %s", e)
        return (
            "[vertex-mock] I would call Vertex AI here if configured.\n"
            "Set VERTEX_PROJECT, VERTEX_LOCATION, and VERTEX_MODEL_ID environment variables and install google-cloud-aiplatform."
        )


def generate_agora_token(uid: Optional[str] = None) -> str:
    """Placeholder for generating an Agora token.

    Replace with Agora's official token builder/server-side SDK to create RTC/RTM tokens.
    Example env vars: AGORA_APP_ID, AGORA_APP_CERTIFICATE
    """
    app_id = os.environ.get("AGORA_APP_ID")
    app_cert = os.environ.get("AGORA_APP_CERTIFICATE")
    if not app_id or not app_cert:
        logger.warning("Agora credentials not set; returning mock token")
        return "MOCK_AGORA_TOKEN"

    # TODO: Use Agora's official token builder from their Python SDK.
    # For now return a placeholder string to show where to plug in token logic.
    return f"AGORA_TOKEN_FOR_{uid or 'anon'}"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message:
        raise HTTPException(status_code=400, detail="message is required")

    # 1) (Optional) Generate Agora token if needed (demo)
    agora_token = generate_agora_token()

    # 2) Call Vertex AI (or mock)
    reply = call_vertex_ai(req.message)

    # 3) Return structured response
    return ChatResponse(reply=reply, source="vertex-ai + agora-placeholder")


@app.get("/agora/token")
async def agora_token_endpoint(uid: Optional[str] = None):
    """Simple endpoint to retrieve a server-generated Agora token (placeholder).

    Replace logic with Agora SDK token generator.
    """
    token = generate_agora_token(uid)
    return {"token": token}

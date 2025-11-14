# RezumAI Backend (FastAPI)

This backend is a minimal FastAPI app with placeholder integrations for:

- Vertex AI (Text generation) via `google-cloud-aiplatform` (optional)
- Agora conversational / token endpoint (placeholder)

Files:
- `main.py` - FastAPI app with `/chat` and `/agora/token` endpoints.
- `.env.example` - example environment variables.
- `requirements.txt` - Python dependencies for development.

Quick start (local, Python):
1. Create a virtualenv and install deps:

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

2. Copy `.env.example` to `.env` and fill in credentials for Vertex/Agora.

3. Run the app:

   .\.venv\Scripts\Activate.ps1; uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Notes:
- The Vertex AI call in `main.py` is a conservative, best-effort example. Adjust the request/response handling to match the exact model you use and the SDK version.
- For Agora tokens, replace the placeholder `generate_agora_token` with the official token builder from Agora.
- When running in Docker, pass environment variables via an env-file or secrets mechanism.

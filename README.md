# RezumAI (Monorepo)

This repository contains a minimal starting scaffold for the RezumAI project:

- `backend/` — FastAPI app with placeholder Vertex AI + Agora integration
- `frontend/` — Next.js app with a simple chat form
- `docker-compose.yml` — brings up both services for local development

Quick run using Docker Compose (PowerShell):

```powershell
# Build and run services
docker-compose up --build
```

Open the frontend at http://localhost:3000 and the backend health at http://localhost:8000/health

Environment variables
- Copy `backend/.env.example` to `backend/.env` and fill values for:
  - VERTEX_PROJECT
  - VERTEX_LOCATION
  - VERTEX_MODEL_ID
  - AGORA_APP_ID
  - AGORA_APP_CERTIFICATE

Notes and next steps
- Vertex AI: replace the conservative example in `backend/main.py` with the exact call signature for your model and SDK version. Configure GCP credentials (ADC) in your environment or container.
- Agora: replace `generate_agora_token` with Agora's official token builder implementation. Add the required server-side SDK as a dependency.
- Frontend: currently posts to `http://localhost:8000/chat`. If you run behind a proxy or in Docker networking, update the URL to the backend service name or use a relative path with a reverse proxy.

If you'd like, I can now:
- Add a proper Dockerfile for `frontend` and `backend` if you want them to be self-contained (I noticed `Dockerfile`s already exist in each folder).
- Replace the Vertex AI placeholder with a tested code snippet matching a specific Vertex model (tell me which model/resource you plan to use).
- Implement real Agora token generation with the correct Python package (if you provide which Agora product and token type you need).

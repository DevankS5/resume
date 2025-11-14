# RezumAI Frontend (Next.js)

This is a minimal Next.js frontend that sends chat messages to the backend `/chat` endpoint.

Scripts:
- `npm run dev` — starts Next.js on port 3000
- `npm run build` / `npm run start` — build and start for production

By default the frontend's sample form POSTs to `http://localhost:8000/chat`. When running with Docker Compose the backend service name is `backend` and you can update the fetch URL accordingly.

To run locally:
1. Install dependencies in `frontend`:

   npm install

2. Start dev server:

   npm run dev


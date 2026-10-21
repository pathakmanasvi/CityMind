# CityMind (Phase 1: reliable core)

Multi-agent city decision-support. Phase 1 = chat + document upload + 5-step LangGraph pipeline
(Router → Retriever → Checker → Fallback → Report).

## 5-step setup
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and add `GOOGLE_API_KEY` (required) and `PINECONE_API_KEY` (for documents)
4. `uvicorn app.main:app --reload`
5. Open http://localhost:8000, go to **Docs**, upload a PDF, then ask a question.

## Modes
- Only `GOOGLE_API_KEY`: general-knowledge + web (Gemini Google Search grounding), labelled honestly.
- Both keys: answers from your documents, falling back to web when the documents are insufficient.
- No keys: app boots, shows "No API key" and friendly errors.

## Deploy
- **Render (recommended):** connect repo, `render.yaml` is picked up. No short request timeout.
- **Vercel:** works for light use, but serverless functions have execution-time and request-size limits
  (Hobby plan: ~10 s by default, ~4.5 MB body). Large uploads and slow LLM calls can fail there.
  In-memory rate limiting also resets per instance.

## Tests
`tests/fixtures/` has a real text PDF and an empty (scanned-like) PDF for upload checks.

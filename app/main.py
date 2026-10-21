import time, collections
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from . import clients, config as C
from .chunking import make_records, slug

app = FastAPI(title="CityMind")
HITS = collections.defaultdict(list)

@app.middleware("http")
async def limiter(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "x")).split(",")[0].strip()
        now = time.time(); HITS[ip] = [t for t in HITS[ip] if now - t < 60]
        if len(HITS[ip]) >= C.RATE_LIMIT:
            return JSONResponse({"error": "Too many requests. Slow down for a minute."}, status_code=429)
        HITS[ip].append(now)
    return await call_next(request)

@app.exception_handler(Exception)
async def on_err(request, exc):
    fe = clients.friendly(exc)
    return JSONResponse({"error": fe.msg}, status_code=fe.status)

class Msg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$"); content: str = Field(max_length=4000)
class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=C.MAX_QUESTION)
    city: str = Field(default="General", max_length=60)
    history: Optional[List[Msg]] = Field(default=None, max_length=12)
class Ingest(BaseModel):
    city: str = Field(min_length=1, max_length=60)
    source: str = Field(min_length=1, max_length=200)
    part: int = Field(ge=0, le=5000)
    text: str = Field(min_length=1, max_length=C.MAX_PART_CHARS)

class Cmp(BaseModel):
    cities: List[str] = Field(min_length=2, max_length=3)
    topic: str = Field(min_length=1, max_length=300)
class Road(BaseModel):
    city: str = Field(min_length=1, max_length=60)
    goal: str = Field(min_length=1, max_length=500)
    years: int = Field(default=5, ge=1, le=20)

@app.post("/api/compare")
def compare(b: Cmp):
    from .features import compare as f
    return f([c.strip()[:60] for c in b.cities if c.strip()], b.topic)

@app.post("/api/roadmap")
def roadmap(b: Road):
    from .features import roadmap as f
    return f(b.city, b.goal, b.years)

@app.get("/api/status")
def status():
    info = {"google": clients.has_google(), "pinecone": clients.has_pinecone(), "uploads": C.ALLOW_UPLOADS,
            "mode": "documents" if clients.has_google() and clients.has_pinecone() else "general" if clients.has_google() else "offline",
            "cities": {}}
    if clients.has_pinecone():
        try:
            st = clients.pinecone_index().describe_index_stats()
            info["cities"] = {k: v.vector_count for k, v in (st.namespaces or {}).items()}
        except Exception as e: info["pinecone_error"] = clients.friendly(e).msg
    return info

@app.post("/api/ask")
def ask(b: Ask):
    from .graph import run
    s = run(b.question, b.city, [m.model_dump() for m in b.history or []])
    return {"answer": s["answer"], "label": s["label"], "trace": s["trace"],
            "sources": [{"n": i+1, **p} for i, p in enumerate(s.get("passages", []))] ,
            "web_sources": s.get("web_sources", [])}

@app.post("/api/ingest")
def ingest(b: Ingest):
    if not C.ALLOW_UPLOADS: return JSONResponse({"error": "Uploads are disabled on this server."}, status_code=403)
    recs = make_records(b.city, b.source, b.part, b.text)
    if not recs: return JSONResponse({"error": "No usable text found in this part."}, status_code=422)
    vecs = clients.embed([r["text"] for r in recs])
    idx = clients.pinecone_index()
    items = [{"id": r["id"], "values": v, "metadata": {"text": r["text"][:3500], "source": r["source"], "city": b.city}} for r, v in zip(recs, vecs)]
    for i in range(0, len(items), 100):
        idx.upsert(vectors=items[i:i+100], namespace=slug(b.city))
    return {"chunks": len(recs)}

@app.delete("/api/city/{city}")
def delete_city(city: str):
    if not C.ALLOW_UPLOADS: return JSONResponse({"error": "Disabled."}, status_code=403)
    clients.pinecone_index().delete(delete_all=True, namespace=slug(city))
    return {"deleted": slug(city)}

STATIC = Path(__file__).resolve().parent.parent / "static"
@app.get("/")
def home(): return FileResponse(STATIC / "index.html")

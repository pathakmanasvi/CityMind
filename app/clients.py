"""Lazy clients: importing this module never needs keys."""
import time
from . import config as C

_genai = _pc = _index = None

class FriendlyError(Exception):
    def __init__(self, msg, status=500):
        super().__init__(msg); self.msg, self.status = msg, status

def friendly(e: Exception) -> FriendlyError:
    if isinstance(e, FriendlyError): return e
    s = str(e).lower()
    if "429" in s or "quota" in s or "rate" in s and "limit" in s:
        return FriendlyError("The AI service is rate-limited right now. Wait a few seconds and try again.", 429)
    if "404" in s or "not found" in s or "not_found" in s:
        return FriendlyError(f"Model not found or not available to your key. Set CHAT_MODEL / EMBED_MODEL to a current model (e.g. gemini-3.8-flash). Details: {str(e)[:150]}", 404)
    if "dimension" in s:
        return FriendlyError(f"Pinecone index dimension mismatch: the index must have dimension {C.EMBED_DIMS}. Delete the old index or change PINECONE_INDEX.", 400)
    if "api key" in s or "api_key" in s or "401" in s or "403" in s or "permission" in s:
        return FriendlyError("An API key looks invalid or missing permissions. Check GOOGLE_API_KEY / PINECONE_API_KEY.", 401)
    if "timeout" in s or "timed out" in s or "deadline" in s:
        return FriendlyError("The request timed out. Try again, or ask a shorter question.", 504)
    return FriendlyError(f"Unexpected error: {str(e)[:200]}", 500)

def retry(fn, tries=3):
    """Retry on 429/rate-limit errors with backoff (handles per-minute limits)."""
    for a in range(tries):
        try: return fn()
        except Exception as e:
            m = str(e).lower()
            if a == tries - 1 or not ("429" in m or "resource_exhausted" in m or "quota" in m): raise
            time.sleep(6 * (a + 1))

def has_google(): return bool(C.env("GOOGLE_API_KEY"))
def has_pinecone(): return bool(C.env("PINECONE_API_KEY"))

def genai_client():
    global _genai
    if not has_google(): raise FriendlyError("GOOGLE_API_KEY is not set.", 400)
    if _genai is None:
        from google import genai
        _genai = genai.Client(api_key=C.env("GOOGLE_API_KEY"))
    return _genai

def pinecone_index():
    global _pc, _index
    if not has_pinecone(): raise FriendlyError("PINECONE_API_KEY is not set (documents disabled).", 400)
    if _index is None:
        from pinecone import Pinecone, ServerlessSpec
        _pc = Pinecone(api_key=C.env("PINECONE_API_KEY"))
        names = _pc.list_indexes().names()
        if C.INDEX_NAME not in names:
            _pc.create_index(name=C.INDEX_NAME, dimension=C.EMBED_DIMS, metric="cosine",
                spec=ServerlessSpec(cloud=C.env("PINECONE_CLOUD", "aws"), region=C.env("PINECONE_REGION", "us-east-1")))
            for _ in range(30):
                if _pc.describe_index(C.INDEX_NAME).status["ready"]: break
                time.sleep(2)
        _index = _pc.Index(C.INDEX_NAME)
    return _index

def embed(texts, query=False):
    from google.genai import types
    out = []
    for i in range(0, len(texts), 50):
        r = retry(lambda: genai_client().models.embed_content(model=C.EMBED_MODEL, contents=texts[i:i+50],
            config=types.EmbedContentConfig(output_dimensionality=C.EMBED_DIMS,
                task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT")))
        out += [e.values for e in r.embeddings]
    return out

def generate(prompt, system=None, web=False, temperature=0.2):
    from google.genai import types
    cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
        tools=[types.Tool(google_search=types.GoogleSearch())] if web else None)
    r = retry(lambda: genai_client().models.generate_content(model=C.CHAT_MODEL, contents=prompt, config=cfg))
    urls = []
    if web:
        try:
            for ch in r.candidates[0].grounding_metadata.grounding_chunks or []:
                urls.append({"title": ch.web.title, "url": ch.web.uri})
        except Exception: pass
    try: text = (r.text or "").strip()
    except Exception: text = ""
    if not text: raise FriendlyError("The model returned an empty answer (possibly blocked). Try rephrasing.", 502)
    return text, urls

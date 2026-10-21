import json, re
from typing import TypedDict, List, Dict, Any
from . import clients, config as C
from .chunking import slug

class S(TypedDict, total=False):
    question: str; city: str; history: List[Dict[str, str]]
    kind: str; passages: List[Dict[str, Any]]; sufficient: bool
    web_notes: str; web_sources: List[Dict[str, str]]
    answer: str; label: str; trace: List[Dict[str, str]]

GREET = re.compile(r"^\s*(hi|hello|hey|namaste|good (morning|afternoon|evening)|thanks|thank you)\b[\s!.?]*$", re.I)

def _t(s, step, status, note=""):
    s.setdefault("trace", []).append({"step": step, "status": status, "note": note})

def router(s: S):
    s["kind"] = "greeting" if GREET.match(s["question"]) else "question"
    _t(s, "Router", "done", s["kind"]); return s

def retriever(s: S):
    s["passages"] = []
    if not (clients.has_pinecone() and clients.has_google()):
        _t(s, "Retriever", "skipped", "no document store configured"); return s
    try:
        q = s["question"]
        if s.get("history"):  # light follow-up resolution
            last = [h["content"] for h in s["history"] if h["role"] == "user"][-1:]
            if len(q.split()) < 8 and last:
                q = f"{last[0]} {q}"
        vec = clients.embed([q], query=True)[0]
        r = clients.pinecone_index().query(vector=vec, top_k=6, include_metadata=True, namespace=slug(s["city"]))
        s["passages"] = [{"text": m.metadata.get("text", ""), "source": m.metadata.get("source", "?"), "score": round(m.score, 3)} for m in r.matches]
        _t(s, "Retriever", "done", f"{len(s['passages'])} passages")
    except Exception as e:
        _t(s, "Retriever", "error", clients.friendly(e).msg)
    return s

def checker(s: S):
    if not s["passages"]:
        s["sufficient"] = False; _t(s, "Checker", "done", "no passages found"); return s
    if C.env("LLM_CHECK", "false").lower() != "true":
        s["sufficient"] = s["passages"][0]["score"] > 0.55; _t(s, "Checker", "done", "similarity check"); return s
    ctx = "\n\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(s["passages"]))
    try:
        raw, _ = clients.generate(f"Question: {s['question']}\n\nPassages:\n{ctx}\n\nDo the passages contain enough to answer the question? "
                                  'Reply ONLY JSON: {"sufficient": true|false, "reason": "short"}')
        j = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        s["sufficient"] = bool(j.get("sufficient")); _t(s, "Checker", "done", j.get("reason", "")[:120])
    except Exception:
        s["sufficient"] = s["passages"][0]["score"] > 0.6; _t(s, "Checker", "done", "fallback heuristic on similarity")
    return s

def fallback(s: S):
    try:
        s["web_notes"], s["web_sources"] = clients.generate(
            f"Answer briefly and factually, only with what you can verify. City context: {s['city']}.\nQuestion: {s['question']}",
            system=C.HONESTY, web=True)
        _t(s, "Fallback", "done", f"web search, {len(s['web_sources'])} sources")
    except Exception as e:
        s["web_notes"], s["web_sources"] = "", []
        _t(s, "Fallback", "error", clients.friendly(e).msg)
    return s

def report(s: S):
    hist = "\n".join(f"{h['role']}: {h['content'][:400]}" for h in (s.get("history") or [])[-6:])
    if s["kind"] == "greeting":
        s["answer"], _ = clients.generate(s["question"], system="You are CityMind, a friendly urban-planning assistant. Reply in one or two warm sentences and invite a question.")
        s["label"] = "General knowledge"; s["passages"] = []
        _t(s, "Report", "done", "greeting"); return s
    passages = s["passages"] if s.get("sufficient") else (s["passages"] if s.get("web_notes") else [])
    ctx = "\n\n".join(f"[{i+1}] ({p['source']}) {p['text']}" for i, p in enumerate(passages)) or "(none)"
    web = f"\n\nWeb notes (not binding, say so):\n{s['web_notes']}" if s.get("web_notes") else ""
    if passages and s.get("sufficient"): s["label"] = "From documents"
    elif passages and s.get("web_notes"): s["label"] = "Partly from documents"
    elif s.get("web_notes") and s.get("web_sources"): s["label"] = "Web"
    else: s["label"] = "General knowledge"
    s["answer"], _ = clients.generate(
        f"Conversation so far:\n{hist}\n\nCity: {s['city']}\nQuestion: {s['question']}\n\nSources:\n{ctx}{web}\n\n"
        f"Write a clear answer in Markdown. The answer is labelled '{s['label']}'; make it consistent with that label.",
        system="You are CityMind, a careful urban-planning decision-support assistant.\n" + C.HONESTY, temperature=0.3)
    s["passages"] = passages
    _t(s, "Report", "done", s["label"]); return s

def run(question, city, history):
    """Router -> Retriever -> Checker -> (Fallback) -> Report, as plain Python (no langgraph)."""
    s: S = {"question": question, "city": city, "history": history or [], "trace": []}
    router(s)
    if s["kind"] != "greeting":
        retriever(s); checker(s)
        if not s["sufficient"]: fallback(s)
    return report(s)

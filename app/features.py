"""Compare + Roadmap features. One LLM call each."""
import json, re
from . import clients
from .chunking import slug

def passages(city, q, k=5):
    if not (clients.has_pinecone() and clients.has_google()):
        return []
    try:
        vec = clients.embed([q], query=True)[0]
        r = clients.pinecone_index().query(vector=vec, top_k=k, include_metadata=True, namespace=slug(city))
        return [m.metadata.get("text", "") for m in r.matches if m.score > 0.45]
    except Exception:
        return []

def _label(counts):
    n = sum(1 for c in counts.values() if c)
    return "From documents" if n == len(counts) else "Partly from documents" if n else "General knowledge"

def _ctx(city, ps):
    body = "\n".join(f"- {t[:700]}" for t in ps) or "(no documents uploaded for this city - use general knowledge and mark it '(general knowledge)')"
    return f"### Sources for {city}\n{body}"

def compare(cities, topic):
    ps = {c: passages(c, topic) for c in cities}
    counts = {c: len(p) for c, p in ps.items()}
    ctx = "\n\n".join(_ctx(c, p) for c, p in ps.items())
    prompt = (f"Compare these cities on: {topic}\n\n{ctx}\n\n"
              "Write Markdown: one short intro sentence, then ONE table (rows = aspects, columns = cities), then 2-3 bullet takeaways. "
              "Use only the sources above; write 'not specified in the sources' where a city has no info. Do not use [1]-style citations.")
    from . import config as C
    ans, _ = clients.generate(prompt, system="You are CityMind, a careful urban-planning analyst.\n" + C.HONESTY, temperature=0.2)
    return {"answer": ans, "label": _label(counts), "coverage": counts}

def roadmap(city, goal, years):
    ps = passages(city, goal)
    counts = {city: len(ps)}
    prompt = (f"City: {city}\nGoal: {goal}\nHorizon: {years} years\n\n{_ctx(city, ps)}\n\n"
              'Create a phased implementation roadmap. Reply ONLY with JSON: {"summary":"1-2 sentences","phases":[{"name":"...","timeframe":"e.g. Year 0-1",'
              '"actions":["..."],"risks":["..."],"kpis":["..."]}]} with 3-5 phases, 2-4 items per list. '
              "Do not invent figures, laws or clause numbers; use only the sources or clearly general practice.")
    from . import config as C
    raw, _ = clients.generate(prompt, system="You are CityMind, a careful urban-planning analyst.\n" + C.HONESTY, temperature=0.3)
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        assert isinstance(data.get("phases"), list) and data["phases"]
    except Exception:
        raise clients.FriendlyError("The model returned an unreadable roadmap. Please try again.", 502)
    data["label"] = _label(counts)
    return data

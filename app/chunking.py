import hashlib, re

def slug(city: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", city.lower()).strip("-")
    return s[:60] or "default"

def chunk_text(text, size=900, overlap=120):
    text = re.sub(r"[ \t]+", " ", text.replace("\r", ""))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    out, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):  # prefer to break at a sentence/paragraph end
            window = text[i:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > size * 0.5:
                end = i + cut + 1
        piece = text[i:end].strip()
        if len(piece) > 20:
            out.append(piece)
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    return out

def make_records(city, source, part, text):
    """Deterministic IDs: same text re-uploaded -> same IDs -> upsert overwrites, no duplicates."""
    recs = []
    for idx, c in enumerate(chunk_text(text, 900, 120)):
        body = f"[{city}] {c}"
        h = hashlib.sha1(f"{slug(city)}|{source}|{part}|{idx}|{c}".encode()).hexdigest()[:32]
        recs.append({"id": h, "text": body, "source": source, "part": part})
    return recs

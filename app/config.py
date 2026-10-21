import os
from dotenv import load_dotenv
load_dotenv()

def env(name, default=""):
    return os.getenv(name, default).strip()

CHAT_MODEL = env("CHAT_MODEL", "gemini-3.8-flash")
EMBED_MODEL = env("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIMS = 768
INDEX_NAME = env("PINECONE_INDEX", "citymind")
ALLOW_UPLOADS = env("ALLOW_UPLOADS", "true").lower() != "false"
RATE_LIMIT = int(env("RATE_LIMIT_PER_MIN", "30") or 30)
MAX_QUESTION = 2000
MAX_PART_CHARS = 40_000
CHUNK_SIZE, CHUNK_OVERLAP = 900, 120

HONESTY = """RULES (never break):
- Never invent figures, clause numbers, fees, dates or laws. If something is not in the sources, say "not specified in the sources".
- Separate binding local rules from general frameworks and best practice.
- Cite passages as [1], [2] matching the numbered sources you were given. Do not cite anything else.
- End with one line reminding the user to verify important decisions with the local authority."""

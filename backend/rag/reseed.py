"""
Limpia los seeds viejos de ChromaDB y los reemplaza con los actuales de knowledge_base.py.
Uso: python -m backend.rag.reseed
"""
from pathlib import Path
import chromadb

_CHROMA_PATH = Path(__file__).parent / "chroma_db"

client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
col = client.get_collection("knowledge")

seeds = col.get(where={"source": "seed"})
if seeds["ids"]:
    col.delete(ids=seeds["ids"])
    print(f"Eliminados {len(seeds['ids'])} seeds viejos.")
else:
    print("No habia seeds para eliminar.")

print(f"Documentos restantes: {col.count()}")

from backend.rag.knowledge_base import _seed_if_missing
_seed_if_missing(col)
print(f"Seeds nuevos cargados. Total documentos: {col.count()}")

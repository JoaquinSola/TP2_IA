"""
Utilidad de inspección: muestra todos los documentos indexados en ChromaDB.
Uso: python -m backend.rag.inspect_db
"""
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

_CHROMA_PATH = Path(__file__).parent / "chroma_db"
_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
ef = SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)
col = client.get_collection(name="knowledge", embedding_function=ef)

print(f"Total documentos en la coleccion: {col.count()}\n")

all_docs = col.get(include=["documents", "metadatas"])
for i, (doc, meta) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
    source = meta.get("source", "?")
    doc_type = meta.get("type", "?")
    preview = doc[:150] + "..." if len(doc) > 150 else doc
    print(f"[{i+1:02d}] source={source} | type={doc_type}")
    print(f"      {preview}")
    print()

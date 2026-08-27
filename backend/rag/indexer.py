"""
Script de indexación para el RAG vectorial.

Uso:
    python -m backend.rag.indexer <carpeta_de_documentos>

Formatos soportados:
    - Texto plano : .txt
    - Imágenes    : .jpg  .jpeg  .png  .webp  (descripción via Llama 4 Scout)
    - PDF         : .pdf  (extracción de texto via pypdf)

Los documentos ya indexados se detectan por ID y se saltean automáticamente.
"""
import argparse
import base64
import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from groq import Groq

_CHROMA_PATH = Path(__file__).parent / "chroma_db"
_COLLECTION_NAME = "knowledge"
_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TEXT_EXTS  = {".txt"}
PDF_EXTS   = {".pdf"}
ALL_EXTS   = IMAGE_EXTS | TEXT_EXTS | PDF_EXTS

_VISION_PROMPT = (
    "Describí detalladamente esta imagen para un sistema RAG de asistencia visual. "
    "Si es una factura de servicio: indicá empresa emisora, monto total, fechas de vencimiento, número de cuenta o cliente, y cualquier dato relevante visible. "
    "Si son billetes de pesos argentinos: indicá denominación, colores predominantes, figura o retrato impreso, y posición en la imagen. "
    "Respondé solo con la descripción en español, sin saludos ni comentarios adicionales."
)

_MIME_MAP = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _get_collection():
    _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    ef = SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def _doc_id(path: Path) -> str:
    return hashlib.md5(str(path.resolve()).encode()).hexdigest()


def _describe_image(groq_client: Groq, path: Path) -> str:
    mime = _MIME_MAP.get(path.suffix.lower(), "image/jpeg")
    b64 = base64.b64encode(path.read_bytes()).decode()
    response = groq_client.chat.completions.create(
        model=_VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": _VISION_PROMPT},
            ],
        }],
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        print("ERROR: pypdf no instalado. Instalá con: pip install pypdf")
        sys.exit(1)
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(p.strip() for p in pages if p.strip())


def index_folder(folder: Path):
    if not folder.exists() or not folder.is_dir():
        print(f"ERROR: '{folder}' no es una carpeta válida.")
        sys.exit(1)

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: GROQ_API_KEY no configurada en .env")
        sys.exit(1)

    groq_client = Groq(api_key=api_key)
    collection = _get_collection()

    files = sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in ALL_EXTS)
    if not files:
        print(f"No se encontraron archivos soportados en '{folder}'.")
        return

    print(f"Encontrados {len(files)} archivo(s) para indexar en '{folder}'.\n")

    added = skipped = errors = 0

    for path in files:
        fid = _doc_id(path)
        existing = collection.get(ids=[fid])
        if existing["ids"]:
            print(f"  [SKIP]  {path.name}")
            skipped += 1
            continue

        suffix = path.suffix.lower()
        print(f"  [INDEX] {path.name} ...", end=" ", flush=True)

        try:
            if suffix in IMAGE_EXTS:
                text = _describe_image(groq_client, path)
                doc_type = "image"
            elif suffix in TEXT_EXTS:
                text = path.read_text(encoding="utf-8")
                doc_type = "text"
            else:
                text = _extract_pdf(path)
                doc_type = "pdf"

            if not text.strip():
                print("vacío, saltando.")
                skipped += 1
                continue

            collection.add(
                documents=[text],
                ids=[fid],
                metadatas=[{"source": path.name, "type": doc_type}],
            )
            print("OK")
            added += 1

        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    print(f"\nIndexado completo: {added} nuevo(s), {skipped} saltado(s), {errors} error(es).")
    print(f"Total documentos en la colección: {collection.count()}")


def main():
    parser = argparse.ArgumentParser(
        description="Indexa documentos (txt, imágenes, PDF) en el RAG vectorial."
    )
    parser.add_argument("folder", type=Path, help="Carpeta con los documentos a indexar.")
    args = parser.parse_args()
    index_folder(args.folder)


if __name__ == "__main__":
    main()

"""
Backend principal - FastAPI
Agente IA para asistencia visual en pago de facturas.
"""
import asyncio
import base64
import io
import os
from pathlib import Path

from dotenv import load_dotenv

# Carga el .env desde la raíz del proyecto (path absoluto para evitar problemas de cwd)
_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_FILE)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage

from backend.models.schemas import ChatRequest, ChatResponse
from backend.sessions.manager import get_or_create_session, reset_transaction
from backend.agent.graph import graph, get_client
from backend.agent.state import AgentState
from backend.observability.logger import log_session_start, get_session_logs

app = FastAPI(title="Agente IA - Asistencia Visual Facturas", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir frontend estático
_FRONTEND = Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


@app.on_event("startup")
async def _warmup():
    """
    Pre-carga el modelo de embeddings (sentence-transformers) en RAM al arrancar.
    Sin esto, la primera request de imagen tarda varios segundos mientras
    sentence-transformers carga los ~199 tensores del modelo desde disco.
    """
    def _load():
        from backend.rag.knowledge_base import retrieve_context
        retrieve_context("warmup factura billete")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load)
    print("[startup] Modelo de embeddings RAG cargado en memoria.")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    index = _FRONTEND / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "ok", "message": "Agente IA backend corriendo."}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint principal de interacción con el agente.
    Acepta texto y opcionalmente imagen en base64.
    """
    session = get_or_create_session(request.session_id)

    # Registrar inicio de sesión si es nueva
    if not session.conversation_history:
        log_session_start(session.session_id)

    # Procesar imagen adjunta si viene
    if request.image_base64:
        try:
            img_bytes = base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="La imagen en base64 no es válida.")

        mime = request.image_mime or "image/jpeg"
        purpose = request.image_purpose or "invoice"

        if purpose == "bills":
            session.current_bills_bytes = img_bytes
            session.current_bills_mime = mime
        else:
            session.current_invoice_bytes = img_bytes
            session.current_invoice_mime = mime

    # Construir mensaje humano (puede ser solo imagen o texto+imagen)
    user_text = request.message or ("Imagen de billetes recibida." if request.image_purpose == "bills" else "Imagen de factura recibida.")

    # Estado inicial para el grafo
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_text)],
        "invoice_data": session.invoice_data,
        "bills_data": session.bills_data,
        "payment_result": session.payment_result,
        "invoice_image_bytes": session.current_invoice_bytes,
        "invoice_image_mime": session.current_invoice_mime,
        "bills_image_bytes": session.current_bills_bytes,
        "bills_image_mime": session.current_bills_mime,
        "next_action": None,
        "session_id": session.session_id,
        "rag_context": "",
        "final_response": None,
        "step": "inicio",
    }

    # Ejecutar el grafo del agente
    try:
        result = await graph.ainvoke(initial_state)
    except Exception as e:
        from backend.observability.logger import log_error
        log_error(session.session_id, "graph_invoke", str(e))
        raise HTTPException(status_code=500, detail="El asistente tuvo un problema. Por favor intentá de nuevo.")

    from backend.models.schemas import InvoiceData, PaymentResult, Bill
    from backend.agent.prompts import format_payment_result

    # Actualizar estado de la sesión con los resultados
    if result.get("invoice_data"):
        session.invoice_data = result["invoice_data"]

    if result.get("bills_data") is not None:
        if request.add_bills and session.bills_data:
            # Acumular: billetes nuevos se suman a los ya escaneados anteriormente
            session.bills_data = session.bills_data + result["bills_data"]
        else:
            session.bills_data = result["bills_data"]

    # Limpiar imágenes de la sesión (ya procesadas)
    session.current_invoice_bytes = None
    session.current_bills_bytes = None

    # Si se agregaron billetes: recalcular con el total acumulado (determinístico, sin LLM)
    final_response = result.get("final_response") or "¿En qué más puedo ayudarte?"
    payment_result_out = None

    if request.add_bills and session.bills_data and session.invoice_data:
        from backend.tools.payment_calculator import calcular_cambio_y_pago
        _invoice = InvoiceData(**session.invoice_data)
        _bills = [Bill(**b) for b in session.bills_data]
        _recalc = calcular_cambio_y_pago(session.session_id, _invoice, _bills)
        session.payment_result = _recalc.model_dump()
        final_response = format_payment_result(_recalc, _invoice)
        payment_result_out = _recalc
    elif result.get("payment_result"):
        pr = result["payment_result"]
        payment_result_out = PaymentResult(
            total_available=pr["total_available"],
            total_required=pr["total_required"],
            sufficient=pr["sufficient"],
            bills_to_use=[Bill(**b) for b in pr.get("bills_to_use", [])],
            change=pr["change"],
            bills_to_keep=[Bill(**b) for b in pr.get("bills_to_keep", [])],
            missing_amount=pr.get("missing_amount", 0.0),
        )
        session.payment_result = result["payment_result"]

    # Guardar historial conversacional
    session.conversation_history.append({"role": "user", "content": user_text})
    session.conversation_history.append({"role": "assistant", "content": final_response})

    invoice_data_out = None
    if result.get("invoice_data"):
        invoice_data_out = InvoiceData(**result["invoice_data"])

    return ChatResponse(
        response=final_response,
        session_id=session.session_id,
        step=result.get("step", "fin"),
        invoice_data=invoice_data_out,
        payment_result=payment_result_out,
    )


@app.post("/api/tts")
async def text_to_speech(body: dict):
    """
    Convierte texto a audio MP3 con gTTS (acento es-AR).
    Usado por mobile donde speechSynthesis no funciona tras operaciones async.
    """
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texto vacío.")

    def _generate() -> bytes:
        from gtts import gTTS
        tts = gTTS(text=text[:800], lang='es', tld='com.ar', slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()

    try:
        audio_bytes = await asyncio.get_event_loop().run_in_executor(None, _generate)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar audio: {str(e)}")


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio con Groq Whisper (STT sin HTTPS para mobile).
    Acepta audio/* y video/* — en Android, capture=microphone puede enviar distintos formatos.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="El archivo supera 25 MB.")

    # Determinar nombre de archivo con extensión correcta para Groq Whisper
    mime = audio.content_type or "audio/mpeg"
    ext_map = {
        "audio/mpeg": "mp3", "audio/mp3": "mp3",
        "audio/mp4": "mp4", "audio/m4a": "m4a",
        "audio/wav": "wav", "audio/x-wav": "wav",
        "audio/ogg": "ogg", "audio/webm": "webm",
        "video/mp4": "mp4", "video/webm": "webm",
        "video/3gpp": "mp4", "video/quicktime": "mp4",
    }
    ext = ext_map.get(mime, "mp3")
    filename = f"audio.{ext}"

    def _transcribe() -> str:
        client = get_client()
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            language="es",
            response_format="text",
        )
        return transcription if isinstance(transcription, str) else transcription.text

    try:
        transcript = await asyncio.get_event_loop().run_in_executor(None, _transcribe)
        return {"transcript": transcript.strip() if transcript else ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al transcribir: {str(e)}")


@app.post("/api/upload", response_model=dict)
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form("invoice"),
    session_id: str = Form(None),
):
    """
    Upload de archivo (imagen o PDF) por multipart form.
    Retorna la imagen en base64 para que el frontend la envíe en /api/chat.
    """
    allowed_types = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo no soportado: {file.content_type}. Usá JPEG, PNG, WebP o PDF."
        )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="El archivo supera 10 MB.")

    return {
        "image_base64": base64.b64encode(contents).decode(),
        "image_mime": file.content_type,
        "image_purpose": purpose,
        "filename": file.filename,
        "size_bytes": len(contents),
    }


@app.post("/api/reset")
async def reset_session(session_id: str):
    """Reinicia el estado de la transacción actual (sin borrar el historial)."""
    from backend.sessions.manager import get_session
    session = get_session(session_id)
    if session:
        reset_transaction(session)
    return {"ok": True, "message": "Transacción reiniciada."}


@app.get("/api/logs/{session_id}")
async def get_logs(session_id: str):
    """Endpoint de observabilidad: retorna los logs de una sesión."""
    logs = get_session_logs(session_id)
    return {"session_id": session_id, "log_count": len(logs), "logs": logs}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "text_model": os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile"),
        "vision_model": os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
    }


# ─── Arranque ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )

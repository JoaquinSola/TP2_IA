"""
Tool: extraer_datos_factura
Analiza imagen o PDF de factura usando Gemini vision y retorna datos estructurados.
"""
import io
import json
import os
import re
from typing import Optional

from PIL import Image

from backend.models.schemas import InvoiceData
from backend.rag.knowledge_base import retrieve_context
from backend.observability.logger import log_llm_call, log_tool_call, log_tool_result, Timer


_EXTRACTION_PROMPT = """Sos un sistema especializado en extraer el monto a pagar de cualquier comprobante de pago de la REPÚBLICA ARGENTINA.
La moneda es PESOS ARGENTINOS (ARS). Todos los montos deben interpretarse en pesos argentinos.

CONTEXTO RELEVANTE:
{rag_context}

TIPOS DE COMPROBANTE VÁLIDOS (todos son is_valid_document = true):
- Facturas de servicios públicos: EPE (electricidad), ASSA (agua), Litoral Gas / Naturgy (gas), Telecom, Claro, Personal, Movistar, EDESUR, EDENOR, AySA, METROGAS, municipalidades, AFIP, ARBA.
- Tickets de supermercado o comercio (Carrefour, Coto, Jumbo, Disco, Farmacity, ferretería, etc.): papel térmico angosto, lista de productos, TOTAL al pie. SIN fecha de vencimiento.
- Tickets fiscales emitidos por controladora fiscal (leyenda "Controlador Fiscal", "CF").
- Facturas electrónicas tipo A, B o C con código CAE (AFIP). La fecha del CAE NO es vencimiento de pago.
- Comprobantes de servicio técnico, plomería, electricista, reparaciones.
- Boletas de colegios, clubes, gimnasios, cuotas de asociaciones.
- Recibos de alquiler, expensas, peaje, estacionamiento.

INSTRUCCIONES DE EXTRACCIÓN:
1. Identificá la empresa, comercio o entidad emisora (ej: "EPE", "Carrefour", "Plomería García").
2. Extraé el TOTAL A PAGAR:
   - Para facturas de servicios: tomá el importe del PRIMER VENCIMIENTO como total_amount.
   - Para tickets de caja/supermercado: tomá el TOTAL que aparece al pie del ticket.
   - Para otros comprobantes: el monto total indicado.
3. Si hay SEGUNDO VENCIMIENTO (solo en facturas de servicios), extraélo como second_amount y su fecha como second_due_date.
4. Fechas: formato DD/MM/YYYY. Si el comprobante no tiene fecha de vencimiento (ticket de caja, servicio técnico), dejá due_date en null.
5. FORMATO NUMÉRICO ARGENTINO — CRÍTICO:
   - En Argentina el PUNTO separa miles y la COMA separa decimales. Ej: "22.000,50" = 22000.50 pesos.
   - Devolvé el monto en formato internacional sin puntos de miles.
   - Ejemplos: "22.000,00" → 22000 | "1.500,50" → 1500.50 | "$34.000" → 34000 | "850,00" → 850.
   - NUNCA devuelvas 22.0 cuando dice "22.000" — eso es veintidós MIL, no veintidós.
   - Los montos de facturas y tickets domésticos en Argentina están entre 100 y 10.000.000 pesos.
   - IGNORÁ códigos de barras, CBU, CUIT y números de cuenta — tienen 13 o más dígitos y NO son montos.
6. Marcá is_valid_document como FALSE únicamente si la imagen NO muestra ningún comprobante de pago (ej: una selfie, una foto de paisaje, un documento de identidad, texto sin montos).
7. Respondé SIEMPRE en formato JSON puro, sin texto adicional, sin bloques markdown.

FORMATO DE RESPUESTA (JSON estricto, sin ```):
{{
  "entity": "nombre del comercio o empresa emisora, o null",
  "total_amount": número en pesos o null,
  "due_date": "DD/MM/YYYY o null si no aplica vencimiento",
  "second_due_date": "DD/MM/YYYY o null",
  "second_amount": número en pesos o null,
  "is_valid_document": true o false,
  "error_message": "motivo solo si is_valid_document es false, sino null"
}}

CRÍTICO: NO inventes datos. Si no podés leer un campo con certeza, usá null."""


def _prepare_image(image_bytes: bytes) -> tuple[bytes, str]:
    try:
        from PIL import ImageOps
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')
        max_dim = 1280
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=90)
        return output.getvalue(), 'image/jpeg'
    except Exception:
        return image_bytes, 'image/jpeg'


def extraer_datos_factura(
    session_id: str,
    image_bytes: Optional[bytes],
    image_mime: Optional[str],
) -> InvoiceData:
    log_tool_call(session_id, "extraer_datos_factura", {"mime": image_mime, "size_bytes": len(image_bytes) if image_bytes else 0})

    if not image_bytes:
        result = InvoiceData(
            is_valid_document=False,
            error_message="No se proporcionó ninguna imagen de factura. Por favor, adjuntá una foto o archivo PDF."
        )
        log_tool_result(session_id, "extraer_datos_factura", result.model_dump(), 0)
        return result

    mime = image_mime or "image/jpeg"

    if mime == "application/pdf":
        rag_context = ""
    else:
        rag_context = retrieve_context("factura servicio empresa proveedor")

    prompt = _EXTRACTION_PROMPT.format(rag_context=rag_context)
    raw, t, token_count = _call_gemini(prompt, image_bytes, mime)

    log_llm_call(
        session_id=session_id,
        node="extraer_datos_factura",
        prompt_summary=prompt[:200],
        response_summary=raw[:200],
        latency_ms=t.elapsed_ms,
        tokens_used=token_count,
    )

    result = _parse_invoice_json(raw)
    log_tool_result(session_id, "extraer_datos_factura", result.model_dump(), t.elapsed_ms)
    return result


def _call_gemini(prompt: str, data_bytes: bytes, mime: str):
    from google.genai import types as genai_types
    from backend.agent.graph import call_gemini_with_retry

    if mime == "application/pdf":
        part = genai_types.Part.from_bytes(data=data_bytes, mime_type="application/pdf")
    else:
        img_bytes, img_mime = _prepare_image(data_bytes)
        part = genai_types.Part.from_bytes(data=img_bytes, mime_type=img_mime)

    contents = [part, genai_types.Part.from_text(text=prompt)]

    config = genai_types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=512,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )
    raw = ""
    token_count = None
    with Timer() as t:
        try:
            response, _ = call_gemini_with_retry(contents, config, label="invoice_extractor")
            raw = response.text.strip() if response.text else ""
            token_count = response.usage_metadata.total_token_count if response.usage_metadata else None
        except Exception as e:
            print(f"[invoice_extractor] Gemini falló: {e}")
            raw = ""
            token_count = None
    return raw, t, token_count


def _parse_invoice_json(raw: str) -> InvoiceData:
    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        clean = re.sub(r'\s*//[^\n\r]*', '', clean)
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not json_match:
            raise ValueError("No se encontró JSON")
        data = json.loads(json_match.group())
        return InvoiceData(
            entity=data.get("entity"),
            total_amount=_parse_amount(data.get("total_amount")),
            due_date=data.get("due_date"),
            second_due_date=data.get("second_due_date"),
            second_amount=_parse_amount(data.get("second_amount")),
            is_valid_document=bool(data.get("is_valid_document", True)),
            error_message=data.get("error_message"),
        )
    except Exception:
        return InvoiceData(
            is_valid_document=False,
            error_message="No pude leer los datos de la factura claramente. Por favor, tomá una foto más nítida con mejor iluminación."
        )


def _parse_amount(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.replace("$", "").replace(".", "").replace(",", ".").strip()
        amount = float(value)
        if amount <= 0 or amount > 50_000_000:
            return None
        return amount
    except (ValueError, TypeError):
        return None

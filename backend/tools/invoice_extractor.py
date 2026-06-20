"""
Tool: extraer_datos_factura
Analiza imagen o PDF de factura usando Groq vision y retorna datos estructurados.
"""
import base64
import io
import json
import re
from typing import Optional

from groq import Groq
from PIL import Image

from backend.models.schemas import InvoiceData
from backend.rag.knowledge_base import retrieve_context
from backend.observability.logger import log_llm_call, log_tool_call, log_tool_result, Timer


_EXTRACTION_PROMPT = """Sos un sistema especializado en extraer datos de facturas de servicios de la REPÚBLICA ARGENTINA.
La moneda es PESOS ARGENTINOS (ARS, símbolo $). Todos los montos deben interpretarse en pesos argentinos.

CONTEXTO RELEVANTE SOBRE PROVEEDORES DE SERVICIOS ARGENTINOS:
{rag_context}

PROVEEDORES COMUNES EN ARGENTINA: EPE (electricidad Santa Fe), ASSA (agua Santa Fe), Litoral Gas / Naturgy (gas),
Telecom, Claro, Personal, Movistar, EDESUR, EDENOR, AySA, METROGAS, municipalidades, AFIP, ARBA.

Las facturas argentinas suelen tener:
- Nombre o razón social de la empresa
- CUIT del emisor y del cliente
- Período de facturación
- PRIMER VENCIMIENTO (importe base)
- SEGUNDO VENCIMIENTO (importe con recargo por mora, generalmente mayor)
- Código de barras o QR para pago electrónico

INSTRUCCIONES:
1. Identificá la empresa o entidad que emite la factura (ej: "EPE", "ASSA", "Litoral Gas").
2. Extraé el importe del PRIMER VENCIMIENTO como total_amount (o el único monto si hay uno solo).
3. Si hay SEGUNDO VENCIMIENTO, extraélo como second_amount y su fecha como second_due_date.
4. Las fechas van en formato DD/MM/YYYY.
5. Los montos son SIEMPRE en Pesos Argentinos (ARS). ATENCIÓN CON EL FORMATO ARGENTINO:
   - En Argentina el PUNTO separa miles y la COMA separa decimales. Ej: "22.000,50" = 22000.50 pesos.
   - Devolvé el monto como número entero o decimal en formato internacional (sin puntos de miles).
   - Ejemplos correctos: "22.000,00" → 22000, "1.500,50" → 1500.50, "$34.000" → 34000.
   - NUNCA devuelvas 22.0 cuando la factura dice "22.000" — eso es veintiún mil, no veintidós.
6. Si el documento NO es una factura o boleta de servicios, marcá is_valid_document como false.
7. Respondé SIEMPRE en formato JSON puro, sin texto adicional, sin bloques markdown.

FORMATO DE RESPUESTA (JSON estricto, sin ```):
{{
  "entity": "nombre de la empresa emisora en Argentina o null",
  "total_amount": número en pesos o null,
  "due_date": "DD/MM/YYYY o null",
  "second_due_date": "DD/MM/YYYY o null si no hay segundo vencimiento",
  "second_amount": número en pesos o null,
  "is_valid_document": true o false,
  "error_message": "motivo si is_valid_document es false, sino null"
}}

CRÍTICO: NO inventes datos. Si no podés leer un campo con certeza, usá null. Los montos son pesos argentinos."""


def _prepare_image(image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    """Redimensiona la imagen si supera 3MB para no exceder límites de la API."""
    if len(image_bytes) <= 3 * 1024 * 1024:
        return image_bytes, mime_type
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')
        max_dim = 1920
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=85)
        return output.getvalue(), 'image/jpeg'
    except Exception:
        return image_bytes, mime_type


def extraer_datos_factura(
    session_id: str,
    image_bytes: Optional[bytes],
    image_mime: Optional[str],
    client: Groq,
    model_name: str,
) -> InvoiceData:
    log_tool_call(session_id, "extraer_datos_factura", {"mime": image_mime, "size_bytes": len(image_bytes) if image_bytes else 0})

    if not image_bytes:
        result = InvoiceData(
            is_valid_document=False,
            error_message="No se proporcionó ninguna imagen de factura. Por favor, adjuntá una foto o archivo PDF."
        )
        log_tool_result(session_id, "extraer_datos_factura", result.model_dump(), 0)
        return result

    rag_context = retrieve_context("factura servicio empresa proveedor")
    prompt = _EXTRACTION_PROMPT.format(rag_context=rag_context)
    mime = image_mime or "image/jpeg"

    img_bytes, img_mime = _prepare_image(image_bytes, mime)
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    with Timer() as t:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{img_mime};base64,{img_b64}"}},
                    ],
                }],
                temperature=0.1,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip() if response.choices else ""
        except Exception as e:
            log_tool_result(session_id, "extraer_datos_factura", {"error": str(e)}, 0)
            return InvoiceData(
                is_valid_document=False,
                error_message="No pude procesar la imagen en este momento. Verificá tu conexión y volvé a intentar."
            )

    token_count = response.usage.total_tokens if response.usage else None
    log_llm_call(
        session_id=session_id,
        node="extraer_datos_factura",
        prompt_summary=prompt[:200],
        response_summary=raw[:200],
        latency_ms=t.elapsed_ms,
        tokens_used=token_count,
    )

    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not json_match:
            raise ValueError("No se encontró JSON en la respuesta")
        data = json.loads(json_match.group())
        result = InvoiceData(
            entity=data.get("entity"),
            total_amount=_parse_amount(data.get("total_amount")),
            due_date=data.get("due_date"),
            second_due_date=data.get("second_due_date"),
            second_amount=_parse_amount(data.get("second_amount")),
            is_valid_document=bool(data.get("is_valid_document", True)),
            error_message=data.get("error_message"),
        )
    except Exception:
        result = InvoiceData(
            is_valid_document=False,
            error_message="No pude leer los datos de la factura claramente. Por favor, tomá una foto más nítida con mejor iluminación."
        )

    log_tool_result(session_id, "extraer_datos_factura", result.model_dump(), t.elapsed_ms)
    return result


def _parse_amount(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.replace("$", "").replace(".", "").replace(",", ".").strip()
        return float(value)
    except (ValueError, TypeError):
        return None

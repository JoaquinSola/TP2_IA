"""
Tool: identificar_billetes
Analiza imagen de dinero en efectivo usando Groq vision y retorna lista estructurada de billetes.
"""
import base64
import io
import json
import re
from typing import Optional

from groq import Groq
from PIL import Image

from backend.models.schemas import Bill
from backend.rag.knowledge_base import retrieve_context
from backend.observability.logger import log_llm_call, log_tool_call, log_tool_result, Timer


_IDENTIFICATION_PROMPT = """Sos un sistema de identificación de billetes de PESOS ARGENTINOS para asistir a personas con discapacidad visual en la Argentina.
Analizá la imagen adjunta que contiene billetes de papel moneda sobre una superficie.

CONTEXTO SOBRE BILLETES ARGENTINOS:
{rag_context}

BILLETES DE PESOS ARGENTINOS EN CIRCULACIÓN:

- $100 → Colores violeta o marrón. Diseños con Eva Perón, Roca o la Taruca (ciervo).
- $200 → Color rosado. Diseños con la Ballena Franca Austral o Martín Miguel de Güemes y Juana Azurduy.
- $500 → Color verde. Diseños con el Yaguareté o Manuel Belgrano y María Remedios del Valle.
- $1.000 → Color naranja o beige. Diseños con el Hornero o José de San Martín.
- $2.000 → Color rojo y gris oscuro. Diseños con los médicos Ramón Carrillo y Cecilia Grierson.
- $10.000 → Color celeste y gris azulado. Diseños con Manuel Belgrano y María Remedios del Valle.
- $20.000 → Color azul. Diseños con Juan Bautista Alberdi.

INSTRUCCIONES CRÍTICAS:
1. PRIMERO leé el NÚMERO impreso en el billete (ángulo, color y tamaño del número son pistas secundarias).
2. El número impreso en el billete ES la denominación. Confiá en el número, no en el color.
3. Identificá CADA billete visible, incluyendo los parcialmente tapados.
4. POSICIÓN ESPACIAL: describí la posición tal como el usuario ve los billetes sobre la mesa mirando hacia abajo. La imagen ya está orientada correctamente. Usá: "izquierda", "centro", "derecha", "arriba", "abajo", "arriba-izquierda", "arriba-derecha", "abajo-izquierda", "abajo-derecha". El eje horizontal (izquierda/derecha) es el lado largo de la mesa; el eje vertical (arriba/abajo) es el lado corto.
5. Marcá valid=false SOLO si el billete es CLARAMENTE una moneda extranjera (dólar, euro, real) o un billete de juego (Monopoly, fichas, etc.) o claramente falso.
6. Billetes argentinos de CUALQUIER denominación → valid=true (incluyendo los viejos de $100, $200, $500).
7. Si no hay billetes en la imagen, retorná lista vacía.
8. Respondé SIEMPRE en formato JSON puro, sin markdown, sin bloques de código.

FORMATO DE RESPUESTA (JSON estricto):
{{
  "bills": [
    {{
      "denomination": número entero (ej: 5000, 10000, 1000),
      "position": "posición espacial",
      "valid": true si es ARS (de cualquier serie), false si es moneda extranjera o billete de juego,
      "currency": "ARS",
      "confidence": número entre 0.0 y 1.0
    }}
  ],
  "description": "descripción breve en español de lo que ves"
}}

CRÍTICO: denomination debe ser un número ENTERO sin puntos ni comas (ej: 5000, NO "5.000" ni "5,000").
NO inventes billetes. Solo incluí los que ves con claridad."""


def _prepare_image(image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    """
    Corrige orientación EXIF (fotos de celular en retrato) y redimensiona si supera 3MB.
    Sin esto, el LLM recibe la imagen rotada 90° y confunde izquierda/derecha con arriba/abajo.
    """
    try:
        from PIL import ImageOps
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)  # aplica la rotación EXIF a los píxeles
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


def _parse_denomination(value) -> int:
    """Parsea denominación en cualquier formato: 5000, "5000", "5.000", "$5.000", 5000.0"""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        clean = value.replace("$", "").replace(".", "").replace(",", "").replace(" ", "").strip()
        try:
            return int(float(clean))
        except (ValueError, TypeError):
            return 0
    return 0


def identificar_billetes(
    session_id: str,
    image_bytes: Optional[bytes],
    image_mime: Optional[str],
    client: Groq,
    model_name: str,
) -> list[Bill]:
    log_tool_call(session_id, "identificar_billetes", {"mime": image_mime, "size_bytes": len(image_bytes) if image_bytes else 0})

    if not image_bytes:
        log_tool_result(session_id, "identificar_billetes", [], 0)
        return []

    rag_context = retrieve_context("billete peso argentino denominación moneda")
    prompt = _IDENTIFICATION_PROMPT.format(rag_context=rag_context)
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
            log_tool_result(session_id, "identificar_billetes", {"error": str(e)}, 0)
            return []

    token_count = response.usage.total_tokens if response.usage else None
    log_llm_call(
        session_id=session_id,
        node="identificar_billetes",
        prompt_summary=prompt[:200],
        response_summary=raw[:200],
        latency_ms=t.elapsed_ms,
        tokens_used=token_count,
    )

    bills: list[Bill] = []
    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not json_match:
            raise ValueError("No se encontró JSON")
        data = json.loads(json_match.group())

        for b in data.get("bills", []):
            try:
                denom = _parse_denomination(b.get("denomination", 0))
            except Exception:
                continue
            if denom <= 0:
                continue
            is_valid = bool(b.get("valid", True))
            bills.append(Bill(
                denomination=denom,
                position=str(b.get("position", "centro")),
                valid=is_valid,
                currency=str(b.get("currency", "ARS")),
                confidence=float(b.get("confidence", 1.0)),
            ))
    except Exception:
        pass

    log_tool_result(session_id, "identificar_billetes", [b.model_dump() for b in bills], t.elapsed_ms)
    return bills

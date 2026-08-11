"""
Tool: identificar_billetes
Analiza imagen de dinero en efectivo usando Gemini vision y retorna lista estructurada de billetes.
"""
import io
import json
import os
import re
from typing import Optional

from PIL import Image

from backend.models.schemas import Bill
from backend.observability.logger import log_llm_call, log_tool_call, log_tool_result, Timer


_IDENTIFICATION_PROMPT = """Sos un asistente de accesibilidad visual que identifica billetes de pesos argentinos para personas con discapacidad visual.

═══ PASO 1: ESCENARIO ═══
Determiná si los billetes están:
- "mano": se ven dedos o palma sujetando los billetes en abanico o pila
- "superficie": billetes sobre mesa u otra superficie plana, sin manos visibles

═══ PASO 2: IDENTIFICÁ CADA BILLETE ═══

Usá estas pistas en simultáneo:
  A) NÚMERO impreso en el billete — si podés leerlo claramente, ES la denominación (prioridad máxima)
  B) COLOR DOMINANTE de la zona visible
  C) CÓDIGO TÁCTIL: letra/s en relieve en alguna esquina
  D) ORIENTACIÓN: VERTICAL (más alto que ancho) o HORIZONTAL

⚠ EN ABANICO (escenario "mano"): analizá de atrás hacia adelante, cada billete POR SU PROPIO COLOR.
  El número "500" o "1000" que asoma entre billetes puede ser del billete de FONDO — no lo asignes al del frente.

BILLETES VIGENTES (valid=true):
  100 → VIOLETA/ROSA intenso (vertical "C" o horizontal Evita)
  200 → AZUL CELESTE claro (vertical "CC") o GRIS AZULADO (horizontal, militares independencia)
  500 → VERDE INTENSO/esmeralda (horizontal "D") o VERDE OLIVA/apagado (horizontal, militares coloniales)
  1000 → NARANJA/MARRÓN CLARO (vertical "M") o NARANJA/SALMÓN (horizontal, cordillera)
  2000 → GRIS OSCURO con trama rosada/bordó (horizontal)
  10000 → TURQUESA/CELESTE INTENSO (horizontal)
  20000 → AZUL con detalles dorados (horizontal)

BILLETES FUERA DE CIRCULACIÓN (valid=false):
  10 → MARRÓN/OCRE rojizo | 20 → ROJO/ROSADO ("XX") | 50 → VERDE AZULADO ("L") o AZUL MARINO

CONFUSIONES FRECUENTES:
  • GRIS AZULADO + uniformes militares → 200 Güemes ✓  (NO confundir con 100)
  • VIOLETA/LILA + escultura o ceibo rojo → 100 Evita ✓
  • NARANJA cálido (fruta) → 1000 ✓  |  VIOLETA/ROSADO frío (uva) → 100 ✓
  • VERDE BRILLANTE → 500 vigente ✓  |  VERDE AZULADO/TEAL → 50 desmonetizado ✗
  • En abanico: el número del billete de atrás NO pertenece al del frente

═══ PASO 3: POSICIÓN ═══
  Superficie → izquierda / centro / derecha / arriba-izquierda / arriba-derecha / abajo-izquierda / abajo-derecha
  Mano/abanico → primero / segundo / tercero / cuarto / quinto (de izquierda a derecha visto de frente)

═══ RESPUESTA ═══
Respondé ÚNICAMENTE con este JSON, sin texto adicional ni markdown:
{{
  "escenario": "superficie" o "mano",
  "bills": [
    {{
      "denomination": número entero (ej: 500, 1000, 10000),
      "position": "posición según las reglas",
      "valid": true o false,
      "currency": "ARS",
      "confidence": número entre 0.0 y 1.0
    }}
  ],
  "description": "descripción breve en español de lo que ves"
}}

Solo incluí los billetes que realmente ves. Sin inventar. Máximo 5."""


def _prepare_image(image_bytes: bytes) -> tuple[bytes, str]:
    try:
        from PIL import ImageOps
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
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
        return image_bytes, 'image/jpeg'


def _parse_denomination(value) -> int:
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
) -> list[Bill]:
    log_tool_call(session_id, "identificar_billetes", {"mime": image_mime, "size_bytes": len(image_bytes) if image_bytes else 0})

    if not image_bytes:
        log_tool_result(session_id, "identificar_billetes", [], 0)
        return []

    from google.genai import types as genai_types
    from backend.agent.graph import call_gemini_with_retry

    img_bytes, img_mime = _prepare_image(image_bytes)

    contents = [
        genai_types.Part.from_bytes(data=img_bytes, mime_type=img_mime),
        genai_types.Part.from_text(text=_IDENTIFICATION_PROMPT),
    ]
    config = genai_types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=1024,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )

    raw = ""
    token_count = None
    with Timer() as t:
        try:
            response, used_model = call_gemini_with_retry(contents, config, label="bill_identifier")
            raw = response.text.strip() if response.text else ""
            token_count = response.usage_metadata.total_token_count if response.usage_metadata else None
            print(f"[DEBUG bill_identifier] raw gemini response ({used_model}):\n{raw[:2000]}")
        except Exception as e:
            print(f"[DEBUG bill_identifier] gemini error: {e}")
            log_tool_result(session_id, "identificar_billetes", {"error": str(e)}, 0)
            return []

    log_llm_call(
        session_id=session_id,
        node="identificar_billetes",
        prompt_summary=_IDENTIFICATION_PROMPT[:200],
        response_summary=raw[:200],
        latency_ms=t.elapsed_ms,
        tokens_used=token_count,
    )

    bills: list[Bill] = []
    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        clean = re.sub(r'\s*//[^\n\r]*', '', clean)

        scenario = "mano" if re.search(r'\bmano\b|\babanico\b', clean, re.IGNORECASE) else "superficie"
        raw_bills: list[dict] = []
        data = None

        for start in reversed([m.start() for m in re.finditer(r'\{', clean)]):
            try:
                candidate = json.loads(clean[start:])
                if isinstance(candidate, dict) and ("bills" in candidate or "escenario" in candidate):
                    data = candidate
                    break
            except Exception:
                continue

        if data:
            sc = str(data.get("escenario", scenario)).lower()
            if sc in ("superficie", "mano"):
                scenario = sc
            raw_bills = data.get("bills") or []
            if not raw_bills:
                for key, val in data.items():
                    if key not in ("escenario", "description") and isinstance(val, list):
                        raw_bills.extend(val)
        else:
            if "mano" in clean.lower() or "abanico" in clean.lower():
                scenario = "mano"
            for m in re.finditer(r'\{[^{}]*"denomination"[^{}]*\}', clean):
                try:
                    bill = json.loads(m.group())
                    if "denomination" in bill:
                        raw_bills.append(bill)
                except Exception:
                    continue

        for b in raw_bills:
            try:
                denom = _parse_denomination(b.get("denomination", 0))
            except Exception:
                continue
            if denom <= 0:
                continue
            bills.append(Bill(
                denomination=denom,
                position=str(b.get("position", "centro")),
                valid=bool(b.get("valid", True)),
                currency=str(b.get("currency", "ARS")),
                confidence=float(b.get("confidence", 1.0)),
                scenario=scenario,
            ))
    except Exception:
        pass

    log_tool_result(session_id, "identificar_billetes", [b.model_dump() for b in bills], t.elapsed_ms)
    return bills

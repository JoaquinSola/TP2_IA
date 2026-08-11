"""
Grafo LangGraph del agente.
Implementa un loop ReAct con nodos especializados para cada herramienta.
"""
import os
import re
import time
from typing import Literal

from google import genai
from google.genai import types as genai_types
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState
from backend.agent.prompts import (
    SYSTEM_PROMPT,
    AGENT_DECISION_PROMPT,
    format_invoice_summary,
    format_bills_summary,
    format_payment_result,
    _fmt_ars,
)
from backend.rag.knowledge_base import retrieve_context
from backend.observability.logger import log_llm_call, log_error, Timer

_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
_FALLBACK_MODEL = "gemini-flash-lite-latest"
_gemini_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY no está configurada. Revisá el archivo .env.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _is_retryable(err_str: str) -> bool:
    return any(x in err_str for x in ("503", "429", "UNAVAILABLE", "Resource has been exhausted"))


def call_gemini_with_retry(contents, config, label: str = "gemini") -> tuple:
    """
    Llama a Gemini con retry + fallback de modelo.
    Intento 1: modelo primario (2.5-flash)
    Intento 2: modelo primario con 1s de espera
    Intento 3+: modelo de fallback (2.0-flash)
    Retorna (response, model_usado).
    """
    client = get_gemini_client()
    models_to_try = [_TEXT_MODEL, _FALLBACK_MODEL, _FALLBACK_MODEL]
    for attempt, model in enumerate(models_to_try):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            if attempt > 0:
                print(f"[{label}] OK en intento {attempt+1} con {model}")
            return response, model
        except Exception as e:
            err_str = str(e)
            if _is_retryable(err_str):
                wait = 1 if attempt == 0 else 2
                print(f"[{label}] 503/429 (intento {attempt+1}), reintentando con {models_to_try[attempt+1] if attempt+1 < len(models_to_try) else 'nada'} en {wait}s")
                if attempt + 1 < len(models_to_try):
                    time.sleep(wait)
                    continue
            raise


# ─── Nodos del grafo ─────────────────────────────────────────────────────────


def rag_node(state: AgentState) -> dict:
    """Recupera contexto relevante de la base de conocimiento (RAG)."""
    # Si hay imagen pendiente, agent_node cortocircuita sin usar el contexto RAG — saltearlo.
    if state.get("invoice_image_bytes") or state.get("bills_image_bytes"):
        return {"rag_context": "", "step": "rag"}
    last_msg = state["messages"][-1].content if state["messages"] else ""
    query = last_msg if isinstance(last_msg, str) else "factura pago billete"
    context = retrieve_context(query)
    return {"rag_context": context, "step": "rag"}


def agent_node(state: AgentState) -> dict:
    # Cortocircuito: si hay imágenes pendientes, rutear directamente sin consultar el LLM.
    # El LLM no es necesario ni confiable para esta decisión; las imágenes siempre tienen prioridad.
    if state.get("invoice_image_bytes"):
        return {"next_action": "extraer_datos_factura", "step": "decidiendo"}
    if state.get("bills_image_bytes"):
        return {"next_action": "identificar_billetes", "step": "decidiendo"}

    # Cortocircuito: si ya tenemos factura + billetes y el pago aún no fue calculado,
    # calcular directamente sin pasar por el LLM (evita que el LLM pida foto de nuevo).
    if state.get("invoice_data") and state.get("bills_data") and not state.get("payment_result"):
        return {"next_action": "calcular_cambio_y_pago", "step": "decidiendo"}

    session_id = state["session_id"]
    messages = state["messages"]
    last_msg = messages[-1].content if messages else ""
    if not isinstance(last_msg, str):
        last_msg = str(last_msg)

    history_text = ""
    for m in messages[:-1][-10:]:
        role = "Usuario" if isinstance(m, HumanMessage) else "Asistente"
        content = m.content if isinstance(m.content, str) else str(m.content)
        history_text += f"{role}: {content}\n"

    from backend.models.schemas import InvoiceData, Bill as BillModel
    invoice_raw = state.get("invoice_data")
    invoice_obj = InvoiceData(**invoice_raw) if invoice_raw else None
    invoice_summary = format_invoice_summary(invoice_obj) if invoice_obj else "Ninguna"

    bills_raw = state.get("bills_data") or []
    bills_objs = [BillModel(**b) for b in bills_raw] if bills_raw else []
    bills_summary = format_bills_summary(bills_objs) if bills_objs else "Ningunos"

    prompt = AGENT_DECISION_PROMPT.format(
        has_invoice="Sí" if state.get("invoice_data") else "No",
        has_bills="Sí" if state.get("bills_data") else "No",
        has_payment="Sí" if state.get("payment_result") else "No",
        awaiting=state.get("step", "ninguno"),
        has_invoice_image="Sí" if state.get("invoice_image_bytes") else "No",
        has_bills_image="Sí" if state.get("bills_image_bytes") else "No",
        invoice_summary=invoice_summary,
        bills_summary=bills_summary,
        user_message=last_msg,
        history=history_text or "(sin historial previo)",
        rag_context=state.get("rag_context", ""),
    )

    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.3,
        max_output_tokens=1024,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )
    with Timer() as t:
        try:
            response, _ = call_gemini_with_retry(prompt, config, label="agent_node")
            raw = response.text.strip() if response.text else ""
        except Exception as e:
            log_error(session_id, "agent_node", str(e))
            return {
                "next_action": "respond",
                "final_response": "Lo siento, tuve un problema técnico. Por favor intentá de nuevo.",
                "step": "error",
            }

    token_count = response.usage_metadata.total_token_count if (response and response.usage_metadata) else None
    log_llm_call(
        session_id=session_id,
        node="agent_node",
        prompt_summary=prompt[:300],
        response_summary=raw[:200],
        latency_ms=t.elapsed_ms,
        tokens_used=token_count,
    )

    if "TOOL:extraer_datos_factura" in raw:
        return {"next_action": "extraer_datos_factura", "step": "decidiendo"}
    elif "TOOL:identificar_billetes" in raw:
        return {"next_action": "identificar_billetes", "step": "decidiendo"}
    elif "TOOL:calcular_cambio_y_pago" in raw:
        return {"next_action": "calcular_cambio_y_pago", "step": "decidiendo"}
    else:
        return {
            "next_action": "respond",
            "final_response": raw,
            "messages": [AIMessage(content=raw)],
            "step": "respondiendo",
        }


def extract_invoice_node(state: AgentState) -> dict:
    from backend.tools.invoice_extractor import extraer_datos_factura

    session_id = state["session_id"]
    try:
        invoice = extraer_datos_factura(
            session_id=session_id,
            image_bytes=state.get("invoice_image_bytes"),
            image_mime=state.get("invoice_image_mime"),
        )
        summary = format_invoice_summary(invoice)
        response_msg = summary

        if not invoice.is_valid_document:
            response_msg = invoice.error_message or "No pude procesar la factura."
        elif invoice.second_due_date and invoice.second_amount:
            response_msg = (
                f"{summary}. "
                f"También hay un segundo vencimiento el {invoice.second_due_date} "
                f"por {_fmt_ars(invoice.second_amount)} pesos. "
                f"¿Con qué monto querés continuar?"
            )

        # Si ya hay billetes escaneados, ir directo al cálculo sin pedir foto de nuevo
        if invoice.is_valid_document and invoice.total_amount and state.get("bills_data"):
            return {
                "invoice_data": invoice.model_dump(),
                "next_action": "calcular_cambio_y_pago",
                "step": "factura_extraida",
            }

        if invoice.is_valid_document and invoice.total_amount:
            response_msg += " Cuando tengas los billetes listos, sacá una foto de ellos sobre la mesa."

        return {
            "invoice_data": invoice.model_dump(),
            "next_action": "respond",
            "final_response": response_msg,
            "messages": [AIMessage(content=response_msg)],
            "step": "factura_extraida",
        }
    except Exception as e:
        log_error(session_id, "extract_invoice_node", str(e))
        err_msg = "No pude procesar la factura en este momento. Verificá tu conexión y volvé a intentar."
        return {
            "next_action": "respond",
            "final_response": err_msg,
            "messages": [AIMessage(content=err_msg)],
            "step": "error",
        }


def identify_bills_node(state: AgentState) -> dict:
    from backend.tools.bill_identifier import identificar_billetes

    session_id = state["session_id"]
    try:
        bills = identificar_billetes(
            session_id=session_id,
            image_bytes=state.get("bills_image_bytes"),
            image_mime=state.get("bills_image_mime"),
        )
        summary = format_bills_summary(bills)

        if state.get("invoice_data") and bills:
            return {
                "bills_data": [b.model_dump() for b in bills],
                "next_action": "calcular_cambio_y_pago",
                "step": "billetes_identificados",
            }

        response_msg = summary
        if not bills:
            response_msg = (
                "No pude identificar billetes en la imagen. "
                "Asegurate de que los billetes estén sobre una superficie plana con buena iluminación y volvé a fotografiarlos."
            )
        elif not state.get("invoice_data"):
            response_msg = f"{summary} Si además querés calcular un pago, podés pasarme la factura."

        return {
            "bills_data": [b.model_dump() for b in bills],
            "next_action": "respond",
            "final_response": response_msg,
            "messages": [AIMessage(content=response_msg)],
            "step": "billetes_identificados",
        }
    except Exception as e:
        log_error(session_id, "identify_bills_node", str(e))
        err_msg = "No pude identificar los billetes en este momento. Volvé a intentar con mejor iluminación."
        return {
            "next_action": "respond",
            "final_response": err_msg,
            "messages": [AIMessage(content=err_msg)],
            "step": "error",
        }


def calculate_node(state: AgentState) -> dict:
    from backend.tools.payment_calculator import calcular_cambio_y_pago
    from backend.models.schemas import InvoiceData, Bill

    session_id = state["session_id"]
    try:
        invoice_dict = state.get("invoice_data") or {}
        bills_list = state.get("bills_data") or []

        invoice = InvoiceData(**invoice_dict) if invoice_dict else InvoiceData()
        bills = [Bill(**b) for b in bills_list]

        result = calcular_cambio_y_pago(session_id=session_id, invoice=invoice, bills=bills)
        full_response = format_payment_result(result, invoice)

        return {
            "payment_result": result.model_dump(),
            "next_action": "respond",
            "final_response": full_response,
            "messages": [AIMessage(content=full_response)],
            "step": "pago_calculado",
        }
    except Exception as e:
        log_error(session_id, "calculate_node", str(e))
        err_msg = "Tuve un problema al calcular el pago. Por favor, intentá enviar la foto de los billetes de nuevo."
        return {
            "next_action": "respond",
            "final_response": err_msg,
            "messages": [AIMessage(content=err_msg)],
            "step": "error",
        }


def _sanitize_currency(text: str) -> str:
    """
    Red de seguridad: elimina el símbolo $ antes de cualquier número en el texto final.
    Convierte "$9.479,02" → "9.479,02", "$34.000" → "34.000".
    Se aplica aunque el LLM haya ignorado la regla del prompt.
    """
    return re.sub(r'\$\s*(\d[\d.,]*)', r'\1', text)


def respond_node(state: AgentState) -> dict:
    response = state.get("final_response") or "¿En qué más puedo ayudarte?"
    response = _sanitize_currency(response)
    return {"step": "fin", "final_response": response}


# ─── Enrutamiento condicional ─────────────────────────────────────────────────

def route_after_agent(state: AgentState) -> Literal[
    "extract_invoice", "identify_bills", "calculate", "respond"
]:
    action = state.get("next_action", "respond")
    if action == "extraer_datos_factura":
        return "extract_invoice"
    elif action == "identificar_billetes":
        return "identify_bills"
    elif action == "calcular_cambio_y_pago":
        return "calculate"
    return "respond"


def route_after_bills(state: AgentState) -> Literal["calculate", "respond"]:
    if state.get("next_action") == "calcular_cambio_y_pago":
        return "calculate"
    return "respond"


# ─── Construcción del grafo ───────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("rag", rag_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("extract_invoice", extract_invoice_node)
    workflow.add_node("identify_bills", identify_bills_node)
    workflow.add_node("calculate", calculate_node)
    workflow.add_node("respond", respond_node)

    workflow.set_entry_point("rag")
    workflow.add_edge("rag", "agent")

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "extract_invoice": "extract_invoice",
            "identify_bills": "identify_bills",
            "calculate": "calculate",
            "respond": "respond",
        },
    )

    workflow.add_conditional_edges(
        "extract_invoice",
        route_after_bills,
        {"calculate": "calculate", "respond": "respond"},
    )
    workflow.add_conditional_edges(
        "identify_bills",
        route_after_bills,
        {"calculate": "calculate", "respond": "respond"},
    )
    workflow.add_edge("calculate", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


graph = build_graph()

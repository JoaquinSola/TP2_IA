from typing import TypedDict, Optional, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Historial de mensajes (LangGraph lo maneja con add_messages)
    messages: Annotated[list[BaseMessage], add_messages]
    # Datos extraídos de herramientas
    invoice_data: Optional[dict]
    bills_data: Optional[list]
    payment_result: Optional[dict]
    # Imágenes en bytes para pasar a los tools
    invoice_image_bytes: Optional[bytes]
    invoice_image_mime: Optional[str]
    bills_image_bytes: Optional[bytes]
    bills_image_mime: Optional[str]
    # Control de flujo
    next_action: Optional[str]  # "extraer_datos_factura" | "identificar_billetes" | "calcular_cambio_y_pago" | "respond"
    session_id: str
    rag_context: str
    # Respuesta final generada
    final_response: Optional[str]
    # Metadata
    step: str

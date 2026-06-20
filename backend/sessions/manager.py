import uuid
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class SessionState:
    session_id: str
    conversation_history: list = field(default_factory=list)
    invoice_data: Optional[dict] = None
    bills_data: Optional[list] = None
    payment_result: Optional[dict] = None
    # Imágenes en bytes + mime type para uso de los tools
    current_invoice_bytes: Optional[bytes] = None
    current_invoice_mime: Optional[str] = None
    current_bills_bytes: Optional[bytes] = None
    current_bills_mime: Optional[str] = None
    # Estado del flujo
    awaiting: Optional[str] = None  # None, "bills_image", "invoice_image", "clarification"


_sessions: dict[str, SessionState] = {}


def get_or_create_session(session_id: Optional[str] = None) -> SessionState:
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    new_id = session_id or str(uuid.uuid4())
    session = SessionState(session_id=new_id)
    _sessions[new_id] = session
    return session


def get_session(session_id: str) -> Optional[SessionState]:
    return _sessions.get(session_id)


def reset_transaction(session: SessionState) -> None:
    """Limpia los datos de la transacción actual pero mantiene el historial."""
    session.invoice_data = None
    session.bills_data = None
    session.payment_result = None
    session.current_invoice_bytes = None
    session.current_invoice_mime = None
    session.current_bills_bytes = None
    session.current_bills_mime = None
    session.awaiting = None

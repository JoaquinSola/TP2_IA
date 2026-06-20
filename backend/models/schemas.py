from pydantic import BaseModel, Field
from typing import Optional, List, Any
from enum import Enum


class BillDenomination(int, Enum):
    B100 = 100
    B200 = 200
    B500 = 500
    B1000 = 1000
    B2000 = 2000
    B5000 = 5000
    B10000 = 10000
    B20000 = 20000
    B50000 = 50000
    B100000 = 100000
    B200000 = 200000
    B500000 = 500000
    B1000000 = 1000000


VALID_DENOMINATIONS = {d.value for d in BillDenomination}


class Bill(BaseModel):
    denomination: int
    position: str = "centro"  # "izquierda", "centro", "derecha", "arriba", "abajo"
    valid: bool = True
    currency: str = "ARS"
    confidence: float = 1.0


class InvoiceData(BaseModel):
    entity: Optional[str] = None
    total_amount: Optional[float] = None
    due_date: Optional[str] = None
    second_due_date: Optional[str] = None
    second_amount: Optional[float] = None
    is_valid_document: bool = True
    error_message: Optional[str] = None


class PaymentResult(BaseModel):
    total_available: float
    total_required: float
    sufficient: bool
    bills_to_use: List[Bill] = Field(default_factory=list)
    change: float = 0.0
    bills_to_keep: List[Bill] = Field(default_factory=list)
    missing_amount: float = 0.0


class ChatRequest(BaseModel):
    message: str = ""
    session_id: Optional[str] = None
    image_base64: Optional[str] = None
    image_mime: Optional[str] = None  # "image/jpeg", "image/png", "application/pdf"
    image_purpose: Optional[str] = None  # "invoice", "bills"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    step: str
    invoice_data: Optional[InvoiceData] = None
    payment_result: Optional[PaymentResult] = None
    error: Optional[str] = None


class LogEvent(BaseModel):
    timestamp: str
    session_id: str
    event_type: str  # "llm_call", "tool_call", "tool_result", "error", "session_start"
    node: Optional[str] = None
    data: Any = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None

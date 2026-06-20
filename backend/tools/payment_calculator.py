"""
Tool: calcular_cambio_y_pago
Lógica DETERMINÍSTICA en Python puro. No usa LLM para evitar alucinaciones aritméticas.
Calcula la combinación óptima de billetes para pagar una factura con mínimo vuelto posible.
"""
import math
from backend.models.schemas import Bill, InvoiceData, PaymentResult
from backend.observability.logger import log_tool_call, log_tool_result, Timer


def calcular_cambio_y_pago(
    session_id: str,
    invoice: InvoiceData,
    bills: list[Bill],
) -> PaymentResult:
    log_tool_call(session_id, "calcular_cambio_y_pago", {
        "invoice_amount": invoice.total_amount,
        "bills_count": len(bills),
    })

    with Timer() as t:
        result = _calculate(invoice, bills)

    log_tool_result(session_id, "calcular_cambio_y_pago", result.model_dump(), t.elapsed_ms)
    return result


def _calculate(invoice: InvoiceData, bills: list[Bill]) -> PaymentResult:
    required = float(invoice.total_amount or 0)
    valid_bills = [b for b in bills if b.valid and b.currency == "ARS"]

    total_available = sum(b.denomination for b in valid_bills)

    if total_available < required:
        return PaymentResult(
            total_available=total_available,
            total_required=required,
            sufficient=False,
            bills_to_use=[],
            change=0.0,
            bills_to_keep=valid_bills,
            missing_amount=round(required - total_available, 2),
        )

    to_use = _find_min_overpayment(valid_bills, required)
    use_ids = {id(b) for b in to_use}
    to_keep = [b for b in valid_bills if id(b) not in use_ids]
    paid = sum(b.denomination for b in to_use)
    change = round(paid - required, 2)

    return PaymentResult(
        total_available=total_available,
        total_required=required,
        sufficient=True,
        bills_to_use=to_use,
        change=change,
        bills_to_keep=to_keep,
        missing_amount=0.0,
    )


def _find_min_overpayment(bills: list[Bill], target: float) -> list[Bill]:
    """
    Encuentra la combinación de billetes que cubre el monto requerido
    con el MÍNIMO vuelto posible.

    Algoritmo: enumeración exhaustiva de todos los 2^n subsets.
    Para n <= 20 (límite real de billetes en billetera) es práctico y exacto.

    Por qué NO greedy: el greedy de "menores primero" puede acumular demasiados
    billetes chicos y perder combinaciones mejores (ej: pagar 6.000 con dos de
    2.000 + uno de 5.000 = 9.000 de vuelto 3.000, cuando uno de 5.000 + uno de
    2.000 = 7.000 de vuelto 1.000).
    """
    subset = bills[:20]
    n = len(subset)
    # Si la factura tiene centavos, cualquier combo de billetes enteros
    # siempre genera algo de vuelto. Usamos ceil para el umbral mínimo de cobertura.
    target_ceil = math.ceil(target)
    denoms = [int(b.denomination) for b in subset]

    best_mask = -1
    best_total = sum(denoms) + 1  # peor caso inicial

    for mask in range(1, 1 << n):
        total = 0
        for i in range(n):
            if mask & (1 << i):
                total += denoms[i]
        if target_ceil <= total < best_total:
            best_total = total
            best_mask = mask
            if total == target_ceil:
                # Mínimo vuelto posible encontrado, no se puede mejorar
                break

    if best_mask == -1:
        return subset  # fallback: no debería ocurrir si total_available >= required

    return [subset[i] for i in range(n) if best_mask & (1 << i)]

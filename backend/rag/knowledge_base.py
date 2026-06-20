"""
Módulo RAG (Retrieval-Augmented Generation).
Utiliza BM25 para recuperar contexto relevante sobre proveedores de servicios
argentinos y billetes, que se inyecta en los prompts de Gemini.
"""
from rank_bm25 import BM25Okapi


_DOCUMENTS = [
    # Proveedores de servicios - Santa Fe
    "EPE Empresa Provincial de la Energía electricidad Santa Fe. La factura muestra CUIL del cliente, período facturado, lectura de medidor en kWh, importe a pagar. Suele tener primer y segundo vencimiento. El logotipo EPE aparece en la parte superior.",
    "ASSA Aguas Santafesinas agua potable Santa Fe. Factura con número de cuenta, consumo en metros cúbicos m3, importe total a pagar, fecha de vencimiento. Logo azul con ondas de agua.",
    "Naturgy Litoral Gas gas natural Santa Fe Entre Ríos. Factura con número de suministro, consumo en metros cúbicos m3, categoría tarifaria, importe total. Varios vencimientos posibles.",
    "Telecom Personal telefonía fija internet banda ancha. Facturas con número de línea, detalle de servicios, importe total del período. Logo rojo y blanco.",
    "Claro telefonía celular móvil datos. Facturas mensuales con detalle de plan, consumos adicionales, importe total. Logo rojo.",
    "Movistar Telefónica telefonía móvil internet servicios. Factura con número de cuenta, plan contratado, importe a pagar.",
    "Municipalidad Santa Fe tasas municipales. Tasa General de Inmuebles TGI, Alumbrado Barrido Limpieza ABL, Derechos de Registro e Inspección DReI. Boleta con CUIT municipal.",
    "DirecTV Telecentro Flow servicio cable televisión por suscripción streaming. Factura con número de abonado, detalle de paquetes, importe mensual.",
    "Expensas consorcio administración edificio propiedad horizontal. Boleta con período, número de unidad, detalle de gastos comunes, importe total.",
    "AFIP ARBA impuestos nacionales provinciales. Boletas de pago con CUIT contribuyente, concepto, período fiscal, importe.",

    # Billetes de Pesos Argentinos en circulación
    "Billete de 1000 pesos argentinos. Color azul. Muestra el quebracho colorado árbol nacional. Denominación en números grandes.",
    "Billete de 2000 pesos argentinos. Color verde oscuro. Muestra el pudú, ciervo más pequeño del mundo nativo de Argentina.",
    "Billete de 5000 pesos argentinos. Color naranja. Muestra la taruca, mamífero andino en peligro de extinción.",
    "Billete de 10000 pesos argentinos. Color violeta púrpura. Muestra el yaguareté, felino emblema de la fauna argentina.",
    "Billete de 20000 pesos argentinos. Color marrón. Muestra el Mburucuyá flor pasionaria emblema floral.",
    "Billete de 50000 pesos argentinos. Color rojo. Muestra la ballena franca austral, mamífero marino.",
    "Billete de 100000 pesos argentinos. Color dorado amarillo. Nuevo billete de alta denominación.",
    "Billete de 200000 pesos argentinos. Color verde claro. Billete de alta denominación en circulación.",
    "Billetes fuera de curso legal Argentina: billetes anteriores a la serie actual, dólares euros moneda extranjera, billetes de juego Monopoly, tarjetas no son billetes.",

    # Información general de facturas argentinas
    "Los datos más importantes de una factura argentina son: razón social empresa emisora, importe total a pagar, fecha de vencimiento primer vencimiento y segundo vencimiento si aplica. El CUIT es el identificador tributario de la empresa.",
    "Una factura puede tener múltiples vencimientos. El primer vencimiento tiene menor importe. El segundo vencimiento tiene recargo por mora o interés punitorio. Si la fecha actual superó el primer vencimiento, se debe pagar el importe del segundo vencimiento.",
    "Los códigos de barras y QR en facturas argentinas permiten el pago electrónico. La información clave está impresa en texto: nombre empresa, importe, fecha vencimiento.",
]

_TOKENIZED = [doc.lower().split() for doc in _DOCUMENTS]
_BM25 = BM25Okapi(_TOKENIZED)


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Recupera los documentos más relevantes para la consulta y los retorna
    como un string de contexto para inyectar en el prompt.
    """
    tokens = query.lower().split()
    scores = _BM25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    relevant = [_DOCUMENTS[i] for i in top_indices if scores[i] > 0]
    if not relevant:
        return ""
    return "\n".join(f"- {doc}" for doc in relevant)

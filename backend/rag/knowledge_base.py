"""
Módulo RAG (Retrieval-Augmented Generation).
Utiliza BM25 para recuperar contexto relevante sobre proveedores de servicios
argentinos y billetes, que se inyecta en los prompts de Llama.
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
    "Billete de 100 pesos argentinos. Colores violeta o marrón. Muestra a Eva Perón, Julio Argentino Roca o la Taruca (taruca, ciervo andino).",
    "Billete de 200 pesos argentinos. Color rosado. Muestra la ballena franca austral o a Martín Miguel de Güemes y Juana Azurduy.",
    "Billete de 500 pesos argentinos. Color verde. Muestra el yaguareté o a Manuel Belgrano y María Remedios del Valle.",
    "Billete de 1000 pesos argentinos. Color naranja o beige. Muestra el hornero o a José de San Martín.",
    "Billete de 2000 pesos argentinos. Color rojo y gris oscuro. Muestra a los médicos Ramón Carrillo y Cecilia Grierson.",
    "Billete de 10000 pesos argentinos. Color celeste y gris azulado. Muestra a Manuel Belgrano y María Remedios del Valle.",
    "Billete de 20000 pesos argentinos. Color azul. Muestra a Juan Bautista Alberdi.",
    "Billetes fuera de curso legal o inválidos en Argentina: dólares, euros, billetes de juego como Monopoly, tarjetas de plástico, monedas y billetes deteriorados ilegibles.",

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

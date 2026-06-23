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
    "Billete de 10 pesos argentinos. Color marrón predominante. Muestra a Manuel Belgrano o un Guanaco.",
    "Billete de 20 pesos argentinos. Color rojo predominante. Muestra a Juan Manuel de Rosas o un Guanaco.",
    "Billete de 50 pesos argentinos. Color azul predominante. Muestra a Domingo F. Sarmiento, Islas Malvinas o Cóndor Andino.",
    "Billete de 100 pesos argentinos. Color violeta predominante. Muestra a Julio A. Roca, Eva Perón o la Taruca.",
    "Billete de 200 pesos argentinos. Color celeste predominante. Muestra la Ballena Franca Austral o a Martín Miguel de Güemes y Juana Azurduy.",
    "Billete de 500 pesos argentinos. Color verde predominante. Muestra el Yaguareté o a María Remedios del Valle y Manuel Belgrano.",
    "Billete de 1000 pesos argentinos. Color naranja predominante. Muestra el Hornero o a José de San Martín.",
    "Billete de 2000 pesos argentinos. Color gris oscuro y rosado. Muestra a la Dra. Cecilia Grierson, Dr. Ramón Carrillo y el Instituto Malbrán.",
    "Billete de 10000 pesos argentinos. Color celeste predominante. Muestra a Manuel Belgrano, María Remedios del Valle y la Jura de la Bandera.",
    "Billete de 20000 pesos argentinos. Color azul y rojo. Muestra a Juan Bautista Alberdi.",
    "Billetes fuera de curso legal en Argentina: billetes anteriores a la serie actual, dólares, euros, moneda extranjera, billetes de juego Monopoly. Las tarjetas de crédito o débito no son billetes argentinos.",

    # Tipos de facturas en Argentina
    "Factura A en Argentina: Emitida por un Responsable Inscripto a otro. Discrimina el IVA en detalle. Montos en pesos argentinos.",
    "Factura B en Argentina: Emitida por un Responsable Inscripto a Consumidor Final, Monotributista o Exento. El IVA está incluido pero no discriminado. Montos en pesos argentinos.",
    "Factura C en Argentina: Emitida por Monotributistas o sujetos exentos. No incluye IVA. Montos en pesos argentinos.",
    "Factura E en Argentina: Factura de Exportación de bienes o servicios. Puede estar en pesos argentinos o moneda extranjera.",
    "Factura M en Argentina: Emitida por Responsables Inscriptos con inconsistencias. Discrimina IVA e incluye retenciones en pesos argentinos.",

    # Información general de facturas argentinas
    "Los datos más importantes de una factura argentina son: razón social empresa emisora, importe total a pagar en pesos argentinos, fecha de vencimiento primer vencimiento y segundo vencimiento si aplica. El CUIT es el identificador tributario de la empresa en Argentina.",
    "Una factura en Argentina puede tener múltiples vencimientos. El primer vencimiento tiene menor importe. El segundo vencimiento tiene recargo por mora o interés punitorio en pesos argentinos. Si la fecha actual superó el primer vencimiento, se debe pagar el importe del segundo vencimiento.",
    "Los códigos de barras y QR en facturas argentinas permiten el pago electrónico en pesos argentinos. La información clave está impresa en texto: nombre de la empresa, importe a pagar en pesos argentinos, fecha de vencimiento.",

    # Métodos de Pago y Billeteras Virtuales en Argentina
    "El CBU (Clave Bancaria Uniforme) tiene 22 dígitos y se usa para cuentas bancarias tradicionales en Argentina. El CVU (Clave Virtual Uniforme) también tiene 22 dígitos pero se usa para billeteras virtuales no bancarias. Ambos pueden tener un Alias de texto.",
    "Billeteras virtuales y pagos QR en Argentina: Aplicaciones como Mercado Pago, MODO, Ualá y Personal Pay permiten pagos a través de Códigos QR interoperables bajo el sistema de Transferencias 3.0. Los montos se debitan en pesos argentinos.",
    "Tarjetas en Argentina: Las tarjetas de débito extraen fondos inmediatos de la cuenta, mientras que las de crédito permiten pagar a mes vencido o en cuotas financiadas en pesos argentinos (ej. programas Cuota Simple o Ahora 12).",

    # Notas de Crédito, Débito y Recibos en Argentina
    "Nota de Crédito en Argentina: Es un documento legal que anula o revierte total o parcialmente una factura emitida previamente, por ejemplo, debido a una devolución, descuento o error de facturación. Los montos se restan.",
    "Nota de Débito en Argentina: Es un comprobante que las empresas emiten para cobrar un recargo a una factura ya emitida. Se usa frecuentemente para cobrar intereses por pagos fuera de término o gastos bancarios. Los montos se suman.",
    "Remitos y Recibos en Argentina: El Remito comercial no tiene validez fiscal como factura, solo respalda el traslado y entrega física de mercadería. El Recibo (ej. Recibo X o C) es el comprobante oficial de que se ha recibido un pago en dinero.",

    # Impuestos y Retenciones (Nivel Básico)
    "El IVA (Impuesto al Valor Agregado) en Argentina tiene tres alícuotas principales: 21% (tasa general), 10,5% (bienes de capital, algunos alimentos, computación) y 27% (servicios públicos como luz, gas y telecomunicaciones).",
    "Ingresos Brutos (IIBB) y Percepciones: En Argentina, los impuestos provinciales como IIBB (ej. ARBA en Buenos Aires o API en Santa Fe) pueden generar cargos extra llamados 'Percepciones' que se suman al total a pagar en la factura.",

    # Tipos de Cambio y Dólar
    "Facturas en moneda extranjera en Argentina: Las facturas de exportación (Tipo E) o servicios de software del exterior suelen estar denominadas en dólares. Al pagarse localmente, a menudo se convierten a pesos argentinos usando la cotización del Dólar Oficial del Banco Nación, pudiendo sumar percepciones e Impuesto PAIS si corresponde.",

    # Servicios Públicos y Avisos de Corte
    "Aviso de Suspensión o Corte de Servicio en Argentina: Las empresas de servicios públicos (ej. EPE, ASSA, Litoral Gas) imprimen leyendas de advertencia de corte si se registran deudas previas o facturas impagas. Generalmente se indica una fecha límite antes de proceder con el corte del suministro.",
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

"""
RAG con embeddings vectoriales usando ChromaDB y sentence-transformers.
La colección persiste en backend/rag/chroma_db/.
Al primer arranque con colección vacía, indexa automáticamente los documentos semilla.
"""
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

_CHROMA_PATH = Path(__file__).parent / "chroma_db"
_COLLECTION_NAME = "knowledge"
_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_MAX_DISTANCE = 0.85  # cosine distance: 0=idéntico, 2=opuesto; >0.85 = irrelevante

_SEED_DOCUMENTS = [
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
    "Billete de 100 pesos argentinos. Colores violeta, marrón o gris. Existen tres diseños legales activos: 1) Julio Argentino Roca (marrón/gris, reverso 'La Conquista del Desierto'); 2) Eva Perón (violeta, reverso un detalle del altar del Monumento a los Caídos); 3) Taruca o ciervo andino (violeta, diseño vertical de la familia Animales Autóctonos).",
    "Billete de 200 pesos argentinos. Colores rosado o azul grisáceo. Existen dos diseños legales activos: 1) Ballena Franca Austral (rosado, diseño vertical de la familia Animales Autóctonos); 2) Martín Miguel de Güemes y Juana Azurduy (azul grisáceo con tonos rosados, familia Heroínas y Héroes de la Patria).",
    "Billete de 500 pesos argentinos. Color verde. Existen dos diseños legales activos: 1) Yaguareté (diseño vertical de la familia Animales Autóctonos); 2) Manuel Belgrano y María Remedios del Valle (familia Heroínas y Héroes de la Patria, reverso con la recreación del Juramento de la Bandera).",
    "Billete de 1000 pesos argentinos. Colores naranja o marrón claro. Existen dos diseños legales activos: 1) Hornero (naranja, diseño vertical de la familia Animales Autóctonos); 2) José de San Martín (marrón claro y naranja, familia Heroínas y Héroes de la Patria, reverso con el Cruce de los Andes).",
    "Billete de 2000 pesos argentinos. Colores gris oscuro, rojo y rosado. Muestra en el anverso los retratos de los médicos precursores de la medicina argentina Cecilia Grierson y Ramón Carrillo. El reverso muestra la fachada del Instituto Nacional de Microbiología Dr. Carlos G. Malbrán.",
    "Billete de 10000 pesos argentinos. Colores celeste y azul. Muestra en el anverso los retratos de Manuel Belgrano y de María Remedios del Valle (nombrada Capitana del Ejército del Norte). El reverso muestra la recreación artística de la Jura de la Bandera del 27 de febrero de 1812.",
    "Billete de 20000 pesos argentinos. Color azul predominante. Muestra en el anverso el retrato de Juan Bautista Alberdi (inspirador de la Constitución Nacional de 1853). El reverso ilustra la recreación de la casa natal del prócer.",
    "Billetes fuera de curso legal o inválidos en Argentina: billetes de 2 y 5 pesos argentinos (ya desmonetizados), divisas extranjeras (dólares, euros), billetes de fantasía o juegos, tarjetas plásticas, monedas (son de curso legal pero no son billetes) y billetes cuya superficie esté deteriorada o fragmentada en más del 40% sin numeración legible.",
    # Información general de facturas argentinas
    "Los datos más importantes de una factura argentina son: razón social empresa emisora, importe total a pagar, fecha de vencimiento primer vencimiento y segundo vencimiento si aplica. El CUIT es el identificador tributario de la empresa.",
    "Una factura puede tener múltiples vencimientos. El primer vencimiento tiene menor importe. El segundo vencimiento tiene recargo por mora o interés punitorio. Si la fecha actual superó el primer vencimiento, se debe pagar el importe del segundo vencimiento.",
    "Los códigos de barras y QR en facturas argentinas permiten el pago electrónico. La información clave está impresa en texto: nombre empresa, importe, fecha vencimiento.",
    # Tickets y comprobantes de pago en Argentina
    "Ticket de caja de supermercado o comercio en Argentina (Carrefour, Walmart, La Anonima, Disco, Vea, Coto, Jumbo, DIA, Farmacity, farmacia, ferretería, librería, kiosco). Encabezado con razón social del comercio y CUIT. Lista de ítems con precios unitarios. Subtotal, descuentos, y TOTAL a pagar en pesos argentinos. Número de comprobante formato XXXX-XXXXXXXX (punto de venta guion número). Fecha y hora de emisión. NO tiene fecha de vencimiento: el pago es inmediato en caja.",
    "Ticket fiscal argentino emitido por controladora fiscal (Hasar, Epson, Bixolon). Leyenda 'Controlador Fiscal' o 'CF' o 'Comprobante no válido como factura'. Encabezado con razón social, domicilio, CUIT, Ingresos Brutos. Detalle de ítems. Totales: subtotal neto, IVA 21%, IVA 10.5%, TOTAL. Pago inmediato, sin fecha de vencimiento.",
    "Factura electrónica argentina tipo A, B o C con Código de Autorización Electrónico CAE y QR de verificación AFIP. Tipo A: entre responsables inscriptos en IVA (tiene IVA discriminado). Tipo B: a consumidores finales. Tipo C: emitida por monotributistas (sin IVA discriminado). Punto de venta y número de comprobante. La fecha del CAE NO es fecha de vencimiento del pago.",
    "Ticket o comprobante de servicio técnico, plomería, electricista, gasista, pintor, carpintero o reparación doméstica en Argentina. Puede ser manuscrito o impreso. Datos del prestador: nombre o razón social, CUIT o DNI, teléfono. Descripción breve del trabajo realizado. Monto total a pagar en pesos. Sin fecha de vencimiento formal, pago al momento del servicio.",
    "Boleta o cuota de colegio, jardín de infantes, universidad, club deportivo, gimnasio o asociación civil en Argentina. Nombre de la institución, nombre del alumno o socio, período o cuota número, importe total en pesos, fecha límite o vencimiento de pago. Puede tener descuento por pago anticipado.",
    "Tipos de comprobantes válidos de pago en Argentina: facturas de servicios públicos (luz, gas, agua, teléfono, internet), tickets de supermercado, tickets fiscales, facturas electrónicas tipo A/B/C, boletas de colegios y clubes, recibos de alquiler, comprobantes de peaje y estacionamiento. Todos son documentos legítimos con monto a pagar.",
    "Diferencias entre ticket y factura en Argentina. Ticket de supermercado: papel térmico angosto, listado de productos, TOTAL al final, sin fecha de vencimiento. Factura de servicio: papel A4 o carta, logo empresa, período facturado, primer y segundo vencimiento con fechas. Ambos tienen CUIT del emisor y monto total. El bot puede leer ambos tipos.",
]

_collection = None


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    ef = SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)
    _collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    _seed_if_missing(_collection)

    return _collection


def _seed_if_missing(collection):
    ids = [hashlib.md5(doc.encode()).hexdigest() for doc in _SEED_DOCUMENTS]
    existing = collection.get(ids=ids)
    missing_ids = set(ids) - set(existing["ids"])
    if not missing_ids:
        return
    pairs = [(i, doc) for i, doc in zip(ids, _SEED_DOCUMENTS) if i in missing_ids]
    collection.add(
        documents=[doc for _, doc in pairs],
        ids=[i for i, _ in pairs],
        metadatas=[{"source": "seed", "type": "text"}] * len(pairs),
    )


def retrieve_context(query: str, top_k: int = 3) -> str:
    if not query.strip():
        return ""

    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return ""

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, total),
        include=["documents", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    relevant = [doc for doc, dist in zip(docs, distances) if dist < _MAX_DISTANCE]

    if not relevant:
        return ""
    return "\n".join(f"- {doc}" for doc in relevant)

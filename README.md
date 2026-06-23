# Agente IA – Asistencia Visual para Pago de Facturas
**UTN Santa Fe – Inteligencia Artificial 2026 – TP2 – Grupo 7**

Valentina Ducasse · Wenceslao Echevarría · Nicolás García · Joaquín Sola · Nicolás Springer

---

## Descripción

Agente inteligente que asiste a personas con discapacidad visual en la gestión y pago de facturas de servicios. Procesa imágenes de facturas físicas y digitales (PDF), identifica billetes en efectivo y calcula la combinación óptima de pago.

**Stack tecnológico:**
- LLM: Llama 3.3 y Llama 4 Scout (multimodal) — Groq
- Orquestador: LangGraph (grafo de estados)
- Backend: FastAPI (Python 3.10+)
- RAG: ChromaDB + sentence-transformers (embeddings vectoriales) sobre documentos de proveedores y billetes argentinos
- Frontend: Vanilla JS / HTML / CSS — mobile-first con Web Speech API

---

## Requisitos

- Python 3.10 o superior
- API Key de Groq (obtener en [Groq Console](https://console.groq.com/keys))

---

## Instalación

```bash
# 1. Clonar o descargar el proyecto
cd TP2

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
copy .env.example .env
# Editar .env y completar GROQ_API_KEY con tu clave
```

---

## Ejecución

```bash
# Desde la raíz del proyecto (con el entorno virtual activado)
python -m uvicorn backend.main:app --reload --port 8000
```

Luego abrir en el navegador: **http://localhost:8000**

Para acceso desde el celular (misma red WiFi):
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Conectar desde el celular a: `http://<IP-de-tu-PC>:8000`

---

## Arquitectura del agente

```
Usuario (voz/texto/imagen)
        ↓
   [Frontend HTML/JS]
        ↓  HTTP POST /api/chat
   [FastAPI Backend]
        ↓
   [LangGraph Graph]
    ├── rag_node        → Recupera contexto (ChromaDB vectorial sobre proveedores AR)
    ├── agent_node      → LLM (Llama) decide qué tool invocar
    ├── extract_invoice → Tool: extraer_datos_factura (Llama Vision multimodal)
    ├── identify_bills  → Tool: identificar_billetes (Llama Vision multimodal)
    ├── calculate       → Tool: calcular_cambio_y_pago (Python determinístico)
    └── respond         → Respuesta final al usuario
        ↓
   [Logging JSON]      → logs/agent_YYYYMMDD_HHMMSS.jsonl
```

## Tools implementadas

| Tool | Tipo | Descripción |
|------|------|-------------|
| `extraer_datos_factura` | LLM (Llama) | Extrae entidad, monto y vencimiento de factura |
| `identificar_billetes` | LLM (Llama) | Identifica denominación y posición de cada billete |
| `calcular_cambio_y_pago` | Determinístico (Python) | Calcula combinación óptima de pago y vuelto exacto |

## Endpoints API

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/chat` | POST | Interacción principal con el agente |
| `/api/upload` | POST | Subida de archivos (imagen/PDF) |
| `/api/reset` | POST | Reiniciar transacción actual |
| `/api/logs/{session_id}` | GET | Logs de observabilidad de la sesión |
| `/api/health` | GET | Estado del servicio |

---

## RAG — Gestión de la base de conocimiento

La base de conocimiento vectorial persiste en `backend/rag/chroma_db/` y tiene dos tipos de documentos:

- **Seeds:** textos generales hardcodeados en `backend/rag/knowledge_base.py` (proveedores argentinos, billetes, info de facturas).
- **Documentos propios:** archivos indexados manualmente desde la carpeta `documentos/` (facturas reales, fotos de billetes, etc.).

### Agregar nuevos documentos

Copiá los archivos a la carpeta `documentos/` (soporta `.txt`, `.jpg`, `.jpeg`, `.png`, `.webp`, `.pdf`) y ejecutá el indexer:

```bash
python -m backend.rag.indexer documentos/
```

Los archivos ya indexados se saltean automáticamente. Podés agregar documentos en cualquier momento sin reiniciar el servidor.

### Modificar los seeds

Editá los textos en `_SEED_DOCUMENTS` dentro de `backend/rag/knowledge_base.py` y luego ejecutá:

```bash
python -m backend.rag.reseed
```

Esto elimina los seeds viejos de la colección y carga los nuevos. No afecta los documentos propios indexados.

---

## Casos de prueba

Ver `Banco de Casos de Prueba` en el informe técnico (10 casos: happy path, límite, adversariales).

---

## Observabilidad

Los logs se guardan en `logs/agent_*.jsonl` con formato JSONL. Cada entrada registra:
- Llamadas al LLM: prompt resumen, respuesta, latencia, tokens
- Invocaciones a tools: entrada y salida
- Errores: nodo y descripción

También disponible en la UI: panel "Ver trazas del agente" en la parte inferior.

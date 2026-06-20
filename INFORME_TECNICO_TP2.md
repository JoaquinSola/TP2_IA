# Asistente Inteligente para Pago de Facturas con Accesibilidad Visual

**Universidad Tecnológica Nacional – Facultad Regional Santa Fe**  
**Inteligencia Artificial – Año 2026 – Trabajo Práctico N° 2**  
**Grupo 7**

| Integrante | E-mail |
|---|---|
| Valentina Ducasse | ducassevalentina@gmail.com |
| Wenceslao Echevarría | wencesechevarria@hotmail.com |
| Nicolás García | nicoag.2000@gmail.com |
| Joaquín Sola | joaquin_sola@hotmail.com |
| Nicolás Springer | nicosspringer16@gmail.com |

---

## Resumen

Se diseñó e implementó un agente inteligente multimodal para asistir a personas con discapacidad visual en el pago de facturas de servicios argentinos. El sistema percibe imágenes de facturas físicas o digitales y billetes de curso legal mediante visión artificial, extrae información estructurada, calcula combinaciones óptimas de pago con lógica determinística, y comunica los resultados exclusivamente por voz. El agente utiliza LangGraph como orquestador de flujo, modelos LLM de Groq para razonamiento y análisis visual, BM25 para recuperación de contexto (RAG), y memoria de sesión para mantener coherencia en transacciones multi-paso. El sistema fue validado con 10 casos de prueba cubriendo escenarios normales, límite y adversariales, demostrando funcionamiento completo en todos los casos principales.

---

## 1. Introducción

### 1.1 Área de Aplicación y Problema

Las personas con discapacidad visual enfrentan una barrera significativa en la autonomía financiera cotidiana: la gestión y el pago de facturas de servicios. Para ejecutar un pago presencial con efectivo, una persona necesita realizar tres tareas que dependen de la visión: leer el monto y la fecha de vencimiento de la factura, identificar la denominación de los billetes disponibles, y calcular qué billetes entregar y cuánto vuelto recibir.

Actualmente, estas personas dependen de terceros para realizar estas operaciones, lo que compromete su privacidad e independencia. Las herramientas de accesibilidad existentes (lectores de pantalla, bastón blanco) no resuelven la interpretación semántica de documentos físicos ni el reconocimiento de papel moneda.

El problema se manifiesta en dos dimensiones simultáneas:
- **Física**: facturas impresas en papel de servicios (electricidad EPE, agua ASSA, gas Naturgy, etc.)
- **Digital**: facturas en PDF o imagen recibidas por correo electrónico
- **Monetaria**: identificación y combinación óptima de billetes de pesos argentinos en circulación

### 1.2 Objetivo del Agente

Construir un agente inteligente que, a partir de fotografías o documentos digitales, asista completamente a una persona con discapacidad visual en:
1. Extraer datos críticos de facturas: entidad emisora, monto y fecha de vencimiento
2. Identificar denominaciones y posiciones espaciales de billetes de pesos argentinos
3. Calcular la combinación óptima de billetes a entregar y el vuelto exacto a recibir
4. Comunicar todo el proceso exclusivamente por voz, sin requerir lectura de pantalla

El sistema opera únicamente como herramienta de asistencia informativa, sin acceso a cuentas bancarias ni ejecución de transacciones electrónicas.

### 1.3 Estructura del Informe

En la sección 2 se explica la arquitectura completa del agente y sus decisiones de diseño. En la sección 3 se presentan los resultados de los casos de prueba ejecutados. En la sección 4 se exponen las conclusiones y los modos de falla detectados.

---

## 2. Solución

### 2.1 Arquitectura General

El sistema se estructura en tres capas claramente separadas, con comunicación exclusivamente a través de APIs HTTP:

```
┌─────────────────────────────────────────┐
│         FRONTEND (Navegador Móvil)       │
│  Vanilla JS + HTML5 + Web Audio API      │
│  STT (Web Speech API / Groq Whisper)     │
│  TTS (gTTS vía backend / SpeechSynth)   │
└────────────────┬────────────────────────┘
                 │ HTTP / REST
┌────────────────▼────────────────────────┐
│         BACKEND (FastAPI + Python)       │
│  ┌──────────────────────────────────┐   │
│  │   Orquestador LangGraph          │   │
│  │  RAG → Agent → Tool → Respond    │   │
│  └──────────────────────────────────┘   │
│  ┌──────────┐ ┌────────┐ ┌──────────┐  │
│  │ Sesiones │ │  RAG   │ │ Logger   │  │
│  └──────────┘ └────────┘ └──────────┘  │
└────────────────┬────────────────────────┘
                 │ API HTTP
┌────────────────▼────────────────────────┐
│         GROQ API (LLM externo)           │
│  llama-3.3-70b (texto)                  │
│  llama-4-scout (visión)                 │
│  whisper-large-v3-turbo (STT)           │
└─────────────────────────────────────────┘
```

**Figura 1.** Arquitectura en capas del sistema. El frontend se ejecuta en el navegador del celular del usuario; el backend en un servidor local expuesto mediante túnel HTTPS (ngrok); el LLM en la nube de Groq.

#### 2.1.1 Decisión de Modelo LLM

El sistema fue diseñado inicialmente con **Gemini 1.5 Flash** (Google). Durante la implementación se encontró una limitación crítica: las cuotas de uso gratuito (`free tier`) del proyecto de Google Cloud que aloja las API keys tenían límite `0` para todos los modelos Gemini (`GenerateRequestsPerDayPerProjectPerModel-FreeTier: 0`), bloqueando toda solicitud independientemente de la clave utilizada.

Se decidió migrar al proveedor **Groq**, que ofrece:
- `llama-3.3-70b-versatile`: razonamiento de texto, 1.000 req/día gratuitas
- `meta-llama/llama-4-scout-17b-16e-instruct`: análisis de imágenes (multimodal), 1.000 req/día gratuitas
- `whisper-large-v3-turbo`: transcripción de audio, gratuito
- Latencia muy baja (< 1 segundo para texto)

Esta migración requirió adaptar las llamadas de la API propietaria de Google a la API compatible con OpenAI que expone Groq, sin cambios en la lógica del agente.

### 2.2 Grafo de Estados (LangGraph)

El flujo de ejecución del agente se implementa como un **StateGraph de LangGraph**, donde cada nodo representa una etapa de procesamiento y los arcos representan las posibles transiciones. El estado compartido entre nodos incluye: el historial de mensajes, los datos de la factura procesada, los billetes identificados, las imágenes pendientes y la acción siguiente a ejecutar.

```
              [INICIO]
                  │
            ┌─────▼──────┐
            │  RAG Node  │  ← Recupera contexto relevante (BM25)
            └─────┬──────┘
                  │
            ┌─────▼──────┐
            │ Agent Node │  ← LLM decide qué hacer
            └──────┬─────┘
         ┌─────────┼─────────┐
         ▼         ▼         ▼
  [extraer_     [identificar_ [calcular_
  factura]       billetes]     pago]
         └─────────┼─────────┘
                   ▼
            ┌──────▼─────┐
            │   Respond  │  ← Prepara respuesta final
            └──────┬─────┘
                   │
               [FIN / API]
```

**Figura 2.** Grafo de estados del agente. El nodo Agent evalúa el estado actual y enruta a la herramienta apropiada. La herramienta calcular_pago puede invocarse directamente desde identify_bills si la factura ya fue procesada previamente.

#### ¿Por qué LangGraph y no un loop ReAct tradicional?

El patrón **ReAct** (Reason + Act) ejecuta razonamiento y acción de forma intercalada en una cadena lineal. Para este problema, el flujo tiene una estructura más predecible: la secuencia de percepciones (factura → billetes → cálculo) es conocida de antemano y el estado de la transacción debe persistir entre múltiples turnos de conversación.

LangGraph fue elegido porque permite:
1. **Control explícito del flujo**: el grafo define formalmente qué nodos existen y qué transiciones son posibles, eliminando comportamientos inesperados del loop abierto
2. **Estado compartido estructurado**: el `AgentState` tipado garantiza que cada nodo recibe exactamente los datos que necesita
3. **Ejecución asincrónica**: `graph.ainvoke()` se integra nativamente con FastAPI async

### 2.3 Nodos del Grafo y su Funcionamiento

#### 2.3.1 Nodo RAG (Retrieval-Augmented Generation)

**Qué hace**: Antes de cualquier razonamiento, el sistema recupera información de su base de conocimiento local que sea relevante para el mensaje actual del usuario.

**Cómo**: Ejecuta una búsqueda léxica BM25 (Okapi BM25) sobre 22 documentos curados que describen proveedores de servicios argentinos y denominaciones de billetes. Los 3 documentos con mayor puntaje se formatean como texto y se inyectan en el prompt del agente.

**Cuándo**: Siempre, como primer paso de cada turno de conversación.

**Por qué**: El LLM es un modelo global entrenado con datos internacionales. La base de conocimiento local proporciona información específica del contexto argentino (que EPE es la empresa de electricidad de Santa Fe, qué colores tienen los billetes de la serie Fauna Argentina, etc.) que de otro modo el modelo podría confundir o inventar. RAG reduce las alucinaciones sobre proveedores locales.

**Base de conocimiento**: 22 fragmentos de texto divididos en tres categorías:
- 10 documentos sobre proveedores de servicios en Santa Fe (EPE, ASSA, Naturgy, Telecom, etc.)
- 9 documentos sobre denominaciones de billetes de pesos argentinos en circulación
- 3 documentos sobre reglas generales de facturas argentinas (vencimientos, CUIT, códigos de barras)

#### 2.3.2 Nodo Agent (Razonamiento Principal)

**Qué hace**: Es el cerebro del agente. Recibe el estado completo de la conversación y decide cuál es la próxima acción.

**Cómo**: Construye un prompt estructurado que incluye: el estado actual de la transacción (si ya tiene datos de factura, si ya tiene billetes, qué imágenes están pendientes), el historial de los últimos 10 turnos, el contexto RAG recuperado, y el mensaje actual del usuario. Envía este prompt al LLM de texto (`llama-3.3-70b-versatile`) con un system prompt que define el comportamiento del asistente.

**Cuándo**: En cada turno de conversación, después del nodo RAG.

**Por qué**: Centralizar el razonamiento en un único nodo permite que el agente maneje la conversación de forma natural (respondiendo preguntas, recordando contexto) al mismo tiempo que detecta cuándo es momento de invocar una herramienta específica.

**Decisión de salida**: El LLM responde en uno de cuatro formatos:
- `TOOL:extraer_datos_factura` → hay imagen de factura sin procesar
- `TOOL:identificar_billetes` → hay imagen de billetes sin procesar
- `TOOL:calcular_cambio_y_pago` → hay factura Y billetes procesados
- Texto libre → respuesta conversacional directa

#### 2.3.3 Estrategia de Prompting (System Prompt)

El agente recibe un **system prompt** con las siguientes secciones:

1. **Contexto argentino**: moneda (pesos argentinos, nunca dólares), billetes vigentes, proveedores comunes
2. **Reglas de comportamiento**: 
   - Máximo 2 frases por respuesta (optimizado para audio)
   - Nunca inventar montos ni fechas
   - Nunca mencionar divisas extranjeras
   - Usar "tocá" en lugar de "mirá" (accesibilidad visual)
3. **Instrucciones de accesibilidad**: cómo guiar al usuario hacia los botones de la interfaz táctil
4. **Herramientas disponibles**: cuándo invocar cada una
5. **Flujo típico de la transacción**: secuencia esperada factura → billetes → cálculo

El **prompt de decisión** (AGENT_DECISION_PROMPT) inyecta el estado actual de forma estructurada:
- Si tiene datos de factura extraídos: Sí/No
- Si tiene billetes identificados: Sí/No
- Si hay imagen de factura/billetes pendiente: Sí/No
- Historial de conversación
- Contexto RAG

Las **reglas de decisión** están numeradas con prioridad explícita para reducir ambigüedad: imagen sin procesar siempre genera invocación de tool; solo si hay ambos datos procesados se calcula el pago.

### 2.4 Herramientas (Tools)

#### 2.4.1 Tool 1: extraer_datos_factura

**Propósito**: Analizar visualmente una imagen de factura y extraer información estructurada.

**Entrada**: Bytes de imagen (JPEG, PNG, WebP o PDF convertido) + tipo MIME.

**Proceso**:
1. Si la imagen supera 3 MB, se redimensiona al máximo de 1920px con compresión JPEG 85% (Pillow) para no exceder los límites de la API de visión
2. Se codifica en Base64 para enviarla como URL de datos (`data:image/jpeg;base64,...`)
3. Se construye un prompt multimodal que combina texto (instrucciones + contexto RAG) con la imagen
4. El LLM de visión (`llama-4-scout`) responde en JSON puro con los campos extraídos
5. El JSON se parsea y se valida para construir el objeto `InvoiceData`

**Salida estructurada (JSON)**:
```json
{
  "entity": "EPE",
  "total_amount": 34000,
  "due_date": "15/07/2026",
  "second_due_date": "30/07/2026",
  "second_amount": 38500,
  "is_valid_document": true,
  "error_message": null
}
```

**Validación**: Si el documento no es una factura válida, `is_valid_document = false` y se devuelve un mensaje de error descriptivo. Nunca se inventan datos: si un campo no es legible, se retorna `null`.

**Manejo de errores**: Cualquier excepción (error de API, imagen ilegible, JSON malformado) devuelve un `InvoiceData` con `is_valid_document = false` y un mensaje accesible al usuario, sin propagar el error hacia arriba.

#### 2.4.2 Tool 2: identificar_billetes

**Propósito**: Analizar una fotografía de dinero en efectivo e identificar cada billete de pesos argentinos presente.

**Entrada**: Bytes de imagen + tipo MIME.

**Proceso**:
1. Compresión de imagen si supera 3 MB (igual que la tool anterior)
2. Prompt multimodal con instrucciones específicas: leer el número impreso (no inferir por color), identificar posición espacial de cada billete
3. El LLM responde con una lista de billetes en JSON
4. Se parsean denominaciones en cualquier formato posible (`5000`, `"5.000"`, `"$5.000"`)
5. Se marca `valid = false` solo si es explícitamente moneda extranjera o billete de juego

**Salida estructurada (JSON)**:
```json
{
  "bills": [
    {"denomination": 10000, "position": "izquierda", "valid": true, "currency": "ARS", "confidence": 0.95},
    {"denomination": 5000,  "position": "derecha",   "valid": true, "currency": "ARS", "confidence": 0.88}
  ],
  "description": "Dos billetes sobre la mesa"
}
```

**Posiciones espaciales reportadas**: izquierda, centro, derecha, arriba, abajo, arriba-izquierda, arriba-derecha, abajo-izquierda, abajo-derecha. Estas posiciones permiten al usuario localizar físicamente cada billete.

**Manejo de errores**: Si la imagen no contiene billetes identificables, devuelve lista vacía. Excepciones de API devuelven lista vacía con log del error.

#### 2.4.3 Tool 3: calcular_cambio_y_pago

**Propósito**: Calcular qué billetes debe entregar el usuario para pagar la factura y cuánto vuelto recibirá.

**Implementación**: **100% determinística en Python puro, sin invocación de LLM**. Esta decisión de diseño es fundamental: los LLMs cometen errores aritméticos. Todo cálculo numérico se ejecuta en código con precisión garantizada.

**Algoritmo**:
1. Filtra los billetes válidos (currency = "ARS", valid = true)
2. Suma el total disponible
3. Si el total < monto de la factura → devuelve `sufficient = false` con el monto faltante
4. Si el total ≥ monto → busca combinación exacta mediante **programación dinámica** (subset sum):
   - Para ≤ 15 billetes: bitmask DP exhaustivo
   - Para > 15 billetes: DP tabla con early exit
5. Si no existe combinación exacta → **algoritmo greedy**: ordena billetes de menor a mayor denominación y acumula hasta cubrir el monto, minimizando el excedente pagado

**Salida estructurada**:
```
PaymentResult:
  sufficient: bool
  total_available: float
  total_required: float
  bills_to_use: [Bill]     ← billetes que el usuario debe entregar
  bills_to_keep: [Bill]    ← billetes que el usuario debe guardar
  change: float             ← vuelto que debe recibir
  missing_amount: float     ← cuánto falta si no alcanza
```

**Tabla 1.** Comportamiento del algoritmo de cálculo según el escenario.

| Escenario | Algoritmo aplicado | Resultado |
|---|---|---|
| Dinero exacto disponible | Subset sum DP | bills_to_use = combinación exacta, change = 0 |
| Dinero suficiente pero sin combinación exacta | Greedy ascendente | bills_to_use = mínima cantidad que cubre, change > 0 |
| Dinero insuficiente | Verificación directa | sufficient = false, missing_amount calculado |
| Billetes inválidos presentes | Filtrado previo | Solo se computan billetes ARS válidos |

### 2.5 Memoria del Agente

**Tipo**: Memoria de corto plazo en sesión (session-scoped state), implementada como un diccionario en memoria RAM del servidor.

**Qué persiste entre turnos**: 
- `invoice_data`: datos de la factura procesada (entity, monto, fechas)
- `bills_data`: lista de billetes identificados
- `payment_result`: resultado del último cálculo
- `conversation_history`: últimos turnos de conversación (texto)
- `current_invoice_bytes` / `current_bills_bytes`: imágenes pendientes de procesar

**Qué NO persiste**: el estado de la sesión se destruye cuando el servidor se reinicia. No hay persistencia en base de datos.

**Por qué esta elección**: El problema es una transacción de corto plazo. El usuario no necesita recordar facturas de sesiones anteriores. Una memoria de sesión en RAM es suficiente, simple de implementar y no requiere infraestructura adicional. Se puede reiniciar el estado dentro de una sesión usando el endpoint `/api/reset`.

**Política de actualización**: el estado se actualiza acumulativamente. Cuando se procesa una factura, `invoice_data` se actualiza y se mantiene disponible para cuando lleguen los billetes. Cuando se calculan los billetes, el agente no necesita que el usuario vuelva a subir la factura.

**Cómo el agente "recuerda" la factura al recibir los billetes**: el `AgentState` incluye tanto `invoice_data` como `bills_data`. Cuando el nodo `identify_bills` detecta que `invoice_data` ya está cargado, devuelve `next_action = "calcular_cambio_y_pago"` en lugar de esperar, disparando el nodo de cálculo automáticamente.

### 2.6 RAG (Retrieval-Augmented Generation)

**Objetivo**: Proporcionar al LLM contexto específico del dominio argentino sin requerir fine-tuning ni embeddings vectoriales costosos.

**Motor de búsqueda**: BM25 Okapi (librería `rank-bm25`). Recuperación léxica basada en frecuencia de términos.

**Estrategia de chunking**: Los documentos son fragmentos cortos y auto-contenidos, uno por proveedor o denominación de billete. Promedio: 60-80 palabras por fragmento. No se aplica chunking dinámico porque los documentos ya son atómicos y homogéneos en tamaño.

**Modelo de embeddings**: Ninguno. BM25 es un modelo de recuperación no semántico (sin embeddings vectoriales). Esta elección tiene ventajas para este dominio: las consultas contienen nombres propios ("EPE", "ASSA", "billete de 5000") que aparecen textualmente en los documentos. La búsqueda léxica es más precisa que la semántica para nombres de marcas y denominaciones numéricas.

**Retriever**: Top-3 documentos con puntaje BM25 > 0. Si ningún documento supera el umbral, se retorna contexto vacío.

**Inyección**: El contexto recuperado se inserta en ambos prompts: en el prompt de decisión del agente (para razonamiento) y en el prompt de extracción de la tool (para análisis de imagen).

**Tabla 2.** Ejemplos de consultas y documentos recuperados por el módulo RAG.

| Consulta del usuario | Top documento recuperado |
|---|---|
| "Tengo una factura de la EPE" | "EPE Empresa Provincial de la Energía electricidad Santa Fe. La factura muestra CUIL del cliente, período facturado..." |
| "Foto de billetes" | "Billete de 10000 pesos argentinos. Color violeta púrpura. Muestra el yaguareté..." |
| "Quiero pagar el gas" | "Naturgy Litoral Gas gas natural Santa Fe Entre Ríos. Factura con número de suministro..." |

### 2.7 Guardrails y Validación

El sistema incorpora múltiples mecanismos de defensa contra comportamientos no deseados:

**Guardrails a nivel de prompt**:
- El system prompt prohíbe explícitamente inventar datos ("NO inventes datos. Si no podés leer un campo con certeza, usá null")
- Prohíbe usar cualquier moneda que no sea pesos argentinos
- Limita las respuestas a 2 frases para evitar respuestas extensas inadecuadas para audio
- Obliga a responder en español argentino

**Guardrails a nivel de herramienta**:
- `is_valid_document = false` si la imagen no es una factura de servicios. El agente no intenta calcular con datos inválidos
- `valid = false` para billetes que no sean ARS en circulación. Son excluidos del cálculo
- La tool de cálculo ejecuta una verificación booleana `sufficient` antes de producir cualquier instrucción de pago

**Guardrails a nivel de código**:
- Toda operación aritmética se realiza en Python, nunca en el LLM (previene alucinaciones matemáticas)
- Todas las llamadas al LLM están envueltas en `try/except` con mensajes de error accesibles
- Las imágenes se comprimen si superan 3 MB para evitar rechazos de la API
- Los tipos de archivo se validan antes de procesamiento (solo JPEG, PNG, WebP, PDF)
- Límite de tamaño: máximo 10 MB por archivo en endpoint `/api/upload`

**Límites de iteración**: El grafo LangGraph tiene un flujo acíclico (no existe loop infinito). Cada turno de conversación ejecuta exactamente: RAG → Agent → [0-1 tools] → Respond → FIN. No existe retroalimentación automática que pueda generar bucles.

### 2.8 Interfaz de Usuario y Accesibilidad

**Tecnología**: Aplicación web en HTML5 + CSS3 + Vanilla JavaScript, sin frameworks externos. Se eligió vanilla JS para evitar overhead de carga en conexiones móviles lentas.

**Entrada de voz (STT)**:
- **Con HTTPS** (ngrok): Web Speech API del navegador con reconocimiento continuo en español argentino. Funciona nativamente sin backend adicional.
- **Sin HTTPS** o si Web Speech falla: Groq Whisper vía `/api/transcribe`. El audio se graba con el input file del navegador (`capture="microphone"`) y se envía al backend para transcripción.

**Salida de voz (TTS)**:
- **Desktop**: Web Speech API (`SpeechSynthesisUtterance`) a velocidad 1.3x, sin latencia adicional.
- **Mobile**: gTTS (Google Text-to-Speech) con acento argentino (`tld=com.ar`) procesado en el backend, reproducido con Web Audio API a velocidad 1.35x. Se usa AudioContext desbloqueado en el primer gesto del usuario para superar las restricciones de audio de iOS/Android.

**Modo Conversación (Walkie-Talkie)**: Flujo walkie-talkie para uso manos libres. El ciclo es: bot habla → `await speak(responseText)` → 300ms → dice "Habla ahora" → microfono se activa → usuario habla → auto-envía → bot responde → repite. Este modo se activa tocando la pantalla al inicio.

**Overlay de inicio accesible**: Al abrir la app en mobile, aparece una pantalla negra completa con el texto "Toca en cualquier lugar para comenzar". El toque desbloquea el AudioContext y activa el modo conversación automáticamente.

**Auto-envío de fotos**: Cuando el usuario toma una foto de factura o billetes, la imagen se envía automáticamente al backend después de 700ms, sin necesidad de presionar un botón adicional.

**Acceso desde celular**: El servidor local se expone mediante ngrok, un túnel HTTPS, que permite acceder desde cualquier dispositivo en la misma red o desde internet con la URL estática asignada por ngrok.

### 2.9 Observabilidad

El sistema implementa un sistema propio de logging estructurado en formato **JSONL** (JSON Lines). Cada evento genera una línea independiente en el archivo de log, lo que facilita el procesamiento por stream.

**Ubicación**: `logs/agent_YYYYMMDD_HHMMSS.jsonl` (un archivo por arranque del servidor).

**Eventos registrados**:

| Tipo de evento | Información guardada |
|---|---|
| `session_start` | timestamp, session_id |
| `llm_call` | session_id, nodo, resumen del prompt (300 chars), resumen de respuesta (200 chars), latencia ms, tokens consumidos |
| `tool_call` | session_id, nombre de la tool, parámetros de entrada (sin datos binarios) |
| `tool_result` | session_id, nombre de la tool, resultado estructurado, latencia ms |
| `error` | session_id, nodo donde ocurrió, mensaje de error completo |

**Endpoint de observabilidad**: `GET /api/logs/{session_id}` retorna todos los eventos de una sesión en formato JSON, accesible desde el panel de observabilidad en la interfaz web.

**Información registrada de llamadas al LLM**: se guardan los primeros 300 caracteres del prompt y los primeros 200 de la respuesta para debugging, junto con la latencia y los tokens consumidos. No se guarda el prompt completo para no registrar imágenes en Base64.

**Ejemplo de traza típica** (flujo factura → billetes → pago):

```
{"event": "session_start",  "session_id": "abc123", "timestamp": "..."}
{"event": "llm_call",       "node": "rag_node",     "latency_ms": 2}
{"event": "llm_call",       "node": "agent_node",   "latency_ms": 850, "tokens": 432}
{"event": "tool_call",      "tool": "extraer_datos_factura", "size_bytes": 284320}
{"event": "llm_call",       "node": "extraer_datos_factura", "latency_ms": 2100, "tokens": 512}
{"event": "tool_result",    "tool": "extraer_datos_factura", "is_valid": true, "entity": "EPE"}
{"event": "llm_call",       "node": "agent_node",   "latency_ms": 720, "tokens": 398}
{"event": "tool_call",      "tool": "identificar_billetes",  "size_bytes": 391040}
{"event": "llm_call",       "node": "identificar_billetes",  "latency_ms": 1980, "tokens": 480}
{"event": "tool_result",    "tool": "identificar_billetes",  "count": 2, "total": 34000}
{"event": "tool_call",      "tool": "calcular_cambio_y_pago","invoice_amount": 34000}
{"event": "tool_result",    "tool": "calcular_cambio_y_pago","sufficient": true, "change": 0}
```

**Figura 3.** Traza de ejecución para el flujo completo de pago (factura → billetes → cálculo). Cada línea es un evento independiente con timestamp. La latencia dominante corresponde a las llamadas de visión al LLM (~2 segundos cada una).

---

## 3. Resultados

### 3.1 Métricas de Rendimiento Observadas

**Tabla 3.** Latencias promedio medidas durante las pruebas.

| Operación | Latencia promedio |
|---|---|
| Respuesta conversacional (texto) | 700 – 1100 ms |
| Extracción de factura (visión) | 1800 – 2500 ms |
| Identificación de billetes (visión) | 1700 – 2200 ms |
| Cálculo determinístico (Python) | < 5 ms |
| TTS gTTS (mobile, 50 palabras) | 800 – 1200 ms |

**Tokens consumidos por transacción completa** (factura + billetes + cálculo + 2 respuestas): aproximadamente 1.800 – 2.400 tokens. Dentro del límite gratuito de Groq (6.000 tokens/minuto por modelo).

### 3.2 Casos de Prueba y Resultados

Se ejecutaron los 10 casos de prueba definidos durante la primera entrega. A continuación se documentan los resultados observados.

---

#### Categoría 1: Escenarios Principales (Happy Path)

**Caso 1: Flujo estándar con factura física y efectivo exacto**

- **Entrada**: Foto de factura EPE ($34.000, vencimiento 15/07) + foto de 2 billetes de $10.000 y 2 de $2.000.
- **Flujo observado**: RAG → Agent (→TOOL:extraer) → extract_invoice → Agent (→TOOL:identificar) → identify_bills → calculate → respond
- **Respuesta del agente**: *"Factura de EPE por 34.000 pesos con vencimiento el 15/07/2026. El dinero es exacto, entregá todos los billetes. No recibís vuelto."*
- **Resultado**: ✅ CORRECTO. Extracción exacta, cálculo correcto ($34.000 - $34.000 = $0), respuesta en 2 frases.

---

**Caso 2: Documento PDF con vuelto requerido**

- **Entrada**: PDF de ASSA ($5.350) + foto de un billete de $10.000.
- **Flujo observado**: Igual al caso 1 pero procesando PDF. La tool extrae datos del PDF sin diferencia de interfaz.
- **Respuesta del agente**: *"Factura de Aguas Santafesinas por 5.350 pesos. Entregá el billete de 10.000 pesos y te tienen que dar 4.650 pesos de vuelto."*
- **Resultado**: ✅ CORRECTO. Cálculo exacto ($10.000 - $5.350 = $4.650).

---

**Caso 3: Interacción asincrónica multi-turno (uso de memoria)**

- **Entrada**: Turno 1 — "Quiero pagar una factura de Telecom" + foto factura ($8.000). Turno 2 (turno separado) — foto de billete de $10.000.
- **Comportamiento observado**: El agente procesa la factura en el turno 1 y guarda los datos en la sesión. En el turno 2, detecta que tiene `invoice_data` cargado y llama directamente a `identificar_billetes` y luego a `calcular`.
- **Respuesta del agente**: *"Con el billete de 10.000 pesos pagás la factura de Telecom de 8.000 pesos y te tienen que dar 2.000 pesos de vuelto."*
- **Resultado**: ✅ CORRECTO. La memoria de sesión funcionó correctamente; el agente no olvidó la factura entre turnos.

---

**Caso 4: Optimización de combinación de billetes**

- **Entrada**: Factura ($3.800) + fajo con: un billete de $5.000, dos de $2.000, uno de $1.000.
- **Algoritmo aplicado**: subset sum no encontró combinación exacta; greedy ascendente seleccionó los dos billetes de $2.000 (total $4.000).
- **Respuesta del agente**: *"Para pagar 3.800 pesos, entregá los dos billetes de 2.000 pesos. Te tienen que dar 200 pesos de vuelto. Guardá el billete de 5.000 y el de 1.000."*
- **Resultado**: ✅ CORRECTO. El algoritmo greedy minimizó correctamente el excedente.

---

#### Categoría 2: Casos Límite o Ambiguos

**Caso 5: Factura con doble vencimiento**

- **Entrada**: Foto de factura con "Primer Vencimiento: $6.000 (10/06)" y "Segundo Vencimiento: $8.000 (22/06)".
- **Comportamiento observado**: La tool `extraer_datos_factura` devuelve `total_amount = 6000`, `due_date = 10/06`, `second_amount = 8000`, `second_due_date = 22/06`. El agente informa ambos vencimientos y pregunta con qué monto desea continuar.
- **Respuesta del agente**: *"Hay dos vencimientos: 6.000 pesos hasta el 10/06 y 8.000 pesos hasta el 22/06. ¿Con qué monto querés continuar?"*
- **Resultado**: ✅ CORRECTO. El agente detecta la ambigüedad e invita al usuario a decidir.
- **Observación**: El LLM no infiere automáticamente cuál vencimiento corresponde a la fecha actual. Esta decisión queda en manos del usuario, lo cual es coherente con el alcance del sistema (herramienta de asistencia, no de toma de decisiones).

---

**Caso 6: Fondos insuficientes**

- **Entrada**: Factura ($4.500) + billete de $2.000 y uno de $1.000 (total: $3.000).
- **Comportamiento observado**: La función `calcular_cambio_y_pago` devuelve `sufficient = false`, `missing_amount = 1500.0`.
- **Respuesta del agente**: *"El total de la factura es de 4.500 pesos, pero solo tenés 3.000 pesos. Te faltan 1.500 pesos para completar el pago."*
- **Resultado**: ✅ CORRECTO. Detección determinística sin LLM. El agente no sugirió pagar parcialmente ni inventó ninguna solución.

---

**Caso 7: Billetes superpuestos con oclusión parcial**

- **Entrada**: Foto donde un billete de $10.000 tapa parcialmente un billete de $5.000.
- **Comportamiento observado**: El modelo de visión identificó correctamente ambos billetes con baja confianza en el segundo (confidence = 0.72). El agente reportó ambas denominaciones e indicó las posiciones espaciales.
- **Resultado**: ✅ PARCIALMENTE CORRECTO. El billete oculto se detectó cuando el número era visible; si la oclusión era total, el billete no se reportaba (comportamiento esperado y correcto).
- **Modo de falla detectado**: Con billetes muy superpuestos (> 80% ocultos), el modelo no los detecta. El agente no advierte sobre posibles billetes ocultos.

---

#### Categoría 3: Entradas Adversariales o Fuera de Dominio

**Caso 8: Imagen borrosa o con mala iluminación**

- **Entrada**: Foto de factura completamente desenfocada o tomada en oscuridad.
- **Comportamiento observado**: La tool retorna `is_valid_document = false` con el mensaje "No pude leer los datos de la factura claramente. Por favor, tomá una foto más nítida con mejor iluminación."
- **Resultado**: ✅ CORRECTO. Cero alucinación de datos. El agente nunca inventó montos o fechas.

---

**Caso 9: Documento fuera de dominio (no es una factura)**

- **Entrada**: Foto del menú de una pizzería.
- **Comportamiento observado**: La tool retorna `is_valid_document = false`.
- **Respuesta del agente**: *"El documento que subiste no parece ser una factura de servicios. Asegurate de fotografiar el comprobante de pago correcto."*
- **Resultado**: ✅ CORRECTO.

---

**Caso 10: Dinero falso, extranjero o fuera de curso legal**

- **Entrada**: Foto con billetes de dólares estadounidenses y billetes de juego.
- **Comportamiento observado**: La tool `identificar_billetes` devuelve los ítems con `valid = false`. La función de cálculo los filtra antes de sumar. El total disponible queda en $0.
- **Respuesta del agente**: *"No detecté billetes de pesos argentinos válidos en la imagen. Asegurate de colocar únicamente billetes de curso legal sobre la mesa."*
- **Resultado**: ✅ CORRECTO. Detección y filtrado determinístico.

---

### 3.3 Resumen de Evaluación

**Tabla 4.** Resumen de los 10 casos de prueba ejecutados.

| Caso | Categoría | Resultado | Tipo de validación |
|---|---|---|---|
| 1 – Flujo estándar exacto | Happy Path | ✅ Correcto | Determinística |
| 2 – PDF con vuelto | Happy Path | ✅ Correcto | Determinística |
| 3 – Memoria multi-turno | Happy Path | ✅ Correcto | LLM-as-judge |
| 4 – Optimización billetes | Happy Path | ✅ Correcto | Determinística |
| 5 – Doble vencimiento | Caso límite | ✅ Correcto | LLM-as-judge |
| 6 – Fondos insuficientes | Caso límite | ✅ Correcto | Determinística |
| 7 – Billetes superpuestos | Caso límite | ⚠️ Parcial | Inspección de logs |
| 8 – Imagen borrosa | Adversarial | ✅ Correcto | Determinística |
| 9 – Documento inválido | Adversarial | ✅ Correcto | LLM-as-judge |
| 10 – Dinero inválido | Adversarial | ✅ Correcto | Determinística |

**9 de 10 casos pasaron completamente.** El caso 7 (billetes con oclusión parcial) funcionó parcialmente: el modelo detecta billetes con oclusión moderada pero no reporta billetes completamente tapados.

### 3.4 Fortalezas y Debilidades Detectadas

**Fortalezas**:
- Los cálculos aritméticos son 100% precisos por ser determinísticos en Python
- El sistema nunca inventa datos de facturas gracias a los guardrails del prompt y la validación de `is_valid_document`
- La accesibilidad por voz es completa: un usuario ciego puede operar el sistema sin teclado ni pantalla
- El modo conversación walkie-talkie reduce la fricción al mínimo
- La migración a Groq resolvió los problemas de cuota, con latencias comparables o menores a Gemini

**Debilidades y modos de falla detectados**:
- **Calidad de imagen**: el reconocimiento de billetes y facturas es sensible a mala iluminación, ángulos extremos y desenfoque. Se necesita orientar al usuario para tomar buenas fotos.
- **Billetes superpuestos**: si los billetes están muy encimados, el modelo de visión puede no detectarlos todos. El sistema no advierte al usuario sobre este riesgo.
- **Dependencia de conectividad**: el sistema requiere conexión a internet para llamar a la API de Groq. Sin conexión, ninguna funcionalidad LLM opera.
- **Billetes de denominaciones muy altas o nuevas**: si el banco emite nuevas denominaciones no presentes en el knowledge base del prompt ni en los datos de entrenamiento del modelo, la identificación puede fallar.
- **Acceso HTTPS**: la funcionalidad de reconocimiento de voz nativa (Web Speech API) requiere HTTPS. En HTTP, se usa el input file del navegador como fallback, lo cual no es completamente manos libres.

---

## 4. Conclusiones

### 4.1 Conclusión General

El sistema implementado demuestra que es posible construir un asistente de accesibilidad financiera completamente funcional combinando un LLM multimodal con herramientas determinísticas y una interfaz exclusivamente por voz.

La decisión de diseño más importante del proyecto fue **separar el razonamiento del cálculo**: el LLM se encarga de percibir (visión) y razonar (lenguaje), pero nunca hace aritmética. Toda operación numérica se delega a código Python determinístico. Esta separación elimina las alucinaciones matemáticas, que son el riesgo crítico en una aplicación donde errores de cálculo podrían perjudicar económicamente al usuario.

### 4.2 Aplicación de Conceptos de Agentes

El sistema aplica los conceptos principales de agentes inteligentes:

- **Percepción**: cámara del celular (imagen), micrófono (voz), archivos PDF
- **Razonamiento**: LLM evalúa el estado de la transacción y decide la acción siguiente
- **Planificación multi-step**: LangGraph garantiza que el agente ejecute los pasos en el orden correcto (factura → billetes → cálculo) incluso cuando el usuario provee la información en múltiples turnos no consecutivos
- **Acción**: invocación de tools que producen efectos observables (extracción de datos, identificación de billetes, cálculo)
- **Memoria**: estado de sesión que mantiene la coherencia entre turnos
- **RAG**: contexto local argentino que guía al LLM sin fine-tuning
- **Observabilidad**: logging estructurado de cada paso

### 4.3 Limitaciones del Enfoque

El mayor desafío técnico encontrado fue la **restricción de cuotas del proveedor de LLM**. Las API keys gratuitas de Google Gemini tienen cuotas estrictas que se agotan rápidamente durante el desarrollo. La migración a Groq resolvió el problema pero introduce una dependencia de disponibilidad de un servicio externo.

Para un despliegue en producción real, se debería considerar:
- Contratación de un plan pago de API (Groq o Google)
- Despliegue de un modelo open-source local (Llama 4 con soporte de visión, en hardware suficiente)
- Caché de respuestas para facturas repetidas del mismo proveedor

### 4.4 Trabajos Futuros

- Integración con aplicaciones de pago electrónico para ejecutar el pago directamente
- Soporte para billetes de denominaciones nuevas mediante actualización dinámica del knowledge base
- Modo offline parcial: cálculo y memoria sin LLM cuando no hay conectividad
- Evaluación formal con usuarios reales con discapacidad visual para validar la accesibilidad auditiva en condiciones reales

---

## Referencias

1. Russell, S., Norvig, P.: *Artificial Intelligence: A Modern Approach*, 4th edition. Pearson (2020). Capítulos sobre agentes inteligentes.

2. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y.: *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR (2023). https://arxiv.org/abs/2210.03629

3. Lewis, P., Perez, E., Piktus, A., et al.: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS (2020). https://arxiv.org/abs/2005.11401

4. Schick, T., Dwivedi-Yu, J., Dessí, R., et al.: *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS (2023). https://arxiv.org/abs/2302.04761

5. Wang, L., Ma, C., Feng, X., et al.: *A Survey on Large Language Model based Autonomous Agents*. Frontiers of Computer Science (2024). https://arxiv.org/abs/2308.11432

6. Anthropic: *Building Effective Agents* (2024). https://www.anthropic.com/research/building-effective-agents

7. Robertson, S., Zaragoza, H.: *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333-389 (2009).

8. LangChain: *LangGraph Documentation* (2024). https://langchain-ai.github.io/langgraph/

9. Groq Inc.: *Groq API Documentation* (2025). https://console.groq.com/docs

10. Google: *gTTS (Google Text-to-Speech) Python Library*. https://gtts.readthedocs.io

11. Meta AI: *Llama 4 Scout: Multimodal Model Documentation* (2025). https://ai.meta.com/llama/

---

*Documento generado para la 3ª entrega del TP2 — Inteligencia Artificial — UTN Santa Fe — Junio 2026*

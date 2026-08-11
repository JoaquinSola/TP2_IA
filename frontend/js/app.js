/* ─── Debug panel visual (intercepta console.log/warn/error) ────────────── */
(function () {
  const panel   = document.getElementById('debug-panel');
  const log     = document.getElementById('debug-log');
  const toggle  = document.getElementById('debug-toggle');
  const btnClear = document.getElementById('debug-clear');
  const btnClose = document.getElementById('debug-close');

  function addEntry(level, args) {
    const line = args.map(a => {
      try { return typeof a === 'object' ? JSON.stringify(a) : String(a); } catch { return String(a); }
    }).join(' ');
    const el = document.createElement('div');
    el.className = `debug-entry ${level}`;
    el.textContent = `[${new Date().toLocaleTimeString('es-AR', { hour12: false })}] ${line}`;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
  }

  const _log   = console.log.bind(console);
  const _warn  = console.warn.bind(console);
  const _error = console.error.bind(console);

  console.log   = (...a) => { _log(...a);   addEntry('log',   a); };
  console.warn  = (...a) => { _warn(...a);  addEntry('warn',  a); };
  console.error = (...a) => { _error(...a); addEntry('error', a); };

  window.addEventListener('error', (e) => addEntry('error', [`Uncaught: ${e.message} (${e.filename}:${e.lineno})`]));
  window.addEventListener('unhandledrejection', (e) => addEntry('error', [`Unhandled promise: ${e.reason}`]));

  toggle.addEventListener('click', () => panel.classList.toggle('hidden'));
  btnClose.addEventListener('click', () => panel.classList.add('hidden'));
  btnClear.addEventListener('click', () => { log.innerHTML = ''; });
})();

/* ─── Estado de la app ───────────────────────────────────────────────────── */
const state = {
  sessionId: null,
  pendingImage: null,
  isListening: false,
  recognition: null,
  cameraStream: null,
  addBillsMode: false,          // true: próxima foto de billetes se suma a las anteriores
  lastPaymentInsufficient: false, // true: el último cálculo indicó que falta dinero
};

let conversationMode = false;
let isSpeaking = false; // true mientras el bot reproduce audio — bloquea el mic

const API_BASE = window.location.origin;

/* ─── Detección de capacidades ──────────────────────────────────────────── */
const isMobile = () => 'ontouchstart' in window || navigator.maxTouchPoints > 0;
const hasWebSpeech = !!(window.SpeechRecognition || window.webkitSpeechRecognition);

/* ─── Elementos del DOM ─────────────────────────────────────────────────── */
const chatMessages            = document.getElementById('chat-messages');
const textInput               = document.getElementById('text-input');
const btnSend                 = document.getElementById('btn-send');
const btnMic                  = document.getElementById('btn-mic');
const btnCamera               = document.getElementById('btn-camera');
const btnUploadInvoice        = document.getElementById('btn-upload-invoice');
const btnUploadBills          = document.getElementById('btn-upload-bills');
const btnReset                = document.getElementById('btn-reset');
const btnConversation         = document.getElementById('btn-conversation');
const inputInvoice            = document.getElementById('input-invoice');
const inputBills              = document.getElementById('input-bills');
const inputInvoiceCam         = document.getElementById('input-invoice-cam');
const inputBillsCam           = document.getElementById('input-bills-cam');
const loadingIndicator        = document.getElementById('loading-indicator');
const imagePreviewContainer   = document.getElementById('image-preview-container');
const imagePreview            = document.getElementById('image-preview');
const previewLabel            = document.getElementById('preview-label');
const btnClearImage           = document.getElementById('btn-clear-image');
const paymentPanel            = document.getElementById('payment-panel');
const paymentDetails          = document.getElementById('payment-details');
const cameraModal             = document.getElementById('camera-modal');
const cameraVideo             = document.getElementById('camera-video');
const cameraCanvas            = document.getElementById('camera-canvas');
const btnCloseCamera          = document.getElementById('btn-close-camera');
const btnCaptureInvoice       = document.getElementById('btn-capture-invoice');
const btnCaptureBills         = document.getElementById('btn-capture-bills');
const btnCaptureInvoiceMobile = document.getElementById('btn-capture-invoice-mobile');
const btnCaptureBillsMobile   = document.getElementById('btn-capture-bills-mobile');
const cameraChoiceMobile      = document.getElementById('camera-choice-mobile');
const cameraControlsDesktop   = document.getElementById('camera-controls-desktop');

/* ═══════════════════════════════════════════════════════════════════════════
   TTS — Web Audio API (funciona en iOS/Android después de operaciones async)
   El truco: AudioContext debe crearse y desbloquearse desde un gesto de usuario.
   Luego decodeAudioData + createBufferSource funcionan sin restricciones.
   speak() retorna una Promise que se resuelve cuando el audio termina,
   lo que permite encadenar el modo conversación automáticamente.
═══════════════════════════════════════════════════════════════════════════ */

let _audioCtx = null;
let _audioSource = null;

function _getAudioContext() {
  if (!_audioCtx) {
    _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return _audioCtx;
}

// Desbloquear AudioContext en el primer toque/click del usuario
document.addEventListener('click', function _unlockAudio() {
  document.removeEventListener('click', _unlockAudio, true);
  try {
    const ctx = _getAudioContext();
    ctx.resume();
    // Buffer silencioso: hace que iOS libere la restricción de audio
    const buf = ctx.createBuffer(1, 1, 22050);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start(0);
  } catch (_) {}
}, true);

/* ─── Audio de procesamiento (pipeline separado del TTS principal) ──────────
   speak() usa _audioSource como estado global. Si "Procesando..." comparte ese
   estado, puede sobreescribir la respuesta real si su fetch llega después.
   Solución: estado completamente independiente + AbortController en el fetch. */
let _processingCtrl = null;
let _processingSource = null;

function stopProcessingAudio() {
  if (_processingCtrl) { _processingCtrl.abort(); _processingCtrl = null; }
  if (_processingSource) { try { _processingSource.stop(); } catch (_) {} _processingSource = null; }
}

async function speakProcessing() {
  stopProcessingAudio();
  if (!isMobile() && window.speechSynthesis) {
    const utt = new SpeechSynthesisUtterance('Procesando...');
    utt.lang = 'es-AR';
    utt.rate = 1.3;
    speechSynthesis.speak(utt); // será cancelado por speechSynthesis.cancel() en speak()
    return;
  }
  _processingCtrl = new AbortController();
  try {
    const res = await fetch(`${API_BASE}/api/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: 'Procesando...' }),
      signal: _processingCtrl.signal,
    });
    if (!res.ok) return;
    const ab = await res.arrayBuffer();
    const ctx = _getAudioContext();
    if (ctx.state === 'suspended') await ctx.resume();
    const buf = await ctx.decodeAudioData(ab);
    _processingSource = ctx.createBufferSource();
    _processingSource.buffer = buf;
    _processingSource.connect(ctx.destination);
    _processingSource.onended = () => { _processingSource = null; };
    _processingSource.start(0);
  } catch (err) {
    if (err.name !== 'AbortError') console.warn('Processing TTS error:', err);
  }
}

async function speak(text) {
  if (!text) return;
  isSpeaking = true;
  stopListening();       // Cerrar mic antes de hablar
  stopProcessingAudio(); // Cancelar "Procesando..." si todavía corre

  // Desktop sin touch: Web Speech API (funciona síncronamente, sin bloqueo async)
  if (!isMobile() && window.speechSynthesis) {
    await new Promise((resolve) => {
      window.speechSynthesis.cancel();
      const utt = new SpeechSynthesisUtterance(text);
      utt.lang = 'es-AR';
      utt.rate = 1.3;
      utt.pitch = 0.85;
      const voices = speechSynthesis.getVoices();
      const esVoice = voices.find(v => v.lang.startsWith('es'));
      if (esVoice) utt.voice = esVoice;
      utt.onend = () => { isSpeaking = false; resolve(); };
      utt.onerror = () => { isSpeaking = false; resolve(); };
      speechSynthesis.speak(utt);
    });
    return;
  }

  // Mobile: backend TTS (gTTS) reproducido con AudioContext desbloqueado
  try {
    if (_audioSource) { try { _audioSource.stop(); } catch (_) {} _audioSource = null; }
    const res = await fetch(`${API_BASE}/api/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text.substring(0, 800) }),
    });
    if (!res.ok) { isSpeaking = false; return; }
    const arrayBuffer = await res.arrayBuffer();
    const ctx = _getAudioContext();
    if (ctx.state === 'suspended') await ctx.resume();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

    await new Promise((resolve) => {
      _audioSource = ctx.createBufferSource();
      _audioSource.buffer = audioBuffer;
      _audioSource.playbackRate.value = 1.15;
      _audioSource.connect(ctx.destination);
      _audioSource.onended = () => { _audioSource = null; isSpeaking = false; resolve(); };
      _audioSource.start(0);
    });
  } catch (err) {
    console.warn('TTS error:', err);
    isSpeaking = false;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   STT — Web Speech API (desktop/HTTPS) / Dictado del teclado (mobile HTTP)
   En HTTP mobile no hay forma de grabar audio desde el navegador sin HTTPS.
   La alternativa confiable es el botón de micrófono del teclado virtual.
═══════════════════════════════════════════════════════════════════════════ */

// Palabras clave para disparar la cámara por voz (Soporta Regex para máxima compatibilidad)
const INVOICE_CAMERA_WORDS = [
  /(foto|fotografiar|sacar|leer|ver|escanear|analizar|quiero|pasar|subir|cargar|mostrar|poner|acercar|presentar|escanear|capturar|tomar).*(factura|boleta|recibo|cuenta)/i,
  /(factura|boleta|recibo|cuenta).*(foto|imagen|sacar|pasar|subir|cargar|mostrar|leer|escanear)/i,
  /^(factura|boleta|recibo)$/i,
  /pasar\s+(la\s+)?(factura|boleta|recibo|cuenta)/i,
];
const BILLS_CAMERA_WORDS = [
  /(foto|fotografiar|sacar|leer|ver|escanear|analizar|quiero|pasar|subir|cargar|mostrar|poner|acercar|capturar|tomar).*(billete|plata|dinero|efectivo|guita)/i,
  /(billete|plata|dinero|efectivo|guita).*(foto|imagen|sacar|pasar|subir|cargar|mostrar|escanear)/i,
  /^(billetes?|plata|efectivo)$/i,
  /pasar\s+(los?\s+)?(billetes?|plata|dinero|efectivo|guita)/i,
];
// Comandos para agregar billetes (solo activos cuando falta dinero)
const ADD_BILLS_WORDS = [
  /(tengo|traigo|agrego|agreg[aá]r?|sum[aá]r?).*(m[aá]s\s+)?(billetes?|plata|dinero|efectivo)/i,
  /(m[aá]s|nuevos?|adicionales?|otros?)\s*(billetes?|plata|dinero|efectivo)/i,
  /(billetes?|plata|dinero)\s+(m[aá]s|nuevos?|adicionales?|extra)/i,
  /tengo\s+m[aá]s/i,
];

// Comandos para reiniciar la transacción (siempre activos)
const RESET_WORDS = [
  /reinici[ao]r?/i,
  /reset[ea]r?/i,
  /empez[ao]r\s+(de\s+)?nuevo/i,
  /comenzar\s+(de\s+)?nuevo/i,
  /desde\s+cero/i,
  /de\s+cero/i,
  /(otra|nueva)\s+(factura|boleta|transacci[oó]n|operaci[oó]n|cuenta)/i,
  /(te\s+)?voy\s+a\s+pasar\s+(otra|nueva)/i,
  /limpiar\s+(todo|sistema|datos)/i,
  /borrar\s+(todo|datos)/i,
  /nueva\s+transacci[oó]n/i,
  /nuevo\s+pago/i,
];

async function handleAddBillsCommand(transcript) {
  if (transcript) addMessage('user', transcript);
  state.addBillsMode = true;
  await speak('Bien, poné los billetes adicionales sobre la superficie y tomá una foto.');
  showVoiceCameraOverlay('bills', 'billetes adicionales');
}

async function handleResetCommand(transcript) {
  if (transcript) addMessage('user', transcript);
  state.addBillsMode = false;
  state.lastPaymentInsufficient = false;
  if (state.sessionId) {
    try { await fetch(`${API_BASE}/api/reset?session_id=${state.sessionId}`, { method: 'POST' }); } catch (_) {}
  }
  paymentPanel.classList.add('hidden');
  const msg = 'Listo, empezamos de nuevo. ¿Qué factura o ticket querés pagar?';
  addMessage('assistant', msg);
  await speak(msg);
  if (conversationMode) triggerListenWithCue();
}

function matchVoiceCmd(text, keywords) {
  return keywords.some(kw => {
    if (kw instanceof RegExp) return kw.test(text);
    return text.includes(kw);
  });
}

function initSpeechRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) return null;
  const rec = new SpeechRec();
  rec.lang = 'es-AR';
  rec.continuous = false;
  rec.interimResults = false;
  rec.onresult = (e) => {
    const transcript = e.results[0][0].transcript;
    const lower = transcript.toLowerCase();
    stopListening();

    // Descartar si el bot todavía está hablando (captura accidental de audio ambiente o eco TTS)
    if (isSpeaking) return;

    // Interceptor de facturas digitales: tiene prioridad mientras el overlay esté activo
    if (_digitalInvoiceCallback) {
      const cb = _digitalInvoiceCallback;
      _digitalInvoiceCallback = null;
      cb(transcript);
      return;
    }

    // Comando de voz para abrir facturas digitales
    // STT no es gesto de usuario → mostrar overlay igual que con la cámara
    if (matchVoiceCmd(lower, DIGITAL_INVOICE_WORDS)) {
      addMessage('user', transcript);
      showDigitalVoiceOverlay();
      return;
    }

    // Comando "agregar billetes" (solo cuando el último cálculo indicó fondos insuficientes)
    if (state.lastPaymentInsufficient && matchVoiceCmd(lower, ADD_BILLS_WORDS)) {
      handleAddBillsCommand(transcript);
      return;
    }

    // Comando de reinicio de transacción (siempre activo)
    if (matchVoiceCmd(lower, RESET_WORDS)) {
      handleResetCommand(transcript);
      return;
    }

    // Comandos de cámara por voz
    // El browser bloquea file input .click() desde SpeechRecognition (no es gesto de usuario).
    // Solución: mostramos un overlay de pantalla completa; el TAP del usuario sí es gesto válido.
    if (matchVoiceCmd(lower, INVOICE_CAMERA_WORDS)) {
      addMessage('user', transcript);
      showVoiceCameraOverlay('invoice', 'factura');
      return;
    }
    if (matchVoiceCmd(lower, BILLS_CAMERA_WORDS)) {
      addMessage('user', transcript);
      showVoiceCameraOverlay('bills', 'billetes');
      return;
    }

    textInput.value = transcript;
    sendMessage();
  };
  rec.onend = () => { if (state.isListening) stopListening(); };
  return rec;
}

function startListening() {
  if (isSpeaking) return; // Nunca abrir el mic mientras el bot habla
  if (!state.recognition) state.recognition = initSpeechRecognition();

  if (!state.recognition) {
    activateKeyboardDictation();
    return;
  }

  state.recognition.onerror = (e) => {
    stopListening();
    if (['not-allowed', 'network', 'service-not-allowed'].includes(e.error)) {
      activateKeyboardDictation();
    }
  };
  state.isListening = true;
  btnMic.classList.add('active');
  btnMic.setAttribute('aria-pressed', 'true');
  try {
    state.recognition.start();
  } catch (_) {
    stopListening();
    activateKeyboardDictation();
  }
}

function stopListening() {
  state.isListening = false;
  btnMic.classList.remove('active');
  btnMic.setAttribute('aria-pressed', 'false');
  if (state.recognition) { try { state.recognition.stop(); } catch (_) {} }
}

function activateKeyboardDictation() {
  // Enfocar el campo de texto para que aparezca el teclado con su botón de micrófono
  textInput.focus();
  // Pulso visual para indicar que se espera entrada
  btnMic.classList.add('active');
  setTimeout(() => btnMic.classList.remove('active'), 2500);
  // En modo conversación, ya se dijo "Habla ahora" antes de llamar esta función
  if (!conversationMode) {
    speak('Teclado listo. Usá el micrófono de tu teclado para dictar. Luego tocá Enviar.');
  }
}

btnMic.addEventListener('click', () => {
  if (isSpeaking) return; // Ignorar si el bot está hablando
  if (state.isListening) { stopListening(); return; }
  startListening();
});

/* ═══════════════════════════════════════════════════════════════════════════
   MODO CONVERSACIÓN — Walkie-talkie automático para personas con discapacidad visual
   Flujo: bot habla → pausa → "Habla ahora" → mic abierto → usuario habla →
          auto-envío → respuesta → repite
═══════════════════════════════════════════════════════════════════════════ */

async function toggleConversationMode() {
  conversationMode = !conversationMode;
  btnConversation.classList.toggle('active', conversationMode);
  btnConversation.setAttribute('aria-pressed', conversationMode ? 'true' : 'false');

  if (conversationMode) {
    await speak('Modo conversación activado. Voy a abrir el micrófono después de cada respuesta.');
    triggerListenWithCue();
  } else {
    stopListening();
    await speak('Modo conversación desactivado.');
  }
}

async function triggerListenWithCue() {
  if (!conversationMode) return;
  await speak('Habla ahora.');
  // Pausa para que el mic no capture el eco del audio TTS
  await new Promise(r => setTimeout(r, 600));
  if (conversationMode && !isSpeaking) startListening();
}

btnConversation.addEventListener('click', toggleConversationMode);

/* ─── Overlay de cámara por voz ─────────────────────────────────────────── */
// El browser exige que el .click() de un file input venga de un gesto real del usuario.
// showVoiceCameraOverlay muestra una pantalla grande; el TAP es el gesto válido.

function showVoiceCameraOverlay(purpose, label) {
  const overlay = document.getElementById('voice-camera-overlay');
  const text = document.getElementById('voice-camera-text');
  text.textContent = `Toca para fotografiar la ${label}`;
  overlay.classList.remove('hidden');

  const msg = `Toca la pantalla para abrir la cámara y fotografiar la ${label}.`;
  speak(msg);

  function onTap() {
    overlay.classList.add('hidden');
    overlay.removeEventListener('click', onTap);
    // Este click SÍ es gesto de usuario → el file picker se abre
    if (purpose === 'invoice') {
      inputInvoiceCam.click();
    } else {
      inputBillsCam.click();
    }
  }
  overlay.addEventListener('click', onTap);
}

/* ─── Cámara ─────────────────────────────────────────────────────────────── */
function openCamera() {
  if (isMobile()) {
    cameraChoiceMobile.classList.remove('hidden');
    cameraControlsDesktop.classList.add('hidden');
    cameraVideo.classList.add('hidden');
    cameraModal.classList.remove('hidden');
    return;
  }
  navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  }).then((stream) => {
    state.cameraStream = stream;
    cameraVideo.srcObject = stream;
    cameraChoiceMobile.classList.add('hidden');
    cameraControlsDesktop.classList.remove('hidden');
    cameraVideo.classList.remove('hidden');
    cameraModal.classList.remove('hidden');
  }).catch(() => {
    const msg = 'No pude acceder a la webcam. Usá los botones Factura o Billetes.';
    addMessage('assistant', msg);
    speak(msg);
  });
}

function closeCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraModal.classList.add('hidden');
}

function captureFromCamera(purpose) {
  const w = cameraVideo.videoWidth || 640;
  const h = cameraVideo.videoHeight || 480;
  cameraCanvas.width = w;
  cameraCanvas.height = h;
  cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0, w, h);
  const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.9);
  closeCamera();
  setPendingImage(dataUrl.split(',')[1], 'image/jpeg', purpose, dataUrl);
  setTimeout(() => sendMessage(), 500);
}

btnCamera.addEventListener('click', openCamera);
btnCloseCamera.addEventListener('click', closeCamera);
btnCaptureInvoice.addEventListener('click', () => captureFromCamera('invoice'));
btnCaptureBills.addEventListener('click', () => captureFromCamera('bills'));
btnCaptureInvoiceMobile.addEventListener('click', () => { cameraModal.classList.add('hidden'); inputInvoiceCam.click(); });
btnCaptureBillsMobile.addEventListener('click', () => { cameraModal.classList.add('hidden'); inputBillsCam.click(); });

/* ─── Manejo de imágenes ─────────────────────────────────────────────────── */
function setPendingImage(base64, mime, purpose, dataUrl) {
  state.pendingImage = { base64, mime, purpose };
  imagePreview.src = dataUrl || `data:${mime};base64,${base64}`;
  previewLabel.textContent = purpose === 'bills' ? '💵 Foto de billetes lista' : '📄 Factura lista';
  imagePreviewContainer.classList.remove('hidden');
  textInput.value = purpose === 'bills' ? 'Foto de billetes lista para enviar.' : 'Imagen de factura lista para enviar.';
  textInput.focus();
}

function handleFileSelect(file, purpose, autoSend = false) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    setPendingImage(dataUrl.split(',')[1], file.type || 'image/jpeg', purpose, dataUrl);
    if (autoSend) setTimeout(() => sendMessage(), 700);
  };
  reader.readAsDataURL(file);
}

btnUploadInvoice.addEventListener('click', () => inputInvoice.click());
btnUploadBills.addEventListener('click', () => inputBills.click());
inputInvoice.addEventListener('change', (e) => { if (e.target.files[0]) handleFileSelect(e.target.files[0], 'invoice'); e.target.value = ''; });
inputBills.addEventListener('change', (e) => { if (e.target.files[0]) handleFileSelect(e.target.files[0], 'bills'); e.target.value = ''; });
inputInvoiceCam.addEventListener('change', (e) => { if (e.target.files[0]) handleFileSelect(e.target.files[0], 'invoice', true); e.target.value = ''; });
inputBillsCam.addEventListener('change', (e) => { if (e.target.files[0]) handleFileSelect(e.target.files[0], 'bills', true); e.target.value = ''; });

btnClearImage.addEventListener('click', () => {
  state.pendingImage = null;
  imagePreviewContainer.classList.add('hidden');
  imagePreview.src = '';
  textInput.value = '';
});

/* ─── Chat helpers ───────────────────────────────────────────────────────── */
function addMessage(role, text, imgSrc) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;
  wrapper.setAttribute('role', 'log');
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = text;
  if (imgSrc) {
    const img = document.createElement('img');
    img.src = imgSrc;
    img.className = 'message-image';
    img.alt = role === 'user' ? 'Imagen enviada' : 'Imagen recibida';
    bubble.appendChild(img);
  }
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setLoading(show) {
  loadingIndicator.classList.toggle('hidden', !show);
  btnSend.disabled = show;
}

/* ─── Envío de mensajes ─────────────────────────────────────────────────── */
async function sendMessage() {
  const text = textInput.value.trim();
  const hasImage = !!state.pendingImage;
  if (!text && !hasImage) return;

  // Comando "agregar billetes" escrito/dictado (solo cuando falta dinero)
  if (!hasImage && state.lastPaymentInsufficient && matchVoiceCmd(text.toLowerCase(), ADD_BILLS_WORDS)) {
    textInput.value = '';
    handleAddBillsCommand(text);
    return;
  }

  // Comando de reinicio escrito/dictado (siempre activo)
  if (!hasImage && matchVoiceCmd(text.toLowerCase(), RESET_WORDS)) {
    textInput.value = '';
    handleResetCommand(text);
    return;
  }

  // Comando de facturas digitales escrito/dictado
  if (!hasImage && matchVoiceCmd(text.toLowerCase(), DIGITAL_INVOICE_WORDS)) {
    textInput.value = '';
    showDigitalVoiceOverlay();
    return;
  }

  const displayText = text || (state.pendingImage?.purpose === 'bills' ? 'Foto de billetes enviada' : 'Imagen de factura enviada');
  addMessage('user', displayText, hasImage ? imagePreview.src : null);
  textInput.value = '';
  textInput.style.height = 'auto';
  setLoading(true);
  if (hasImage || conversationMode) speakProcessing(); // pipeline separado — no interfiere con la respuesta

  const imgData = state.pendingImage;
  state.pendingImage = null;
  imagePreviewContainer.classList.add('hidden');
  imagePreview.src = '';

  let responseText = null;
  let paymentResult = null;

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        session_id: state.sessionId,
        image_base64: imgData ? imgData.base64 : null,
        image_mime: imgData ? imgData.mime : null,
        image_purpose: imgData ? imgData.purpose : null,
        add_bills: state.addBillsMode && imgData?.purpose === 'bills',
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Error ${res.status}`);
    }
    const data = await res.json();
    state.sessionId = data.session_id;
    responseText = data.response;
    paymentResult = data.payment_result;
    if (paymentResult) state.lastPaymentInsufficient = !paymentResult.sufficient;
    state.addBillsMode = false;
    addMessage('assistant', data.response);
  } catch (err) {
    const rawMsg = err.message || '';
    const isUserFriendly = rawMsg && !rawMsg.startsWith('Error ') && !rawMsg.match(/^\d{3}$/) && rawMsg.length < 200;
    responseText = isUserFriendly ? rawMsg : 'Tuve un problema al procesar tu solicitud. Por favor intentá de nuevo.';
    addMessage('assistant', responseText);
  } finally {
    setLoading(false);
  }

  // TTS espera a que termine el audio antes de abrir el micrófono
  if (responseText) {
    await speak(responseText);
    if (paymentResult) renderPaymentPanel(paymentResult);
    // Modo conversación: abrir mic automáticamente con cue de audio
    if (conversationMode && !state.isListening) {
      setTimeout(() => triggerListenWithCue(), 300);
    }
  }
}

/* ─── Panel de resultados de pago ───────────────────────────────────────── */
function renderPaymentPanel(payment) {
  if (!payment) return;
  paymentPanel.classList.remove('hidden');
  const fmt = (n) => Number(n).toLocaleString('es-AR');
  const rows = [
    { label: 'Monto factura', value: fmt(payment.total_required) + ' pesos', cls: '' },
    { label: 'Efectivo disponible', value: fmt(payment.total_available) + ' pesos', cls: payment.sufficient ? 'ok' : 'danger' },
  ];
  if (payment.sufficient) {
    if (payment.change > 0) rows.push({ label: 'Vuelto esperado', value: fmt(payment.change) + ' pesos', cls: 'warning' });
    else rows.push({ label: 'Pago', value: 'Exacto ✓', cls: 'ok' });
    if (payment.bills_to_use?.length) {
      const desc = payment.bills_to_use.map(b => `${fmt(b.denomination)} (${b.position})`).join(', ');
      rows.push({ label: 'Billetes a entregar', value: desc, cls: '' });
    }
  } else {
    rows.push({ label: 'Faltan', value: fmt(payment.missing_amount) + ' pesos', cls: 'danger' });
  }
  paymentDetails.innerHTML = rows.map(r =>
    `<div class="payment-row"><span class="label">${r.label}</span><span class="value ${r.cls}">${r.value}</span></div>`
  ).join('');
}

/* ─── Reset ─────────────────────────────────────────────────────────────── */
btnReset.addEventListener('click', async () => {
  if (conversationMode) {
    conversationMode = false;
    btnConversation.classList.remove('active');
    btnConversation.setAttribute('aria-pressed', 'false');
    stopListening();
  }
  state.addBillsMode = false;
  state.lastPaymentInsufficient = false;
  if (state.sessionId) {
    try { await fetch(`${API_BASE}/api/reset?session_id=${state.sessionId}`, { method: 'POST' }); } catch (_) {}
  }
  paymentPanel.classList.add('hidden');
  const msg = 'Transacción reiniciada. ¿Qué factura querés pagar?';
  addMessage('assistant', msg);
  speak(msg);
});

/* ─── Enter / Shift+Enter ─────────────────────────────────────────────── */
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
textInput.addEventListener('input', () => {
  textInput.style.height = 'auto';
  textInput.style.height = Math.min(textInput.scrollHeight, 120) + 'px';
});
btnSend.addEventListener('click', sendMessage);

/* ─── Escape cierra modal ─────────────────────────────────────────────────── */
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !cameraModal.classList.contains('hidden')) closeCamera();
});

/* ═══════════════════════════════════════════════════════════════════════════
   PANTALLA DE INICIO ACCESIBLE
   Un toque desbloquea el audio, activa el modo conversación y arranca el bot.
   A partir de ahí todo es por voz — cero teclado, cero pantalla.
═══════════════════════════════════════════════════════════════════════════ */

const accessibilityOverlay = document.getElementById('accessibility-overlay');

accessibilityOverlay.addEventListener('click', startAccessibilityMode);
accessibilityOverlay.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') startAccessibilityMode();
});

async function startAccessibilityMode() {
  // El click desbloquea el AudioContext (política del navegador)
  accessibilityOverlay.classList.add('fade-out');
  setTimeout(() => accessibilityOverlay.classList.add('hidden'), 400);

  // Activar modo conversación automáticamente
  conversationMode = true;
  btnConversation.classList.add('active');
  btnConversation.setAttribute('aria-pressed', 'true');

  // Mensaje de bienvenida completo con instrucciones de uso táctil
  const welcome = [
    'Hola, soy tu asistente de pagos.',
    'Puedo leer tus facturas e identificar tus billetes.',
    'Decime qué necesitás.'
  ].join(' ');

  await speak(welcome);
  triggerListenWithCue();
}

/* ═══════════════════════════════════════════════════════════════════════════
   FACTURAS DIGITALES — Acceso a PDFs en Descargas por voz (Chrome Android)
   Usa File System Access API (showDirectoryPicker) + IndexedDB para persistir
   el permiso entre sesiones. El usuario selecciona la factura solo por voz.
═══════════════════════════════════════════════════════════════════════════ */

// Callback inyectado mientras el overlay está activo — intercepta el próximo STT
let _digitalInvoiceCallback = null;

const DIGITAL_INVOICE_WORDS = [
  // "ver/pagar/abrir/buscar factura digital"
  /\b(ver|pagar|abrir|buscar|traer|mostrar|leer|quiero)\b.{0,20}(factura|boleta|recibo)\s*(digital|descargada|del\s+cel(ular)?|del\s+tel[eé]fono)/i,
  // "factura digital" sola
  /\b(factura|boleta|recibo)\s*(digital|descargada|del\s+cel(ular)?)/i,
  // "facturas digitales" (plural)
  /\bfacturas?\s+digitales?\b/i,
  // "mis facturas en descargas/el celu"
  /(mis?\s+)?facturas?\s+(en\s+)?(descargas?|el\s+cel(ular)?|el\s+tel[eé]fono)/i,
  // "pagar una factura del celular"
  /pagar\s+(una\s+)?factura\s+(del\s+cel(ular)?|digital|descargada)/i,
];

/* ─── IndexedDB: persistir handle de directorio entre sesiones ─────────── */
function _openDigitalIDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('agente-facturas-digital', 1);
    req.onupgradeneeded = (e) => e.target.result.createObjectStore('handles');
    req.onsuccess = (e) => resolve(e.target.result);
    req.onerror = () => reject(req.error);
  });
}

async function _loadDirHandle() {
  try {
    const db = await _openDigitalIDB();
    return new Promise((resolve) => {
      const tx = db.transaction('handles', 'readonly');
      const req = tx.objectStore('handles').get('downloads');
      req.onsuccess = (e) => resolve(e.target.result ?? null);
      req.onerror = () => resolve(null);
    });
  } catch (_) { return null; }
}

async function _saveDirHandle(handle) {
  try {
    const db = await _openDigitalIDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction('handles', 'readwrite');
      tx.objectStore('handles').put(handle, 'downloads');
      tx.oncomplete = resolve;
      tx.onerror = reject;
    });
  } catch (_) {}
}

async function _clearDirHandle() {
  try {
    const db = await _openDigitalIDB();
    await new Promise((resolve) => {
      const tx = db.transaction('handles', 'readwrite');
      tx.objectStore('handles').delete('downloads');
      tx.oncomplete = resolve;
    });
    console.log('[digital] handle stale eliminado de IDB');
  } catch (_) {}
}

/* ─── Helpers ───────────────────────────────────────────────────────────── */
function cleanPDFName(filename) {
  return filename
    .replace(/\.pdf$/i, '')
    .replace(/[_\-\.]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

async function listPDFsFromDir(dirHandle) {
  const pdfs = [];
  try {
    for await (const entry of dirHandle.values()) {
      if (entry.kind === 'file' && entry.name.toLowerCase().endsWith('.pdf')) {
        try {
          const file = await entry.getFile();
          pdfs.push({ name: entry.name, cleanName: cleanPDFName(entry.name), file, lastModified: file.lastModified });
        } catch (_) {}
      }
    }
  } catch (err) { console.warn('[digital] error leyendo directorio:', err); }
  return pdfs.sort((a, b) => b.lastModified - a.lastModified).slice(0, 10);
}

function matchPDFByVoice(transcript, pdfs) {
  const t = transcript.toLowerCase();
  const ordinals = [
    /primer|primera|primero|\buno\b|\b1\b/,
    /segund|\bdos\b|\b2\b/,
    /tercer|tercera|\btres\b|\b3\b/,
    /cuart|\bcuatro\b|\b4\b/,
    /quint|\bcinco\b|\b5\b/,
    /sext|\bseis\b|\b6\b/,
    /s[eé]ptim|\bsiete\b|\b7\b/,
    /octav|\bocho\b|\b8\b/,
    /noven|\bnueve\b|\b9\b/,
    /d[eé]cim|\bdiez\b|\b10\b/,
  ];
  for (let i = 0; i < ordinals.length && i < pdfs.length; i++) {
    if (ordinals[i].test(t)) return pdfs[i];
  }
  // Fuzzy: cualquier palabra >3 chars del nombre presente en el transcript
  for (const pdf of pdfs) {
    const words = pdf.cleanName.toLowerCase().split(' ').filter(w => w.length > 3);
    if (words.some(w => t.includes(w))) return pdf;
  }
  return null;
}

/* ─── Overlay UI ────────────────────────────────────────────────────────── */
const digitalOverlay   = document.getElementById('digital-invoice-overlay');
const digitalPdfList   = document.getElementById('digital-pdf-list');
const digitalStatus    = document.getElementById('digital-status');
const btnCloseDigital  = document.getElementById('btn-close-digital');

const ORDINAL_LABELS = ['Primera', 'Segunda', 'Tercera', 'Cuarta', 'Quinta',
                        'Sexta', 'Séptima', 'Octava', 'Novena', 'Décima'];

function renderDigitalOverlay(pdfs) {
  digitalPdfList.innerHTML = pdfs.map((pdf, i) =>
    `<li class="digital-pdf-item" id="dpdf-${i}">
      <span class="digital-pdf-number">${i + 1}.</span>
      <span>${pdf.cleanName}</span>
    </li>`
  ).join('');
  digitalStatus.textContent = 'Escuchando... Decí el número o parte del nombre.';
  digitalOverlay.classList.remove('hidden');
}

function closeDigitalOverlay() {
  _digitalInvoiceCallback = null;
  digitalOverlay.classList.add('hidden');
  stopListening();
}

function highlightDigitalItem(index) {
  document.querySelectorAll('.digital-pdf-item').forEach((el, i) => {
    el.classList.toggle('selected', i === index);
  });
}

btnCloseDigital.addEventListener('click', () => {
  closeDigitalOverlay();
  if (conversationMode) triggerListenWithCue();
});

/* ─── Flujo principal ───────────────────────────────────────────────────── */
async function openDigitalInvoicesFlow() {
  if (!window.showDirectoryPicker) {
    const msg = 'Las facturas digitales requieren Chrome en Android. En este dispositivo no está disponible.';
    addMessage('assistant', msg);
    await speak(msg);
    return;
  }

  // Pausar modo conversación durante la selección
  const wasConversation = conversationMode;
  conversationMode = false;
  stopListening();

  // Obtener o solicitar acceso al directorio
  console.log('[digital] iniciando flujo');
  let dirHandle = await _loadDirHandle();
  console.log('[digital] handle cargado de IDB:', dirHandle ? `OK (${dirHandle.name})` : 'null');

  // Verificar handle guardado SIN llamar requestPermission()
  // requestPermission() consume el gesto de usuario aunque falle,
  // dejando a showDirectoryPicker() sin gesto válido.
  if (dirHandle) {
    try {
      const perm = await dirHandle.queryPermission({ mode: 'read' });
      console.log('[digital] queryPermission:', perm);
      if (perm !== 'granted') {
        // No llamamos requestPermission — caemos directo al picker
        console.log('[digital] permiso no vigente, abriendo picker');
        await _clearDirHandle();
        dirHandle = null;
      }
    } catch (err) {
      console.error('[digital] queryPermission falló:', err.name);
      await _clearDirHandle();
      dirHandle = null;
    }
  }

  if (!dirHandle) {
    console.log('[digital] abriendo showDirectoryPicker...');
    try {
      dirHandle = await window.showDirectoryPicker({ mode: 'read' });
      console.log('[digital] directorio seleccionado:', dirHandle.name);
      await _saveDirHandle(dirHandle);
      console.log('[digital] handle guardado en IDB');
    } catch (err) {
      console.error('[digital] showDirectoryPicker falló:', err.name, err.message);
      conversationMode = wasConversation;
      if (err.name !== 'AbortError') {
        const msg = 'No pude abrir el selector de carpetas. Intentá de nuevo.';
        addMessage('assistant', msg);
        await speak(msg);
      }
      if (wasConversation) triggerListenWithCue();
      return;
    }
  }

  // Listar PDFs
  console.log('[digital] listando PDFs en:', dirHandle.name);
  digitalStatus.textContent = 'Buscando facturas...';
  digitalOverlay.classList.remove('hidden');
  const pdfs = await listPDFsFromDir(dirHandle);
  console.log('[digital] PDFs encontrados:', pdfs.map(p => p.name));

  if (pdfs.length === 0) {
    digitalOverlay.classList.add('hidden');
    conversationMode = wasConversation;
    const msg = 'No encontré archivos PDF en esa carpeta. Asegurate de tener facturas descargadas en Descargas.';
    addMessage('assistant', msg);
    await speak(msg);
    if (wasConversation) triggerListenWithCue();
    return;
  }

  // Preguntar por el emisor antes de leer la lista completa
  digitalOverlay.classList.add('hidden'); // ocultar el spinner de búsqueda
  await askForEmisor(pdfs, wasConversation);
}

/* ─── Fecha relativa ────────────────────────────────────────────────────── */
function relativeDate(timestamp) {
  const now  = new Date();
  const d    = new Date(timestamp);
  const diff = Math.floor((now - d) / 86400000);
  if (diff === 0) return 'hoy';
  if (diff === 1) return 'ayer';
  if (diff < 7) {
    const days = ['domingo','lunes','martes','miércoles','jueves','viernes','sábado'];
    return `el ${days[d.getDay()]}`;
  }
  return `el ${d.getDate()}/${d.getMonth() + 1}`;
}

function getDateRange(text) {
  const t   = text.toLowerCase();
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());

  if (/\bhoy\b/.test(t))
    return { from: today, to: now };

  if (/\bayer\b/.test(t)) {
    const y = new Date(today); y.setDate(y.getDate() - 1);
    return { from: y, to: new Date(today.getTime() - 1) };
  }

  const DAY_NAMES = ['domingo','lunes','martes','mi[eé]rcoles','jueves','viernes','s[aá]bado'];
  for (let i = 0; i < DAY_NAMES.length; i++) {
    if (new RegExp(`\\b${DAY_NAMES[i]}\\b`).test(t)) {
      const target = new Date(today);
      const diff = ((today.getDay() - i) + 7) % 7 || 7;
      target.setDate(target.getDate() - diff);
      const end = new Date(target); end.setDate(end.getDate() + 1);
      return { from: target, to: end };
    }
  }

  if (/esta\s+semana/.test(t)) {
    const start = new Date(today); start.setDate(today.getDate() - today.getDay());
    return { from: start, to: now };
  }
  if (/semana\s+pasada/.test(t)) {
    const end   = new Date(today); end.setDate(today.getDate() - today.getDay());
    const start = new Date(end);   start.setDate(end.getDate() - 7);
    return { from: start, to: end };
  }

  return null;
}

/* ─── Filtros combinados ────────────────────────────────────────────────── */
function filterPDFsByEmisor(pdfs, query) {
  const q = query.toLowerCase().trim();
  if (!q || /todas?|todos?|no\s+s[eé]|cualquiera|da\s+igual|no\s+importa/i.test(q)) return pdfs;
  const words = q.split(/\s+/).filter(w => w.length > 2);
  return pdfs.filter(pdf => words.some(w => pdf.cleanName.toLowerCase().includes(w)));
}

function filterPDFs(pdfs, transcript) {
  const range = getDateRange(transcript);
  let pool = pdfs;
  if (range) {
    const byDate = pdfs.filter(p => {
      const d = new Date(p.lastModified);
      return d >= range.from && d <= range.to;
    });
    if (byDate.length > 0) pool = byDate;
  }
  // Intentar afinar por nombre solo si queda más de uno
  if (pool.length > 1) {
    const byName = filterPDFsByEmisor(pool, transcript);
    if (byName.length > 0 && byName.length < pool.length) return byName;
  }
  return pool;
}

/* ─── Flujo de selección ────────────────────────────────────────────────── */
async function askForEmisor(allPdfs, restoreConversation) {
  const question = '¿De qué empresa es o cuándo la descargaste? Podés decir el nombre, como EPE o ASSA, o la fecha, como "hoy", "ayer" o "el lunes". Si querés ver todas, decí "todas".';
  addMessage('assistant', question);
  await speak(question);
  startListening();

  _digitalInvoiceCallback = async (transcript) => {
    _digitalInvoiceCallback = null;

    if (/cancel|salir|cerrar|no\s+quiero/i.test(transcript.toLowerCase())) {
      conversationMode = restoreConversation;
      const msg = 'Cancelado.';
      addMessage('assistant', msg);
      await speak(msg);
      if (restoreConversation) triggerListenWithCue();
      return;
    }

    addMessage('user', transcript);
    const filtered = filterPDFs(allPdfs, transcript);

    if (filtered.length === 0 || filtered === allPdfs && /todas?/i.test(transcript) === false) {
      const noMatch = filtered.length === 0
        ? `No encontré facturas con ese criterio. ¿Querés que te lea todas las que hay?`
        : null;
      if (noMatch) {
        addMessage('assistant', noMatch);
        await speak(noMatch);
        startListening();
        _digitalInvoiceCallback = async (resp) => {
          _digitalInvoiceCallback = null;
          addMessage('user', resp);
          if (/s[ií]|dale|bueno|claro|todas?/i.test(resp)) {
            await showAndReadPDFList(allPdfs, restoreConversation);
          } else {
            conversationMode = restoreConversation;
            const msg = 'De acuerdo. Cuando quieras, pedí las facturas digitales de nuevo.';
            addMessage('assistant', msg);
            await speak(msg);
            if (restoreConversation) triggerListenWithCue();
          }
        };
        return;
      }
    }

    const pool = filtered.length > 0 ? filtered : allPdfs;

    if (pool.length === 1) {
      const pdf = pool[0];
      const msg = `Encontré una factura: ${pdf.cleanName}, descargada ${relativeDate(pdf.lastModified)}. La proceso ahora.`;
      addMessage('assistant', msg);
      await speak(msg);
      conversationMode = restoreConversation;
      await processDigitalPDF(pdf.file, pdf.cleanName);
      return;
    }

    await showAndReadPDFList(pool, restoreConversation);
  };
}

async function showAndReadPDFList(pdfs, restoreConversation) {
  renderDigitalOverlay(pdfs);
  const listText =
    `Encontré ${pdfs.length} factura${pdfs.length > 1 ? 's' : ''}. ` +
    pdfs.map((p, i) => `${ORDINAL_LABELS[i]}: ${p.cleanName}, descargada ${relativeDate(p.lastModified)}.`).join(' ') +
    ' ¿Cuál querés? Decí el número o parte del nombre.';
  addMessage('assistant', listText);
  await speak(listText);
  startDigitalSelectionLoop(pdfs, restoreConversation);
}

function startDigitalSelectionLoop(pdfs, restoreConversation) {
  let attempts = 0;

  function listen() {
    startListening();

    _digitalInvoiceCallback = async (transcript) => {
      _digitalInvoiceCallback = null;

      if (/cancel|ninguna|salir|cerrar|no\s+quiero/i.test(transcript.toLowerCase())) {
        closeDigitalOverlay();
        conversationMode = restoreConversation;
        const msg = 'Cancelado.';
        addMessage('assistant', msg);
        await speak(msg);
        if (restoreConversation) triggerListenWithCue();
        return;
      }

      const matched = matchPDFByVoice(transcript, pdfs);

      if (matched) {
        const idx = pdfs.indexOf(matched);
        highlightDigitalItem(idx);
        digitalStatus.textContent = `Seleccionada: ${matched.cleanName}`;
        addMessage('user', transcript);
        const confirmMsg = `Procesando factura de ${matched.cleanName}.`;
        addMessage('assistant', confirmMsg);
        await speak(confirmMsg);
        closeDigitalOverlay();
        conversationMode = restoreConversation;
        await processDigitalPDF(matched.file, matched.cleanName);
      } else {
        attempts++;
        if (attempts >= 3) {
          closeDigitalOverlay();
          conversationMode = restoreConversation;
          const msg = 'No pude identificar la selección. Usá el botón Factura para subir el PDF manualmente.';
          addMessage('assistant', msg);
          await speak(msg);
          if (restoreConversation) triggerListenWithCue();
          return;
        }
        digitalStatus.textContent = `No entendí (intento ${attempts}/3). Intentá de nuevo.`;
        const retryMsg = 'No entendí. Decí el número, por ejemplo "la primera", o el nombre de la empresa.';
        await speak(retryMsg);
        listen();
      }
    };
  }

  listen();
}

async function processDigitalPDF(file, cleanName) {
  const base64 = await new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result.split(',')[1]);
    reader.readAsDataURL(file);
  });

  imagePreview.src = '';
  state.pendingImage = { base64, mime: 'application/pdf', purpose: 'invoice' };
  textInput.value = `Factura digital: ${cleanName}`;
  setTimeout(() => sendMessage(), 400);
}

/* ─── Botón y comando de voz ────────────────────────────────────────────── */
const btnDigitalInvoice = document.getElementById('btn-digital-invoice');
btnDigitalInvoice.addEventListener('click', openDigitalInvoicesFlow);

function showDigitalVoiceOverlay() {
  // Reutiliza el voice-cam-overlay: el tap del usuario SÍ es gesto válido para el picker
  const overlay = document.getElementById('voice-camera-overlay');
  const text    = document.getElementById('voice-camera-text');
  text.textContent = 'Toca para acceder a tus facturas digitales';
  overlay.classList.remove('hidden');
  speak('Toca la pantalla para abrir tus facturas digitales.');

  function onTap() {
    overlay.classList.add('hidden');
    overlay.removeEventListener('click', onTap);
    openDigitalInvoicesFlow(); // tap real → gesto válido → picker se abre
  }
  overlay.addEventListener('click', onTap);
}

/* ─── Init ───────────────────────────────────────────────────────────────── */
if (window.speechSynthesis) speechSynthesis.onvoiceschanged = () => {};

if (isMobile()) {
  const tip = document.getElementById('desktop-tip');
  if (tip) tip.remove();
} else {
  // En desktop, saltear la pantalla de inicio y enfocar el input
  accessibilityOverlay.classList.add('hidden');
  textInput.focus();
}

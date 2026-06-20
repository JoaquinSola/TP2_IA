/* ─── Estado de la app ───────────────────────────────────────────────────── */
const state = {
  sessionId: null,
  pendingImage: null,
  isListening: false,
  recognition: null,
  cameraStream: null,
};

let conversationMode = false;

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
const btnLoadLogs             = document.getElementById('btn-load-logs');
const obsLogs                 = document.getElementById('obs-logs');
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

async function speak(text) {
  if (!text) return;

  // Desktop sin touch: Web Speech API (funciona síncronamente, sin bloqueo async)
  if (!isMobile() && window.speechSynthesis) {
    await new Promise((resolve) => {
      window.speechSynthesis.cancel();
      const utt = new SpeechSynthesisUtterance(text);
      utt.lang = 'es-AR';
      utt.rate = 1.3;
      const voices = speechSynthesis.getVoices();
      const esVoice = voices.find(v => v.lang.startsWith('es'));
      if (esVoice) utt.voice = esVoice;
      utt.onend = resolve;
      utt.onerror = resolve;
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
    if (!res.ok) return;
    const arrayBuffer = await res.arrayBuffer();
    const ctx = _getAudioContext();
    if (ctx.state === 'suspended') await ctx.resume();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

    await new Promise((resolve) => {
      _audioSource = ctx.createBufferSource();
      _audioSource.buffer = audioBuffer;
      _audioSource.playbackRate.value = 1.35;
      _audioSource.connect(ctx.destination);
      _audioSource.onended = () => { _audioSource = null; resolve(); };
      _audioSource.start(0);
    });
  } catch (err) {
    console.warn('TTS error:', err);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   STT — Web Speech API (desktop/HTTPS) / Dictado del teclado (mobile HTTP)
   En HTTP mobile no hay forma de grabar audio desde el navegador sin HTTPS.
   La alternativa confiable es el botón de micrófono del teclado virtual.
═══════════════════════════════════════════════════════════════════════════ */

// Palabras clave para disparar la cámara por voz
const INVOICE_CAMERA_WORDS = [
  'foto factura', 'fotografiar factura', 'foto de la factura', 'foto de factura',
  'sacar factura', 'capturar factura', 'escanear factura', 'analizar factura',
  'sacar foto factura', 'abrir camara factura', 'abrir cámara factura',
];
const BILLS_CAMERA_WORDS = [
  'foto billete', 'foto billetes', 'fotografiar billete', 'foto de los billetes',
  'foto del dinero', 'foto de la plata', 'foto de los billetes', 'foto billete',
  'sacar billete', 'capturar billete', 'sacar foto billete', 'abrir camara billete',
  'abrir cámara billete', 'foto del efectivo',
];

function matchVoiceCmd(text, keywords) {
  return keywords.some(kw => text.includes(kw));
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
  if (conversationMode) startListening();
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

  const displayText = text || (state.pendingImage?.purpose === 'bills' ? 'Foto de billetes enviada' : 'Imagen de factura enviada');
  addMessage('user', displayText, hasImage ? imagePreview.src : null);
  textInput.value = '';
  textInput.style.height = 'auto';
  setLoading(true);

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
  const fmt = (n) => '$' + Number(n).toLocaleString('es-AR');
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

/* ─── Observabilidad ─────────────────────────────────────────────────────── */
btnLoadLogs.addEventListener('click', async () => {
  if (!state.sessionId) { obsLogs.textContent = 'Aún no hay sesión activa.'; return; }
  try {
    const res = await fetch(`${API_BASE}/api/logs/${state.sessionId}`);
    const data = await res.json();
    obsLogs.textContent = JSON.stringify(data.logs, null, 2);
  } catch (err) { obsLogs.textContent = `Error: ${err.message}`; }
});

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
    'Bienvenido al asistente de pagos.',
    'Para analizar una factura, tocá el botón Factura, abajo a la izquierda.',
    'Para identificar billetes, tocá el botón Billetes, en el centro.',
    'Podés hablarme cuando quieras. Empecemos.'
  ].join(' ');

  await speak(welcome);
  triggerListenWithCue();
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

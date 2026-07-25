// HUD de ARGOS. Tres responsabilidades: WebSocket, captura de micrófono y el núcleo animado.
//
// El micrófono se captura con la Web Audio API y se envía como PCM int16 a 16 kHz.
// No se usa MediaRecorder (WebM/Opus) a propósito: decodificarlo en el servidor
// exigiría ffmpeg, y Whisper quiere exactamente este formato de todas formas.

const $ = (id) => document.getElementById(id);
const stream = $("stream");
let ws = null, provs = [], selected = null;

// ─────────────────────────── conversación ───────────────────────────

function addMsg(role, text) {
  const el = document.createElement("div");
  if (role === "sys" || role === "err") {
    el.className = "msg sys" + (role === "err" ? " err" : "");
    el.textContent = text;
  } else {
    el.className = "msg " + role;
    el.innerHTML = `<span class="who">${role === "user" ? "TÚ" : "ARGOS"}</span><div class="body"></div>`;
    el.querySelector(".body").textContent = text;
  }
  stream.appendChild(el);
  stream.scrollTop = stream.scrollHeight;
}

function node(id, on) { $(id).classList.toggle("on", !!on); }

function setState(s) {
  $("coreState").textContent = s;
  core.state = s;
}

// ─────────────────────────── WebSocket ───────────────────────────

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen = () => { node("node-link", true); setState("en espera"); };
  ws.onclose = () => {
    node("node-link", false); node("node-llm", false);
    setState("desconectado");
    setTimeout(connect, 2000);   // reconexión: el servidor se reinicia a menudo en desarrollo
  };

  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case "ready":
        node("node-llm", true);
        $("skills").innerHTML = m.skills.map(s => `<span class="chip">${s}</span>`).join("");
        break;
      case "state": setState(m.value); break;
      case "message": addMsg(m.role, m.text); break;
      case "transcript":
        if (m.text) addMsg("sys", `oído (${m.seconds}s → ${m.took_s}s): "${m.text}"`);
        else addMsg("sys", "no se oyó nada");
        node("node-stt", true);
        break;
      case "telemetry":
        $("t-lat").textContent = `${m.latency_ms} ms`;
        $("t-prov").textContent = m.provider;
        $("t-tok").textContent = m.tokens;
        $("t-usd").textContent = m.usd > 0 ? `$${m.usd.toFixed(5)}` : "local";
        $("t-skills").textContent = m.skills.length ? m.skills.join(", ") : "—";
        $("t-mem").textContent = m.recalled ? "recuperados" : "sin contexto";
        node("node-mem", m.recalled);
        if (m.nudges) addMsg("sys", `hubo que corregir al modelo ${m.nudges} vez(ces)`);
        break;
      case "speak": if ($("speak").checked) speak(m.text); break;
      case "providers": renderProviders(m.items); break;
      case "permission": askPermission(m); break;
      case "notice": addMsg("sys", m.message); break;
      case "error": addMsg("err", m.message); break;
    }
  };
}

function send(obj) { if (ws?.readyState === 1) ws.send(JSON.stringify(obj)); }

// ─────────────────────────── voz de salida ───────────────────────────
// Web Speech API: voces locales del sistema. Sin red, sin VRAM.
// Kokoro entrará aquí cuando haya espeak-ng disponible.

function speak(text) {
  if (!window.speechSynthesis) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "es-ES";
  const voz = speechSynthesis.getVoices().find(v => v.lang.startsWith("es"));
  if (voz) u.voice = voz;
  u.rate = 1.05;
  speechSynthesis.speak(u);
}

// ─────────────────────────── micrófono ───────────────────────────

let audioCtx = null, micStream = null, procesador = null, grabando = false;

async function startMic() {
  if (grabando) return;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch {
    addMsg("err", "sin acceso al micrófono. Revisa el permiso del navegador.");
    return;
  }

  // 16 kHz directo: evita remuestrear a mano y es lo que Whisper espera.
  audioCtx = new AudioContext({ sampleRate: 16000 });
  const fuente = audioCtx.createMediaStreamSource(micStream);
  procesador = audioCtx.createScriptProcessor(4096, 1, 1);

  procesador.onaudioprocess = (e) => {
    if (!grabando || ws?.readyState !== 1) return;
    const f32 = e.inputBuffer.getChannelData(0);
    const i16 = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    ws.send(i16.buffer);
  };

  fuente.connect(procesador);
  procesador.connect(audioCtx.destination);
  grabando = true;
  $("mic").classList.add("rec");
  send({ type: "audio_start" });
}

function stopMic() {
  if (!grabando) return;
  grabando = false;
  $("mic").classList.remove("rec");
  send({ type: "audio_end" });
  procesador?.disconnect();
  micStream?.getTracks().forEach(t => t.stop());
  audioCtx?.close();
  audioCtx = micStream = procesador = null;
}

// Mantener pulsado para hablar: más fiable que detectar el fin de frase.
const mic = $("mic");
mic.addEventListener("mousedown", startMic);
mic.addEventListener("touchstart", (e) => { e.preventDefault(); startMic(); });
["mouseup", "mouseleave", "touchend"].forEach(ev => mic.addEventListener(ev, stopMic));

// Barra espaciadora como push-to-talk, salvo escribiendo.
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && document.activeElement !== $("input")) {
    e.preventDefault(); startMic();
  }
});
document.addEventListener("keyup", (e) => { if (e.code === "Space") stopMic(); });

// ─────────────────────────── entrada de texto ───────────────────────────

$("input").addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const t = e.target.value.trim();
  if (!t) return;
  send({ type: "text", content: t, speak: $("speak").checked });
  e.target.value = "";
});

document.querySelectorAll("[data-say]").forEach(b =>
  b.addEventListener("click", () => send({ type: "text", content: b.dataset.say, speak: $("speak").checked }))
);

// ─────────────────────────── proveedores ───────────────────────────

function renderProviders(items) {
  provs = items;
  const activo = items.find(p => p.active);
  if (activo) $("brandSub").textContent = activo.label;
  selected = selected || activo?.name;

  $("provList").innerHTML = items.map(p => `
    <div class="prov ${p.active ? "active" : ""}" data-name="${p.name}">
      <span class="name">${p.label}</span>
      <span class="tag ${p.ready ? "" : "no-key"}">
        ${p.local ? "LOCAL" : "REMOTO"}${p.ready ? "" : " · falta clave"}
      </span>
    </div>`).join("");

  document.querySelectorAll(".prov").forEach(el =>
    el.addEventListener("click", () => {
      selected = el.dataset.name;
      const p = provs.find(x => x.name === selected);
      if (p.ready) send({ type: "set_provider", name: selected });
      else addMsg("sys", `${p.label} necesita la clave ${p.needs_secret}. Pégala abajo y guarda.`);
      document.querySelectorAll(".prov").forEach(x => x.classList.remove("active"));
      el.classList.add("active");
    })
  );
}

$("gear").addEventListener("click", () => $("cfg").showModal());
$("closeCfg").addEventListener("click", () => $("cfg").close());
$("saveSecret").addEventListener("click", () => {
  const p = provs.find(x => x.name === selected);
  const val = $("secretVal").value.trim();
  if (!p?.needs_secret) return addMsg("sys", "ese proveedor no necesita clave.");
  if (!val) return;
  send({ type: "set_secret", key: p.needs_secret, value: val });
  $("secretVal").value = "";
  send({ type: "set_provider", name: p.name });
});

// ─────────────────────────── permisos ───────────────────────────

function askPermission(m) {
  $("permText").textContent = `El agente quiere ejecutar: ${m.skill}`;
  $("permWhy").textContent = [m.reason, JSON.stringify(m.params)].filter(Boolean).join(" · ");
  $("perm").showModal();
}
$("permYes").addEventListener("click", () => { send({ type: "permission_reply", allowed: true }); $("perm").close(); });
$("permNo").addEventListener("click", () => { send({ type: "permission_reply", allowed: false }); $("perm").close(); });

// ─────────────────────────── núcleo animado ───────────────────────────
// Firma visual: tres anillos cuya energía sigue el estado del agente.

const core = { cv: $("core"), ctx: $("core").getContext("2d"), t: 0, state: "en espera" };

const ENERGIA = {
  "en espera": 0.25, escuchando: 0.85, transcribiendo: 0.6,
  pensando: 1.0, error: 0.4, desconectado: 0.05, conectando: 0.15,
};

function drawCore() {
  const { ctx, cv } = core, R = cv.width / 2;
  const e = ENERGIA[core.state] ?? 0.3;
  core.t += 0.012 + e * 0.02;

  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.translate(R, R);

  const color = core.state === "error" ? "255,92,114"
              : core.state === "desconectado" ? "93,112,133" : "53,224,208";

  for (let anillo = 0; anillo < 3; anillo++) {
    const base = R * (0.36 + anillo * 0.17);
    const dir = anillo % 2 ? -1 : 1;
    ctx.beginPath();
    for (let i = 0; i <= 90; i++) {
      const a = (i / 90) * Math.PI * 2;
      // Deformación senoidal: el anillo "respira" más cuanto mayor es la energía.
      const r = base + Math.sin(a * (3 + anillo) + core.t * dir * (1 + anillo)) * R * 0.045 * e;
      const [x, y] = [Math.cos(a) * r, Math.sin(a) * r];
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = `rgba(${color},${0.16 + e * 0.4 - anillo * 0.04})`;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  const nucleo = R * 0.12 * (1 + Math.sin(core.t * 2) * 0.09 * e);
  const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, nucleo * 2.6);
  grad.addColorStop(0, `rgba(${color},${0.55 * e + 0.1})`);
  grad.addColorStop(1, `rgba(${color},0)`);
  ctx.fillStyle = grad;
  ctx.beginPath(); ctx.arc(0, 0, nucleo * 2.6, 0, Math.PI * 2); ctx.fill();

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  requestAnimationFrame(drawCore);
}

speechSynthesis?.getVoices();   // precarga la lista de voces
drawCore();
connect();

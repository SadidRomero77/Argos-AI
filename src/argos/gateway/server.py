"""Gateway: HTTP + WebSocket que conecta la interfaz con el agente.

## Por qué el audio lo captura el navegador y no el servidor

WSL2 no expone los dispositivos de audio de forma fiable, y pelearse con eso
cuesta días. El navegador corre en Windows, ya tiene resuelto el permiso de
micrófono, y manda **PCM crudo a 16 kHz** — justo lo que Whisper quiere. Cero
dependencias de sistema, cero `ffmpeg`.

La síntesis de voz también vive en el navegador (Web Speech API, voces locales de
Windows). Es peor que Kokoro en calidad, pero Kokoro necesita `espeak-ng` y aquí
no hay sudo. La interfaz `speak` deja meterlo después sin tocar nada más.

## Protocolo

Cliente → servidor: JSON de control, o frames **binarios** con PCM int16.
Servidor → cliente: sólo JSON. El estado del agente se emite en cada transición
para que el HUD pueda animar el núcleo.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from argos.config import Settings, get_settings
from argos.harness.loop import AgentLoop
from argos.harness.permissions import PermissionGate, PermissionRequest
from argos.memory.embeddings import OllamaEmbedder
from argos.memory.store import MemoryStore
from argos.models.base import TaskKind
from argos.models.providers import ProviderCatalog, ProviderKind, ProviderSpec
from argos.models.router import ModelRouter
from argos.perception.stt import SpeechToText
from argos.tools.registry import default_registry
from argos.trace import Event, Tracer

STATIC = Path(__file__).parent / "static"


class WebApprover:
    """Aprueba permisos preguntando por la interfaz y esperando la respuesta.

    Si nadie contesta en `timeout`, **deniega**. Un agente esperando para siempre
    a una pestaña que el usuario cerró es peor que uno que se detiene.
    """

    def __init__(self, hub: Hub, timeout: float = 60.0) -> None:
        self.hub = hub
        self.timeout = timeout
        self._pendiente: asyncio.Future[bool] | None = None

    def resolve(self, allowed: bool) -> None:
        if self._pendiente is not None and not self._pendiente.done():
            self._pendiente.set_result(allowed)

    def ask(self, request: PermissionRequest) -> bool:
        """Síncrono porque el `PermissionGate` lo es; se puentea al bucle asíncrono."""
        loop = self.hub.loop
        if loop is None:
            return False

        async def preguntar() -> bool:
            self._pendiente = loop.create_future()
            await self.hub.send(
                {
                    "type": "permission",
                    "skill": request.skill,
                    "params": request.params,
                    "reason": request.reason,
                }
            )
            try:
                return await asyncio.wait_for(self._pendiente, timeout=self.timeout)
            except TimeoutError:
                return False
            finally:
                self._pendiente = None

        return asyncio.run_coroutine_threadsafe(preguntar(), loop).result()


class Hub:
    """Estado del gateway y difusión a las pestañas conectadas."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

        self.tracer = Tracer(settings=self.settings)
        self.catalog = ProviderCatalog(settings=self.settings)
        self.stt = SpeechToText()
        self.approver = WebApprover(self)
        self.gate = PermissionGate(
            approver=self.approver, tracer=self.tracer, settings=self.settings
        )
        self.registry = default_registry(gate=self.gate, tracer=self.tracer)
        self.memory: MemoryStore | None = None
        self.session = self.tracer.session_id
        self._history: list[dict[str, Any]] = []
        self._busy = False

    # ── memoria ───────────────────────────────────────────────────────────

    def open_memory(self) -> MemoryStore | None:
        """La memoria es opcional: sin servidor de embeddings el agente sigue
        funcionando, sólo sin continuidad entre sesiones."""
        if self.memory is not None:
            return self.memory
        try:
            embedder = OllamaEmbedder()
            ok, _ = embedder.health()
            if not ok:
                return None
            self.memory = MemoryStore(self.settings.paths.resolved("memory_db"), embedder)
        except Exception:
            return None
        return self.memory

    # ── difusión ──────────────────────────────────────────────────────────

    async def send(self, message: dict[str, Any]) -> None:
        muertos = []
        for ws in self.clients:
            try:
                await ws.send_json(message)
            except Exception:
                muertos.append(ws)
        for ws in muertos:
            self.clients.discard(ws)

    def send_sync(self, message: dict[str, Any]) -> None:
        """Desde un hilo de trabajo. El agente corre fuera del bucle asíncrono."""
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.send(message), self.loop)

    async def set_state(self, estado: str) -> None:
        await self.send({"type": "state", "value": estado})

    # ── proveedores ───────────────────────────────────────────────────────

    async def broadcast_providers(self) -> None:
        await self.send({"type": "providers", "items": self.catalog.status()})

    def build_loop(self) -> AgentLoop:
        provider = self.catalog.build()
        router = ModelRouter(
            providers=dict.fromkeys(TaskKind, provider),
            local_provider=provider,
            settings=self.settings,
            tracer=self.tracer,
        )
        return AgentLoop(
            router=router, registry=self.registry, tracer=self.tracer, settings=self.settings
        )

    # ── turno del agente ──────────────────────────────────────────────────

    def run_turn(self, texto: str) -> dict[str, Any]:
        """Ejecuta un turno completo. Se llama desde un hilo aparte."""
        inicio = time.monotonic()
        memoria = self.open_memory()

        # Recordar ANTES de pensar: lo relevante del pasado entra como contexto.
        contexto = ""
        if memoria is not None:
            try:
                recuerdos = memoria.recall_episodes(texto, limit=4)
                reglas = memoria.rules(limit=8)
                partes = []
                if recuerdos:
                    lineas = "\n".join(f"- {r.content}" for r in recuerdos)
                    partes.append(f"Recuerdas de conversaciones anteriores:\n{lineas}")
                if reglas:
                    lineas = "\n".join(f"- {r.render()}" for r in reglas)
                    partes.append(f"Reglas que has aprendido:\n{lineas}")
                contexto = "\n\n".join(partes)
            except Exception:
                contexto = ""

        mensaje = f"{contexto}\n\n---\n\n{texto}" if contexto else texto

        loop_agente = self.build_loop()
        resultado = loop_agente.run(mensaje, history=list(self._history))

        # Registrar DESPUÉS: la conversación de hoy es el recuerdo de mañana.
        if memoria is not None:
            try:
                memoria.remember_episode(self.session, "user", texto)
                if resultado.text:
                    memoria.remember_episode(self.session, "agent", resultado.text)
            except Exception:
                pass

        self._history.append({"role": "user", "content": texto})
        if resultado.text:
            self._history.append({"role": "assistant", "content": resultado.text})
        # La ventana se recorta aquí; el ContextCurator de Fase 1 hará esto mejor.
        self._history = self._history[-12:]

        return {
            "text": resultado.text,
            "skills": resultado.skill_calls,
            "iterations": resultado.iterations,
            "tokens": resultado.usage.total_tokens,
            "usd": round(resultado.usd, 6),
            "nudges": resultado.nudges,
            "stopped": str(resultado.stopped_because),
            "latency_ms": int((time.monotonic() - inicio) * 1000),
            "provider": self.catalog.active_name,
            "recalled": contexto != "",
        }


def create_app(settings: Settings | None = None) -> FastAPI:
    hub = Hub(settings)
    app = FastAPI(title="ARGOS Gateway")
    app.state.hub = hub

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        memoria = hub.open_memory()
        return {
            "ok": True,
            "provider": hub.catalog.active_name,
            "memory": memoria.stats() if memoria is not None else None,
            "stt_ready": hub.stt.ready,
            "skills": hub.registry.names,
        }

    @app.on_event("startup")
    async def startup() -> None:
        hub.loop = asyncio.get_running_loop()
        # Cargar Whisper al arrancar, no en la primera frase: si no, el usuario
        # habla y el sistema tarda diez segundos sin explicar por qué.
        await asyncio.to_thread(hub.stt.load)

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await ws.accept()
        hub.clients.add(ws)
        await hub.broadcast_providers()
        await ws.send_json({"type": "ready", "session": hub.session, "skills": hub.registry.names})

        buffer = bytearray()
        try:
            while True:
                evento = await ws.receive()

                if evento.get("type") == "websocket.disconnect":
                    break

                if (datos := evento.get("bytes")) is not None:
                    buffer.extend(datos)
                    continue

                if (texto := evento.get("text")) is None:
                    continue

                import json

                mensaje = json.loads(texto)
                tipo = mensaje.get("type")

                if tipo == "audio_end":
                    audio, buffer = bytes(buffer), bytearray()
                    await hub.set_state("transcribiendo")
                    transcripcion = await asyncio.to_thread(hub.stt.transcribe, audio)
                    await ws.send_json(
                        {
                            "type": "transcript",
                            "text": transcripcion.text,
                            "seconds": transcripcion.seconds_of_audio,
                            "took_s": transcripcion.duration_s,
                        }
                    )
                    if transcripcion.is_empty:
                        await hub.set_state("en espera")
                        continue
                    await _procesar(hub, transcripcion.text, hablar=True)

                elif tipo == "audio_start":
                    buffer = bytearray()
                    await hub.set_state("escuchando")

                elif tipo == "text":
                    contenido = (mensaje.get("content") or "").strip()
                    if contenido:
                        await _procesar(hub, contenido, hablar=bool(mensaje.get("speak")))

                elif tipo == "permission_reply":
                    hub.approver.resolve(bool(mensaje.get("allowed")))

                elif tipo == "set_provider":
                    try:
                        hub.catalog.activate(mensaje["name"])
                    except (KeyError, ValueError) as exc:
                        await ws.send_json({"type": "error", "message": str(exc)})
                    await hub.broadcast_providers()

                elif tipo == "set_secret":
                    hub.catalog.set_secret(mensaje["key"], mensaje["value"])
                    await hub.broadcast_providers()
                    await ws.send_json(
                        {"type": "notice", "message": f"Clave {mensaje['key']} guardada."}
                    )

                elif tipo == "add_provider":
                    try:
                        hub.catalog.add(
                            ProviderSpec(
                                name=mensaje["name"],
                                kind=ProviderKind(mensaje.get("kind", "openai")),
                                model=mensaje["model"],
                                base_url=mensaje.get("base_url"),
                                secret_key=mensaje.get("secret_key") or None,
                                label=mensaje.get("label", ""),
                            )
                        )
                    except Exception as exc:
                        await ws.send_json({"type": "error", "message": str(exc)})
                    await hub.broadcast_providers()

        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    return app


async def _procesar(hub: Hub, texto: str, hablar: bool) -> None:
    if hub._busy:
        await hub.send({"type": "notice", "message": "Estoy con la petición anterior."})
        return

    hub._busy = True
    try:
        await hub.send({"type": "message", "role": "user", "text": texto})
        await hub.set_state("pensando")

        resultado = await asyncio.to_thread(hub.run_turn, texto)

        await hub.send({"type": "message", "role": "agent", "text": resultado["text"]})
        await hub.send({"type": "telemetry", **resultado})
        if hablar and resultado["text"]:
            await hub.send({"type": "speak", "text": resultado["text"]})
        await hub.set_state("en espera")
    except Exception as exc:
        hub.tracer.emit(Event.ERROR, where="procesar", error=repr(exc))
        await hub.send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        await hub.set_state("error")
    finally:
        hub._busy = False


app = create_app()

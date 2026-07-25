"""Traza JSONL append-only: la única forma de depurar un agente sin adivinar.

Cada evento relevante (turno, llamada al modelo, decisión del router, ejecución de
skill, decisión de permiso, costo) se escribe como una línea JSON. El archivo es
append-only a propósito: es el registro de auditoría, no un log de aplicación.

Los secretos se redactan automáticamente antes de escribir. Esto no es defensa en
profundidad opcional — un embedding biométrico o una API key en una traza es una
fuga permanente, porque las trazas se conservan.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import SecretStr

from argos.config import Settings, get_settings

# Claves cuyo valor nunca se escribe, sin importar de dónde vengan.
_REDACT_KEYS = frozenset(
    {
        "api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "authorization",
        "authorization_token",
        "bus_key",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "client_secret",
        # Biométricos: el embedding facial ES el rostro (hay reconstrucción
        # demostrada vía difusión). Nunca sale del almacén cifrado.
        "embedding",
        "face_embedding",
    }
)

_REDACTED = "«redactado»"
_MAX_STR = 4000  # los valores largos se truncan; la traza no es un almacén de datos


class Event(StrEnum):
    """Tipos de evento. Añadir aquí antes de emitir uno nuevo."""

    SESSION_START = "session_start"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    MODEL_CALL = "model_call"
    ROUTER_DECISION = "router_decision"
    PERMISSION = "permission"
    SKILL_CALL = "skill_call"
    SKILL_RESULT = "skill_result"
    NUDGE = "nudge"
    BUDGET = "budget"
    ERROR = "error"


def _secret_values(settings: Settings) -> list[str]:
    """Valores literales de los secretos cargados, para redactarlos si se filtran."""
    values: list[str] = []
    for field in ("anthropic_api_key", "gemini_api_key", "bus_key"):
        secret: SecretStr | None = getattr(settings, field, None)
        if secret is not None:
            raw = secret.get_secret_value()
            if raw and len(raw) >= 8:  # ignora placeholders vacíos o triviales
                values.append(raw)
    return values


def redact(value: Any, secrets: list[str] | None = None, _depth: int = 0) -> Any:
    """Copia `value` con secretos y claves sensibles reemplazados.

    Redacta por dos vías complementarias: por *nombre de clave* (una clave llamada
    `api_key` se va, contenga lo que contenga) y por *valor literal* (si el secreto
    real aparece incrustado en cualquier cadena, se sustituye).
    """
    secrets = secrets if secrets is not None else []

    if _depth > 12:  # corta ciclos y estructuras patológicas
        return "«demasiado profundo»"

    if isinstance(value, SecretStr):
        return _REDACTED

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in _REDACT_KEYS:
                out[str(key)] = _REDACTED
            else:
                out[str(key)] = redact(item, secrets, _depth + 1)
        return out

    if isinstance(value, (list, tuple)):
        return [redact(item, secrets, _depth + 1) for item in value]

    if isinstance(value, str):
        for secret in secrets:
            if secret in value:
                value = value.replace(secret, _REDACTED)
        if len(value) > _MAX_STR:
            return value[:_MAX_STR] + f"…«truncado {len(value) - _MAX_STR} chars»"
        return value

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    return redact(str(value), secrets, _depth + 1)


class Tracer:
    """Escritor JSONL thread-safe. Un archivo por sesión."""

    def __init__(self, settings: Settings | None = None, session_id: str | None = None) -> None:
        self._settings = settings or get_settings()
        self._secrets = _secret_values(self._settings)
        self.session_id = session_id or uuid.uuid4().hex[:12]

        directory = self._settings.paths.resolved("traces")
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        self.path = directory / f"{stamp}-{self.session_id}.jsonl"

        self._lock = threading.Lock()
        self._seq = 0
        self.emit(Event.SESSION_START, pid=os.getpid())

    def emit(self, event: Event, **fields: Any) -> dict[str, Any]:
        """Escribe un evento. Devuelve el registro tal como quedó en disco."""
        with self._lock:
            self._seq += 1
            record = {
                "seq": self._seq,
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "session": self.session_id,
                "event": str(event),
                **redact(fields, self._secrets),
            }
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            return record

    def read_all(self) -> list[dict[str, Any]]:
        """Relee la traza. Para pruebas y para el dashboard de Fase 3."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def new_tracer(**kwargs: Any) -> Tracer:
    return Tracer(**kwargs)

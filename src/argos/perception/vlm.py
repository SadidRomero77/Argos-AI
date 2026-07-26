"""Modelo de visión-lenguaje: responder preguntas abiertas sobre una imagen.

Va por Ollama, que ya es la infraestructura del proyecto. Eso resuelve solo el
problema de VRAM: con 5 GB no caben a la vez el cerebro y el VLM, pero Ollama
carga y descarga modelos por su cuenta según se piden. Se paga en latencia —unos
segundos de intercambio— y a cambio no hay que orquestar nada.

## Dos detalles que costaron pruebas

**Sin mensaje de sistema.** Se probó con uno («Answer briefly and factually…») y
Moondream devolvía cadenas VACÍAS. La misma pregunta sin él responde bien. Es un
modelo de 1,8B: las instrucciones extra compiten con la pregunta en vez de
guiarla.

**Se usa `/api/generate`, no `/api/chat`.** Moondream no está entrenado para
conversación con roles; es un modelo de pregunta-respuesta sobre una imagen.

## Responde en inglés, y así se deja

Traducir aquí exigiría otra llamada a otro modelo. El agente ya habla español y
recibe la respuesta como dato para redactar la suya: la traducción sale gratis
en el turno que ya iba a ocurrir.

## Qué NO se le pide

Contar. Ante «¿cuántos dedos ves?» un VLM responde con seguridad y se equivoca a
menudo, porque contar es geometría disfrazada de pregunta. Para eso está
`perception.hands`, que cuenta con los puntos de la mano en código determinista.

La regla del proyecto aplicada a la vista: **el VLM describe, no mide.**
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import httpx

MODELO = "moondream"


@dataclass(slots=True)
class Vision:
    text: str
    seconds: float = 0.0
    model: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class VisionLanguageModel:
    """Cliente de un modelo de visión servido por Ollama."""

    def __init__(
        self,
        model: str = MODELO,
        base_url: str = "http://localhost:11434",
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        # Generoso: la primera pregunta puede incluir cargar el VLM y descargar el
        # cerebro para hacerle sitio en la VRAM.
        self._client = client or httpx.Client(timeout=httpx.Timeout(180.0, connect=3.0))

    def ask(self, jpeg: bytes, question: str) -> Vision:
        inicio = time.monotonic()
        respuesta = self._client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "stream": False,
                "prompt": question,
                "images": [base64.b64encode(jpeg).decode()],
            },
        )
        respuesta.raise_for_status()
        return Vision(
            text=(respuesta.json().get("response") or "").strip(),
            seconds=round(time.monotonic() - inicio, 2),
            model=self.model,
        )

    def describe(self, jpeg: bytes) -> Vision:
        """Descripción general. Es la pregunta con la que mejor responde."""
        return self.ask(jpeg, "Describe this image.")

    def health(self) -> tuple[bool, str]:
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            return False, f"Ollama no responde en {self.base_url} ({type(exc).__name__})"

        nombres = [m.get("name", "").split(":")[0] for m in r.json().get("models", [])]
        if self.model.split(":")[0] not in nombres:
            return False, f"falta el modelo de visión: ejecuta `ollama pull {self.model}`"
        return True, f"visión lista con '{self.model}'"

"""Texto a voz.

Dos implementaciones, y la elección tiene una contrapartida real que conviene
tener presente:

- **`EdgeTTS`** — voces neuronales de Microsoft. Calidad muy superior, 45 voces en
  español, gratis y sin clave. **Pero envía el texto a un servidor de Microsoft.**
  Para un agente que presume de local, eso importa: lo que ARGOS diga en voz alta
  sale de la máquina, aunque el razonamiento no lo haga.
- **`BrowserTTS`** — no genera nada; le dice al navegador que use su propia
  síntesis (voces SAPI locales de Windows). Peor calidad, pero **nada sale del
  equipo**.

Kokoro sería lo mejor de ambos —local y de calidad— pero necesita `espeak-ng`,
que aquí requiere sudo. Cuando esté disponible entra como tercera implementación
sin tocar nada más.

El audio generado son bytes MP3 que viajan por el WebSocket y reproduce el
navegador. No hace falta `ffmpeg` ni tarjeta de sonido en WSL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

# Voz por defecto: español de Colombia, que es donde está el usuario.
VOZ_POR_DEFECTO = "es-CO-GonzaloNeural"

# Marcas que se leen fatal en voz alta. El modelo escribe markdown aunque se le
# pida que no; limpiarlo aquí es más fiable que insistir en el prompt.
_MARKDOWN = re.compile(r"[*_`#]+")
_ENLACES = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BLOQUES_CODIGO = re.compile(r"```.*?```", re.DOTALL)


def para_voz(texto: str, max_chars: int = 1200) -> str:
    """Limpia el texto para que suene bien dicho en alto."""
    texto = _BLOQUES_CODIGO.sub(" (bloque de código) ", texto)
    texto = _ENLACES.sub(r"\1", texto)
    texto = _MARKDOWN.sub("", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    if len(texto) > max_chars:
        # Cortar en la última frase completa evita terminar a media palabra.
        recorte = texto[:max_chars]
        corte = max(recorte.rfind(". "), recorte.rfind("? "), recorte.rfind("! "))
        texto = recorte[: corte + 1] if corte > max_chars // 2 else recorte
    return texto


@dataclass(slots=True)
class Speech:
    audio: bytes
    mime: str = "audio/mpeg"
    voice: str = ""
    chars: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.audio


class TextToSpeech(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def local(self) -> bool:
        """True si nada sale de la máquina."""
        ...

    async def synth(self, text: str) -> Speech: ...


class EdgeTTS:
    """Voces neuronales de Microsoft. Requiere internet; el texto sale del equipo."""

    def __init__(self, voice: str = VOZ_POR_DEFECTO, rate: str = "+8%") -> None:
        self.voice = voice
        # Un pelín más rápido que el original: a velocidad nominal suena a
        # locución de aeropuerto, no a alguien hablando.
        self.rate = rate

    @property
    def name(self) -> str:
        return f"edge:{self.voice}"

    @property
    def local(self) -> bool:
        return False

    async def synth(self, text: str) -> Speech:
        limpio = para_voz(text)
        if not limpio:
            return Speech(audio=b"")

        import edge_tts

        comunicador = edge_tts.Communicate(limpio, self.voice, rate=self.rate)
        trozos = bytearray()
        async for evento in comunicador.stream():
            if evento["type"] == "audio":
                trozos.extend(evento["data"])
        return Speech(audio=bytes(trozos), voice=self.voice, chars=len(limpio))

    @staticmethod
    async def voices(prefix: str = "es") -> list[dict[str, str]]:
        import edge_tts

        todas = await edge_tts.list_voices()
        return [
            {"id": v["ShortName"], "gender": v["Gender"], "locale": v["Locale"]}
            for v in todas
            if v["Locale"].startswith(prefix)
        ]


class BrowserTTS:
    """No sintetiza: delega en el navegador. Nada sale del equipo."""

    @property
    def name(self) -> str:
        return "browser"

    @property
    def local(self) -> bool:
        return True

    async def synth(self, text: str) -> Speech:
        # Devuelve vacío a propósito: el gateway lo interpreta como "que hable el
        # navegador" y le manda el texto en vez del audio.
        return Speech(audio=b"", chars=len(para_voz(text)))

"""Voz a texto con faster-whisper.

**Corre en CPU a propósito.** Medido en este equipo (i7-12700H, 8 hilos):

    base   int8:  0,54 s para 5 s de audio   →  9,2x tiempo real
    small  int8:  1,48 s para 5 s de audio   →  3,4x tiempo real

`small` es de sobra rápido para una frase hablada y notablemente más preciso en
español. Usar CPU libera ~1 GB de VRAM para el cerebro y evita la dependencia de
`libcublas`/`libcudnn`, que en WSL2 falla con un error poco descriptivo.

El audio llega como **PCM crudo a 16 kHz mono** desde el navegador. No se usa
WebM ni MP3 a propósito: decodificarlos exigiría `ffmpeg`, que aquí requiere sudo,
y Whisper quiere exactamente este formato de todas formas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16_000

# Por debajo de esto es ruido de fondo, no habla. Whisper alucina frases enteras
# ("Gracias por ver el vídeo", "Subtítulos por la comunidad") cuando le das
# silencio, así que se corta antes de invocarlo.
_UMBRAL_RMS = 0.005
_MIN_SEGUNDOS = 0.4


@dataclass(slots=True)
class Transcription:
    text: str
    language: str = ""
    duration_s: float = 0.0
    seconds_of_audio: float = 0.0
    segments: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Convierte PCM int16 little-endian (lo que manda el navegador) a float32 [-1, 1]."""
    if not data:
        return np.zeros(0, dtype=np.float32)
    enteros = np.frombuffer(data, dtype="<i2")
    return (enteros.astype(np.float32) / 32768.0).copy()


def is_silence(audio: np.ndarray) -> bool:
    """¿Es silencio o demasiado corto para molestarse en transcribir?"""
    if audio.size < SAMPLE_RATE * _MIN_SEGUNDOS:
        return True
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) < _UMBRAL_RMS


class SpeechToText:
    """Envoltorio de faster-whisper. El modelo se carga perezosamente."""

    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 8,
        language: str | None = "es",
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.language = language
        self._model = None

    def load(self) -> None:
        """Carga el modelo. Explícito para poder hacerlo al arrancar el servidor
        y no en la primera frase del usuario, que sería una espera desconcertante."""
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )

    @property
    def ready(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray | bytes) -> Transcription:
        import time

        if isinstance(audio, (bytes, bytearray)):
            audio = pcm16_to_float32(bytes(audio))

        segundos = audio.size / SAMPLE_RATE
        if is_silence(audio):
            # Se devuelve vacío en vez de invocar al modelo: con silencio, Whisper
            # inventa frases con total confianza.
            return Transcription(text="", seconds_of_audio=segundos)

        self.load()
        inicio = time.monotonic()
        segmentos, info = self._model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,  # descarta tramos sin voz dentro del clip
            beam_size=1,  # greedy: la calidad extra del beam no compensa la latencia
        )
        piezas = [s.text.strip() for s in segmentos]

        return Transcription(
            text=" ".join(p for p in piezas if p).strip(),
            language=getattr(info, "language", "") or "",
            duration_s=round(time.monotonic() - inicio, 3),
            seconds_of_audio=round(segundos, 2),
            segments=piezas,
        )

"""Reconocimiento facial con InsightFace. El cierre del círculo de la identidad.

Un embedding facial de ArcFace son 512 números que entran **en la misma columna
`entity.embedding`** que la memoria ya tenía reservada desde el primer día. Por eso
esto no necesita esquema nuevo: una persona reconocida por su cara y una reconocida
por su nombre son la misma entidad.

Corre en **CPU**, igual que Whisper. Detectar y describir una cara tarda decenas de
milisegundos, así que la GPU no aporta nada y esos 300 MB de VRAM quedan para el
cerebro.

## Privacidad — no es un apartado decorativo

**Un embedding facial no es un dato anónimo: es el rostro.** Hay trabajo publicado
en 2026 que reconstruye caras realistas a partir del vector usando modelos de
difusión. Es un identificador biométrico permanente, y se trata como tal:

- Nunca sale de `var/memory.db`, que está fuera de git.
- **Nunca entra en un prompt**, ni siquiera con un proveedor local. El modelo
  recibe un nombre, jamás el vector.
- Nunca aparece en una traza: `argos.trace._REDACT_KEYS` incluye `embedding` y
  `face_embedding`, y hay una prueba que lo verifica.
- El enrolamiento es explícito. ARGOS no guarda la cara de nadie por su cuenta:
  hace falta que alguien diga su nombre.

## Umbral

Por debajo de `UMBRAL` se crea una identidad nueva en vez de asignar la más
parecida. Confundir a dos personas es mucho peor que no reconocer a una: lo
primero mezcla dos memorias y es difícil de deshacer, lo segundo se arregla
preguntando el nombre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ArcFace sobre buffalo_l. Con vectores normalizados, el coseno de dos fotos de la
# misma persona suele pasar de 0,5; entre personas distintas ronda 0,1.
UMBRAL = 0.42

# Caras más pequeñas que esto son ruido de fondo: alguien que pasa al fondo del
# encuadre, no la persona con la que se habla.
_AREA_MINIMA = 60 * 60


@dataclass(slots=True)
class Face:
    embedding: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float
    area: int

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2


class FaceRecognizer:
    """Detecta caras y produce embeddings. El modelo se carga perezosamente."""

    def __init__(self, model: str = "buffalo_l", det_size: int = 640) -> None:
        self.model_name = model
        self.det_size = det_size
        self._app: Any = None

    def load(self) -> None:
        """Carga explícita: la primera vez descarga ~300 MB y conviene controlarlo."""
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        # Sólo detección y reconocimiento. Los módulos de edad, género y puntos
        # faciales no se usan y cuestan memoria y tiempo de carga.
        app = FaceAnalysis(
            name=self.model_name,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
        self._app = app

    @property
    def ready(self) -> bool:
        return self._app is not None

    def detect(self, jpeg: bytes) -> list[Face]:
        """Caras en la imagen, de mayor a menor. La más grande es la más cercana."""
        import cv2

        imagen = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if imagen is None:
            return []

        self.load()
        caras = []
        for cara in self._app.get(imagen):
            x1, y1, x2, y2 = (int(v) for v in cara.bbox)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area < _AREA_MINIMA:
                continue
            vector = np.asarray(cara.normed_embedding, dtype=np.float32)
            caras.append(
                Face(
                    embedding=vector,
                    bbox=(x1, y1, x2, y2),
                    score=float(cara.det_score),
                    area=area,
                )
            )
        # La cara dominante primero: quien está hablando suele ser quien está cerca.
        caras.sort(key=lambda c: c.area, reverse=True)
        return caras

    def primary(self, jpeg: bytes) -> Face | None:
        caras = self.detect(jpeg)
        return caras[0] if caras else None

    def health(self) -> tuple[bool, str]:
        try:
            self.load()
        except Exception as exc:
            return False, f"no se pudo cargar '{self.model_name}': {type(exc).__name__}: {exc}"
        return True, f"reconocimiento facial listo ({self.model_name}, CPU)"

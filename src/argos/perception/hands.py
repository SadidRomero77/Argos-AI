"""Gestos y expresiones por geometría. Aquí NO interviene ningún modelo de lenguaje.

Es el ejemplo más limpio del principio rector en toda la percepción. Preguntarle a
un VLM «¿cuántos dedos ves?» produce una respuesta segura y a menudo equivocada,
porque **contar no es una tarea de lenguaje**: es mirar puntos y comparar
coordenadas. Lo mismo con «¿sonrío?».

MediaPipe da 21 puntos por mano y 52 *blendshapes* faciales. A partir de ahí:

- **Dedo extendido** ⟺ la punta está más lejos de la muñeca que su nudillo.
- **Sonrisa** ⟺ los blendshapes `mouthSmileLeft`/`Right` superan un umbral.

Ambos son números exactos que el agente sólo tiene que comunicar. Una alucinación
no puede convertir tres dedos en cinco ni una cara seria en una sonrisa.

Los modelos `.task` viven en `var/models/`, fuera de git por tamaño. Se descargan
con `./dev models`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from argos.config import ROOT

MODELOS = ROOT / "var" / "models"
MANO_TASK = MODELOS / "hand_landmarker.task"
CARA_TASK = MODELOS / "face_landmarker.task"

MUNECA = 0
PUNTAS = {"pulgar": 4, "indice": 8, "medio": 12, "anular": 16, "menique": 20}
NUDILLOS = {"pulgar": 2, "indice": 6, "medio": 10, "anular": 14, "menique": 18}
BASE_INDICE = 5

# Por debajo de esto la comisura sube por hablar o por gesto, no por sonreír.
UMBRAL_SONRISA = 0.35


@dataclass(slots=True)
class Hand:
    fingers: int
    extended: list[str] = field(default_factory=list)
    label: str = ""
    confidence: float = 0.0

    @property
    def gesture(self) -> str:
        """Nombre coloquial cuando lo hay: más natural que decir un número."""
        ext = set(self.extended)
        if not ext:
            return "puño"
        if ext == {"indice", "medio"}:
            return "señal de victoria"
        if ext == {"pulgar"}:
            return "pulgar arriba"
        if ext == {"indice"}:
            return "señalando"
        if ext == {"pulgar", "menique"}:
            return "gesto de llamar"
        if len(ext) == 5:
            return "mano abierta"
        return ""


@dataclass(slots=True)
class Expression:
    smiling: bool
    smile_score: float = 0.0
    eyes_closed: bool = False
    mouth_open: bool = False


def _a_imagen_mp(jpeg: bytes) -> Any:
    import cv2
    import mediapipe as mp

    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    # MediaPipe espera RGB; OpenCV decodifica en BGR.
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


class HandCounter:
    """Cuenta dedos extendidos. Carga el modelo perezosamente."""

    def __init__(self, max_hands: int = 2, min_confidence: float = 0.5) -> None:
        self.max_hands = max_hands
        self.min_confidence = min_confidence
        self._detector: Any = None

    def load(self) -> None:
        if self._detector is not None:
            return
        if not MANO_TASK.is_file():
            raise FileNotFoundError(f"falta {MANO_TASK}; ejecuta `./dev models`")

        from mediapipe.tasks import python as mtp
        from mediapipe.tasks.python import vision

        self._detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mtp.BaseOptions(model_asset_path=str(MANO_TASK)),
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.min_confidence,
            )
        )

    @property
    def ready(self) -> bool:
        return self._detector is not None

    @staticmethod
    def _extendidos(puntos: np.ndarray, es_derecha: bool) -> list[str]:
        extendidos = []
        muneca = puntos[MUNECA]

        for dedo in ("indice", "medio", "anular", "menique"):
            punta, nudillo = puntos[PUNTAS[dedo]], puntos[NUDILLOS[dedo]]
            # Distancias a la muñeca, no coordenada Y: así el conteo funciona con
            # la mano girada, no sólo apuntando hacia arriba.
            if np.linalg.norm(punta - muneca) > np.linalg.norm(nudillo - muneca):
                extendidos.append(dedo)

        # El pulgar se dobla hacia dentro, no hacia abajo: su criterio es cuánto se
        # separa lateralmente de la base del índice.
        punta_pulgar, base = puntos[PUNTAS["pulgar"]], puntos[BASE_INDICE]
        if (punta_pulgar[0] > base[0]) if es_derecha else (punta_pulgar[0] < base[0]):
            extendidos.append("pulgar")

        return extendidos

    def count(self, jpeg: bytes) -> list[Hand]:
        imagen = _a_imagen_mp(jpeg)
        if imagen is None:
            return []

        self.load()
        resultado = self._detector.detect(imagen)
        if not resultado.hand_landmarks:
            return []

        manos: list[Hand] = []
        for i, marcas in enumerate(resultado.hand_landmarks):
            puntos = np.array([[p.x, p.y, p.z] for p in marcas], dtype=np.float32)

            etiqueta, confianza = "", 0.0
            if i < len(resultado.handedness) and resultado.handedness[i]:
                clase = resultado.handedness[i][0]
                etiqueta, confianza = clase.category_name, float(clase.score)

            # MediaPipe etiqueta desde el punto de vista de la cámara, que ve la
            # escena en espejo respecto a quien está delante.
            extendidos = self._extendidos(puntos, es_derecha=etiqueta == "Left")
            manos.append(
                Hand(
                    fingers=len(extendidos),
                    extended=extendidos,
                    label=etiqueta,
                    confidence=round(confianza, 2),
                )
            )
        return manos

    def health(self) -> tuple[bool, str]:
        try:
            self.load()
        except Exception as exc:
            return False, f"conteo de dedos no disponible: {exc}"
        return True, "conteo de dedos listo (MediaPipe Hands, CPU)"


class ExpressionReader:
    """Sonrisa, ojos y boca a partir de blendshapes faciales."""

    def __init__(self) -> None:
        self._detector: Any = None

    def load(self) -> None:
        if self._detector is not None:
            return
        if not CARA_TASK.is_file():
            raise FileNotFoundError(f"falta {CARA_TASK}; ejecuta `./dev models`")

        from mediapipe.tasks import python as mtp
        from mediapipe.tasks.python import vision

        self._detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mtp.BaseOptions(model_asset_path=str(CARA_TASK)),
                output_face_blendshapes=True,  # es lo único que interesa aquí
                num_faces=1,
            )
        )

    def read(self, jpeg: bytes) -> Expression | None:
        imagen = _a_imagen_mp(jpeg)
        if imagen is None:
            return None

        self.load()
        resultado = self._detector.detect(imagen)
        if not resultado.face_blendshapes:
            return None

        puntuaciones = {b.category_name: b.score for b in resultado.face_blendshapes[0]}
        # Se promedian los dos lados: una sonrisa asimétrica sigue siendo sonrisa,
        # pero quedarse con un solo lado da falsos positivos al hablar.
        sonrisa = (
            puntuaciones.get("mouthSmileLeft", 0.0) + puntuaciones.get("mouthSmileRight", 0.0)
        ) / 2
        ojos = (puntuaciones.get("eyeBlinkLeft", 0.0) + puntuaciones.get("eyeBlinkRight", 0.0)) / 2

        return Expression(
            smiling=sonrisa >= UMBRAL_SONRISA,
            smile_score=round(sonrisa, 3),
            eyes_closed=ojos >= 0.5,
            mouth_open=puntuaciones.get("jawOpen", 0.0) >= 0.35,
        )

    def health(self) -> tuple[bool, str]:
        try:
            self.load()
        except Exception as exc:
            return False, f"lectura de expresión no disponible: {exc}"
        return True, "lectura de expresión lista (MediaPipe FaceLandmarker, CPU)"


def descargar_modelos(destino: Path | None = None) -> list[str]:
    """Descarga los .task si faltan. Los usa `./dev models`."""
    import httpx

    destino = destino or MODELOS
    destino.mkdir(parents=True, exist_ok=True)
    base = "https://storage.googleapis.com/mediapipe-models"
    rutas = {
        f"{nombre}_landmarker.task": (
            f"{base}/{nombre}_landmarker/{nombre}_landmarker/float16/1/{nombre}_landmarker.task"
        )
        for nombre in ("hand", "face")
    }

    hechos = []
    with httpx.Client(timeout=120.0, follow_redirects=True) as cliente:
        for nombre, url in rutas.items():
            ruta = destino / nombre
            if ruta.is_file():
                hechos.append(f"{nombre} (ya estaba)")
                continue
            respuesta = cliente.get(url)
            respuesta.raise_for_status()
            ruta.write_bytes(respuesta.content)
            hechos.append(f"{nombre} ({len(respuesta.content) // 1024} KB)")
    return hechos

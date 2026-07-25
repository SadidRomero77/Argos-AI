"""Embeddings: convierten texto en vectores para buscar por significado.

Por qué hace falta una interfaz y no una llamada directa: **la suite rápida no
puede depender de un servidor**. `FakeEmbedder` produce vectores deterministas sin
red, así que toda la lógica de memoria se prueba sin infraestructura, y sólo las
pruebas marcadas `needs_llm` usan el modelo real.

Modelo por defecto: `bge-m3`, multilingüe. Un embedder sólo-inglés degradaría
notablemente la recuperación en un proyecto cuyo contenido está en español.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import httpx
import numpy as np


class Embedder(Protocol):
    """Contrato mínimo. El almacén de memoria nunca sabe quién lo implementa."""

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Devuelve una matriz (n, dimensions) de float32, ya normalizada."""
        ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Normaliza a norma 1 para que el coseno sea un simple producto escalar.

    Normalizar al guardar evita recalcular la norma en cada búsqueda, que es la
    operación que se repite miles de veces.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    normas = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Un vector nulo se deja tal cual: dividir por cero daría NaN y envenenaría
    # todas las búsquedas posteriores en silencio.
    normas[normas == 0] = 1.0
    return (matrix / normas).astype(np.float32)


class OllamaEmbedder:
    """Embeddings vía el endpoint nativo de Ollama."""

    def __init__(
        self,
        model: str = "bge-m3",
        base_url: str = "http://localhost:11434",
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self._dimensions: int | None = None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(120.0, connect=3.0),
        )

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Se descubre con una llamada real: hardcodearlo se desincroniza en
            # cuanto alguien cambia de modelo, y el desajuste sólo se manifiesta
            # como búsquedas que no encuentran nada.
            self._dimensions = int(self.embed(["dimensión"]).shape[1])
        return self._dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimensions or 0), dtype=np.float32)

        respuesta = self._client.post(
            f"{self.base_url}/api/embed", json={"model": self.model, "input": texts}
        )
        respuesta.raise_for_status()
        vectores = respuesta.json().get("embeddings") or []
        if len(vectores) != len(texts):
            raise RuntimeError(
                f"'{self.model}' devolvió {len(vectores)} vectores para {len(texts)} textos"
            )
        return normalize(np.asarray(vectores, dtype=np.float32))

    def health(self) -> tuple[bool, str]:
        try:
            self.embed(["prueba"])
        except httpx.HTTPError as exc:
            return False, (
                f"no hay servidor de embeddings en {self.base_url} ({type(exc).__name__}). "
                "Arranca Ollama con `./dev up`."
            )
        except Exception as exc:
            return (
                False,
                f"el modelo '{self.model}' falló: {exc}. Prueba `ollama pull {self.model}`",
            )
        return True, f"embeddings OK con '{self.model}' ({self.dimensions} dimensiones)"


class FakeEmbedder:
    """Vectores deterministas derivados del texto. Sólo para pruebas.

    No captura significado —textos distintos dan vectores no relacionados— pero sí
    garantiza las dos propiedades que la lógica de memoria necesita verificar: el
    mismo texto da siempre el mismo vector, y textos distintos dan vectores
    distintos. Con eso se puede probar todo el almacén sin red.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimensions), dtype=np.float32)
        vectores = []
        for texto in texts:
            # La semilla se deriva de un hash del texto COMPLETO. No se usa hash()
            # porque Python lo aleatoriza entre procesos y las pruebas dejarían de
            # ser reproducibles. Tampoco vale tomar los primeros bytes: con
            # little-endian y módulo 2^32 la semilla acababa dependiendo sólo de
            # los 4 primeros caracteres, y "rostro-ana" y "rostro-de-otro" daban
            # vectores idénticos — un falso negativo que invalidaba en silencio
            # cualquier prueba de similitud.
            digest = hashlib.blake2b(texto.encode("utf-8"), digest_size=4).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "big"))
            vectores.append(rng.standard_normal(self._dimensions))
        return normalize(np.asarray(vectores, dtype=np.float32))


class LexicalEmbedder:
    """Bolsa de palabras con hashing. Sin modelo, sin red, sin dependencias.

    No entiende significado —"can" y "perro" son ajenos para él— pero sí captura
    solapamiento de vocabulario, que basta para dos cosas:

    1. **Respaldo real**: si no hay servidor de embeddings, la memoria sigue
       funcionando en modo degradado en vez de desaparecer.
    2. **Pruebas de relevancia**: textos que comparten palabras dan similitud
       alta y textos ajenos la dan baja, que es justo la propiedad que hay que
       verificar. `FakeEmbedder` no sirve para eso: sus vectores son aleatorios,
       así que dos frases del mismo tema salen tan distintas como dos ajenas.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @staticmethod
    def _tokens(texto: str) -> list[str]:
        limpio = "".join(c.lower() if c.isalnum() else " " for c in texto)
        # Las palabras de una letra son ruido y las tildes ya las normaliza casefold.
        return [t for t in limpio.split() if len(t) > 1]

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimensions), dtype=np.float32)
        matriz = np.zeros((len(texts), self._dimensions), dtype=np.float32)
        for fila, texto in enumerate(texts):
            for token in self._tokens(texto):
                indice = (
                    int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big")
                    % self._dimensions
                )
                matriz[fila, indice] += 1.0
        return normalize(matriz)

"""Memoria del agente: episódica, semántica, procedimental y entidades.

Los tres niveles replican la taxonomía en la que convergió el ecosistema en 2026,
que a su vez viene de décadas de ciencia cognitiva:

- **Episódica** — qué pasó y cuándo. Se recupera por *similitud + recencia*: lo
  parecido importa, pero lo reciente importa más.
- **Semántica** — hechos y preferencias estables ("prefiere respuestas cortas").
  Se deduplica por `(sujeto, hecho)`: repetir un dato no crea uno nuevo.
- **Procedimental** — reglas aprendidas ("cuando X, hacer Y"). Se inyectan en el
  prompt de sistema, así que son las únicas que cambian el comportamiento base.

**La tabla `entity` existe desde el día uno aunque hoy sólo se pueble con texto.**
En Fase 2 llega el embedding facial de InsightFace y entra en la misma columna, sin
migrar nada: una persona, un objeto y un lugar son la misma estructura con distinto
`kind`.

## Por qué búsqueda por fuerza bruta y no un índice vectorial

Se descartó `sqlite-vec`: exige cargar una extensión nativa, que es una fuente de
fallos de despliegue (y este proyecto acabará corriendo en una Raspberry Pi). Con
numpy, buscar entre 10.000 recuerdos de 1024 dimensiones son ~10 ms — irrelevante
frente a los segundos que tarda el modelo. La interfaz permite cambiarlo si algún
día el volumen lo justifica; hoy no lo justifica.

## Aviso de privacidad

Los embeddings biométricos (Fase 2) **no salen de este archivo**: nunca en un
prompt a una API externa, nunca en una traza. Ver `argos.trace._REDACT_KEYS`.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from argos.memory.embeddings import Embedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS entity (
    id            INTEGER PRIMARY KEY,
    kind          TEXT NOT NULL,          -- persona | objeto | lugar | tema
    label         TEXT NOT NULL,
    embedding     BLOB,                   -- float32; en Fase 2, el vector facial
    meta          TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL,
    UNIQUE(kind, label)
);

CREATE TABLE IF NOT EXISTS episodic (
    id          INTEGER PRIMARY KEY,
    session     TEXT NOT NULL,
    role        TEXT NOT NULL,            -- user | agent
    content     TEXT NOT NULL,
    embedding   BLOB,
    entity_id   INTEGER REFERENCES entity(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodic_creado ON episodic(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_entidad ON episodic(entity_id);

CREATE TABLE IF NOT EXISTS semantic (
    id          INTEGER PRIMARY KEY,
    subject     TEXT NOT NULL,
    fact        TEXT NOT NULL,
    embedding   BLOB,
    entity_id   INTEGER REFERENCES entity(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE(subject, fact)
);

CREATE TABLE IF NOT EXISTS procedural (
    id            INTEGER PRIMARY KEY,
    trigger       TEXT NOT NULL,
    action        TEXT NOT NULL,
    times_applied INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    UNIQUE(trigger, action)
);
"""


class EntityKind(StrEnum):
    PERSONA = "persona"
    OBJETO = "objeto"
    LUGAR = "lugar"
    TEMA = "tema"


class Entity(BaseModel):
    id: int
    kind: EntityKind
    label: str
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    last_seen_at: str


class Episode(BaseModel):
    id: int
    session: str
    role: str
    content: str
    entity_id: int | None = None
    created_at: str
    score: float = 0.0


class Fact(BaseModel):
    id: int
    subject: str
    fact: str
    entity_id: int | None = None
    updated_at: str
    score: float = 0.0


class Rule(BaseModel):
    id: int
    trigger: str
    action: str
    times_applied: int = 0

    def render(self) -> str:
        return f"Cuando {self.trigger}, {self.action}."


def _ahora() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _to_blob(vector: np.ndarray | None) -> bytes | None:
    return None if vector is None else np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob: bytes | None, dimensions: int) -> np.ndarray | None:
    if not blob:
        return None
    vector = np.frombuffer(blob, dtype=np.float32)
    # Un vector de otra dimensión viene de un modelo de embeddings distinto. Se
    # ignora en vez de reventar: cambiar de modelo no debe romper el arranque,
    # sólo dejar inaccesibles los recuerdos antiguos hasta reindexar.
    return vector if vector.shape[0] == dimensions else None


class MemoryStore:
    """Los tres niveles de memoria sobre SQLite. Sin servidor, sin dependencias nativas."""

    def __init__(self, path: Path | str, embedder: Embedder) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── entidades ─────────────────────────────────────────────────────────

    def upsert_entity(
        self,
        kind: EntityKind | str,
        label: str,
        embedding: np.ndarray | None = None,
        **meta: Any,
    ) -> int:
        """Crea la entidad o actualiza su `last_seen_at`. Devuelve el id."""
        kind = str(kind)
        ahora = _ahora()
        fila = self._db.execute(
            "SELECT id, meta FROM entity WHERE kind = ? AND label = ?", (kind, label)
        ).fetchone()

        if fila is None:
            cursor = self._db.execute(
                "INSERT INTO entity (kind, label, embedding, meta, created_at, last_seen_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (kind, label, _to_blob(embedding), json.dumps(meta), ahora, ahora),
            )
            self._db.commit()
            return int(cursor.lastrowid)

        fusionado = {**json.loads(fila["meta"]), **meta}
        if embedding is not None:
            self._db.execute(
                "UPDATE entity SET last_seen_at = ?, meta = ?, embedding = ? WHERE id = ?",
                (ahora, json.dumps(fusionado), _to_blob(embedding), fila["id"]),
            )
        else:
            self._db.execute(
                "UPDATE entity SET last_seen_at = ?, meta = ? WHERE id = ?",
                (ahora, json.dumps(fusionado), fila["id"]),
            )
        self._db.commit()
        return int(fila["id"])

    def get_entity(self, kind: EntityKind | str, label: str) -> Entity | None:
        fila = self._db.execute(
            "SELECT * FROM entity WHERE kind = ? AND label = ?", (str(kind), label)
        ).fetchone()
        return None if fila is None else self._fila_a_entidad(fila)

    def find_entity_by_embedding(
        self, embedding: np.ndarray, kind: EntityKind | str, threshold: float = 0.75
    ) -> tuple[Entity, float] | None:
        """La entidad más parecida por vector. Es la base del reconocimiento facial.

        Devuelve None si nada supera el umbral: **crear una identidad nueva es
        preferible a confundir a dos personas**.
        """
        filas = self._db.execute(
            "SELECT * FROM entity WHERE kind = ? AND embedding IS NOT NULL", (str(kind),)
        ).fetchall()
        mejor, mejor_score = None, -1.0
        objetivo = np.asarray(embedding, dtype=np.float32).ravel()
        for fila in filas:
            vector = _from_blob(fila["embedding"], objetivo.shape[0])
            if vector is None:
                continue
            score = float(np.dot(objetivo, vector))
            if score > mejor_score:
                mejor, mejor_score = fila, score
        if mejor is None or mejor_score < threshold:
            return None
        return self._fila_a_entidad(mejor), mejor_score

    def forget_entity(self, kind: EntityKind | str, label: str) -> int:
        """Borra la entidad y TODO lo asociado. Sin fricción: es un derecho, no un favor."""
        cursor = self._db.execute(
            "DELETE FROM entity WHERE kind = ? AND label = ?", (str(kind), label)
        )
        self._db.commit()
        return cursor.rowcount

    @staticmethod
    def _fila_a_entidad(fila: sqlite3.Row) -> Entity:
        return Entity(
            id=fila["id"],
            kind=fila["kind"],
            label=fila["label"],
            meta=json.loads(fila["meta"]),
            created_at=fila["created_at"],
            last_seen_at=fila["last_seen_at"],
        )

    # ── episódica ─────────────────────────────────────────────────────────

    def remember_episode(
        self, session: str, role: str, content: str, entity_id: int | None = None
    ) -> int:
        vector = self.embedder.embed([content])[0]
        cursor = self._db.execute(
            "INSERT INTO episodic (session, role, content, embedding, entity_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (session, role, content, _to_blob(vector), entity_id, _ahora()),
        )
        self._db.commit()
        return int(cursor.lastrowid)

    def recall_episodes(
        self,
        query: str,
        limit: int = 5,
        entity_id: int | None = None,
        recency_weight: float = 0.3,
        min_similarity: float = 0.45,
    ) -> list[Episode]:
        """Recupera por similitud **y** recencia, descartando lo irrelevante.

        `min_similarity` es un filtro sobre la similitud **cruda**, aplicado antes
        de mezclar con la recencia. Sin él, recordar siempre devuelve los `limit`
        más parecidos aunque ninguno venga a cuento: con un solo recuerdo guardado
        ("hay 33 archivos .py"), un saludo como "hola" lo recuperaba y el agente
        respondía al saludo hablando de archivos. Devolver *nada* es la respuesta
        correcta cuando nada es pertinente.

        `recency_weight` es cuánto pesa lo reciente frente a lo parecido: sólo por
        similitud, el agente desentierra conversaciones de hace meses ignorando lo
        de hace cinco minutos.
        """
        sql = "SELECT * FROM episodic WHERE embedding IS NOT NULL"
        params: list[Any] = []
        if entity_id is not None:
            sql += " AND entity_id = ?"
            params.append(entity_id)
        filas = self._db.execute(sql, params).fetchall()
        if not filas:
            return []

        objetivo = self.embedder.embed([query])[0]
        dims = objetivo.shape[0]

        candidatos: list[tuple[sqlite3.Row, float]] = []
        for fila in filas:
            vector = _from_blob(fila["embedding"], dims)
            if vector is None:
                continue
            similitud = float(np.dot(objetivo, vector))
            # Filtro sobre la similitud CRUDA, antes de reescalar: tras el
            # reescalado el mejor candidato siempre vale 1,0 aunque sea basura.
            if similitud >= min_similarity:
                candidatos.append((fila, similitud))
        if not candidatos:
            return []

        # La recencia se calcula por posición, no por reloj: así el resultado no
        # depende de cuánto tiempo lleve el agente parado.
        orden_temporal = sorted(candidatos, key=lambda par: par[0]["created_at"])
        posicion = {id(fila): i for i, (fila, _) in enumerate(orden_temporal)}
        n = max(len(candidatos) - 1, 1)

        # La similitud se re-escala al rango observado ANTES de mezclarla con la
        # recencia. Sin esto se combinan dos señales en escalas incomparables:
        # bge-m3 devuelve similitudes en una banda estrecha (~0,4 a 0,9) mientras la
        # recencia ocupa el 0 a 1 completo, así que la recencia aplasta diferencias
        # de similitud reales. Se observó a "me gustan las respuestas cortas"
        # (lo más reciente) ganarle a "la GPU es una RTX 3060" ante la pregunta
        # "¿qué tarjeta gráfica tengo?".
        sims = [sim for _, sim in candidatos]
        piso, techo = min(sims), max(sims)
        rango = techo - piso

        def escalada(sim: float) -> float:
            # Todo igual de parecido: la similitud no discrimina, decide la recencia.
            return 1.0 if rango < 1e-6 else (sim - piso) / rango

        puntuados = [
            (
                fila,
                (1 - recency_weight) * escalada(sim) + recency_weight * (posicion[id(fila)] / n),
            )
            for fila, sim in candidatos
        ]
        puntuados.sort(key=lambda par: par[1], reverse=True)

        return [
            Episode(
                id=fila["id"],
                session=fila["session"],
                role=fila["role"],
                content=fila["content"],
                entity_id=fila["entity_id"],
                created_at=fila["created_at"],
                score=round(score, 4),
            )
            for fila, score in puntuados[:limit]
        ]

    # ── semántica ─────────────────────────────────────────────────────────

    def remember_fact(self, subject: str, fact: str, entity_id: int | None = None) -> int:
        """Guarda un hecho. Repetirlo actualiza el existente en vez de duplicarlo."""
        ahora = _ahora()
        existente = self._db.execute(
            "SELECT id FROM semantic WHERE subject = ? AND fact = ?", (subject, fact)
        ).fetchone()
        if existente is not None:
            self._db.execute(
                "UPDATE semantic SET updated_at = ? WHERE id = ?", (ahora, existente["id"])
            )
            self._db.commit()
            return int(existente["id"])

        vector = self.embedder.embed([f"{subject}: {fact}"])[0]
        cursor = self._db.execute(
            "INSERT INTO semantic (subject, fact, embedding, entity_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (subject, fact, _to_blob(vector), entity_id, ahora, ahora),
        )
        self._db.commit()
        return int(cursor.lastrowid)

    def recall_facts(
        self,
        query: str,
        limit: int = 5,
        subject: str | None = None,
        min_similarity: float = 0.4,
    ) -> list[Fact]:
        """Hechos relevantes. Como en los episodios, lo irrelevante no se devuelve."""
        sql = "SELECT * FROM semantic WHERE embedding IS NOT NULL"
        params: list[Any] = []
        if subject is not None:
            sql += " AND subject = ?"
            params.append(subject)
        filas = self._db.execute(sql, params).fetchall()
        if not filas:
            return []

        objetivo = self.embedder.embed([query])[0]
        dims = objetivo.shape[0]
        puntuados = []
        for fila in filas:
            vector = _from_blob(fila["embedding"], dims)
            if vector is None:
                continue
            similitud = float(np.dot(objetivo, vector))
            if similitud >= min_similarity:
                puntuados.append((fila, similitud))
        puntuados.sort(key=lambda par: par[1], reverse=True)

        return [
            Fact(
                id=fila["id"],
                subject=fila["subject"],
                fact=fila["fact"],
                entity_id=fila["entity_id"],
                updated_at=fila["updated_at"],
                score=round(score, 4),
            )
            for fila, score in puntuados[:limit]
        ]

    def facts_about(self, subject: str, limit: int = 40) -> list[Fact]:
        """TODOS los hechos de un sujeto, sin filtro de similitud.

        La búsqueda vectorial falla justo donde más importa: los hechos se guardan
        en tercera persona ("Sadid es físico") y el usuario pregunta en primera
        ("¿quién soy?"), así que los vectores no casan. El perfil del usuario no
        debe depender de que acierte un embedding: se inyecta siempre.
        """
        filas = self._db.execute(
            "SELECT * FROM semantic WHERE subject = ? ORDER BY id LIMIT ?", (subject, limit)
        ).fetchall()
        return [
            Fact(
                id=f["id"],
                subject=f["subject"],
                fact=f["fact"],
                entity_id=f["entity_id"],
                updated_at=f["updated_at"],
            )
            for f in filas
        ]

    def forget_fact(self, fact_id: int) -> bool:
        cursor = self._db.execute("DELETE FROM semantic WHERE id = ?", (fact_id,))
        self._db.commit()
        return cursor.rowcount > 0

    # ── procedimental ─────────────────────────────────────────────────────

    def learn_rule(self, trigger: str, action: str) -> int:
        """Regla aprendida. Se inyecta en el prompt: cambia el comportamiento base."""
        ahora = _ahora()
        existente = self._db.execute(
            "SELECT id FROM procedural WHERE trigger = ? AND action = ?", (trigger, action)
        ).fetchone()
        if existente is not None:
            self._db.execute(
                "UPDATE procedural SET times_applied = times_applied + 1 WHERE id = ?",
                (existente["id"],),
            )
            self._db.commit()
            return int(existente["id"])

        cursor = self._db.execute(
            "INSERT INTO procedural (trigger, action, created_at) VALUES (?, ?, ?)",
            (trigger, action, ahora),
        )
        self._db.commit()
        return int(cursor.lastrowid)

    def rules(self, limit: int = 20) -> list[Rule]:
        """Reglas ordenadas por uso: las más aplicadas encabezan el prompt."""
        filas = self._db.execute(
            "SELECT * FROM procedural ORDER BY times_applied DESC, id ASC LIMIT ?", (limit,)
        ).fetchall()
        return [
            Rule(
                id=f["id"],
                trigger=f["trigger"],
                action=f["action"],
                times_applied=f["times_applied"],
            )
            for f in filas
        ]

    def forget_rule(self, rule_id: int) -> bool:
        cursor = self._db.execute("DELETE FROM procedural WHERE id = ?", (rule_id,))
        self._db.commit()
        return cursor.rowcount > 0

    # ── utilidades ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        return {
            tabla: int(self._db.execute(f"SELECT COUNT(*) AS n FROM {tabla}").fetchone()["n"])
            for tabla in ("entity", "episodic", "semantic", "procedural")
        }

    def recent_episodes(self, limit: int = 10, session: str | None = None) -> Iterable[Episode]:
        sql = "SELECT * FROM episodic"
        params: list[Any] = []
        if session is not None:
            sql += " WHERE session = ?"
            params.append(session)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        for fila in self._db.execute(sql, params).fetchall():
            yield Episode(
                id=fila["id"],
                session=fila["session"],
                role=fila["role"],
                content=fila["content"],
                entity_id=fila["entity_id"],
                created_at=fila["created_at"],
            )

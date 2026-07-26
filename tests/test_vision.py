"""Skills de visión, con dobles en lugar de cámara y modelo reales.

Lo que se prueba aquí no es InsightFace —eso es de sus autores— sino **el
cableado**: que reconocer a alguien no cree una entidad duplicada, que no se
enrole a nadie sin nombre, y que un vector biométrico nunca llegue a una traza.

Las pruebas contra la cámara de verdad viven en `test_integracion_local.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from argos.config import Settings
from argos.memory.embeddings import LexicalEmbedder
from argos.memory.store import EntityKind, MemoryStore
from argos.perception.camera import Frame
from argos.perception.identity import UMBRAL, Face
from argos.skills.vision import LookAround, RememberFace, WhoIsThis
from argos.trace import Event, Tracer


class CamaraFalsa:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    def health(self):
        return (True, "cámara falsa") if self.ok else (False, "no hay puente de cámara")

    def grab(self):
        return Frame(jpeg=b"jpeg-falso")


class ReconocedorFalso:
    """Devuelve las caras que se le programen, sin cargar ningún modelo."""

    def __init__(self, caras: list[Face] | None = None) -> None:
        self.caras = caras or []

    def detect(self, jpeg):
        return self.caras

    def primary(self, jpeg):
        return self.caras[0] if self.caras else None


def cara(semilla: int, area: int = 90_000, x: int = 640) -> Face:
    rng = np.random.default_rng(semilla)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    mitad = int((area**0.5) // 2)
    return Face(
        embedding=v, bbox=(x - mitad, 200, x + mitad, 200 + 2 * mitad), score=0.9, area=area
    )


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "m.db", embedder=LexicalEmbedder()) as s:
        yield s


class Avisos(list):
    """Recoge las llamadas a on_identified, que recibe (nombre, entity_id)."""

    def __call__(self, nombre: str, entity_id: int) -> None:
        self.append(nombre)


@pytest.fixture
def identificados():
    return Avisos()


# ─────────────────────────── reconocer ────────────────────────────────────────


def test_sin_cara_delante_lo_dice(store, identificados):
    skill = WhoIsThis(store, CamaraFalsa(), ReconocedorFalso([]), identificados)
    result = skill.execute({})

    assert result.ok
    assert result.data["faces"] == 0
    assert not identificados


def test_una_cara_desconocida_no_se_inventa_un_nombre(store, identificados):
    """El fallo peor posible sería asignar la identidad más parecida por descarte."""
    skill = WhoIsThis(store, CamaraFalsa(), ReconocedorFalso([cara(1)]), identificados)
    result = skill.execute({})

    assert result.data["known"] is False
    assert "no lo reconozco" in result.output
    assert not identificados, "no debe reasignar el interlocutor a alguien no reconocido"


def test_reconoce_a_quien_esta_enrolado_y_trae_lo_que_sabe(store, identificados):
    rostro = cara(7)
    entidad_id = store.upsert_entity(EntityKind.PERSONA, "Lady", embedding=rostro.embedding)
    store.remember_fact("Lady", "es veterinaria", entity_id=entidad_id)

    skill = WhoIsThis(store, CamaraFalsa(), ReconocedorFalso([rostro]), identificados)
    result = skill.execute({})

    assert result.data["known"] is True
    assert result.data["person"] == "Lady"
    assert "veterinaria" in result.output
    # El mismo callback que usa identify_speaker: las dos vías convergen.
    assert identificados == ["Lady"]


def test_otra_cara_no_se_confunde_con_la_enrolada(store, identificados):
    """Confundir a dos personas mezcla dos memorias y es difícil de deshacer."""
    store.upsert_entity(EntityKind.PERSONA, "Lady", embedding=cara(7).embedding)

    skill = WhoIsThis(store, CamaraFalsa(), ReconocedorFalso([cara(99)]), identificados)
    assert skill.execute({}).data["known"] is False


def test_sin_puente_de_camara_falla_con_mensaje_util(store, identificados):
    skill = WhoIsThis(store, CamaraFalsa(ok=False), ReconocedorFalso([]), identificados)
    result = skill.execute({})

    assert not result.ok
    assert "puente de cámara" in result.error


# ─────────────────────────── enrolar ──────────────────────────────────────────


def test_enrolar_une_la_cara_a_la_entidad_que_YA_existe(store, identificados):
    """Punto donde convergen las dos vías de identidad: por nombre y por rostro."""
    previo = store.upsert_entity(EntityKind.PERSONA, "Sadid")
    store.remember_fact("Sadid", "es licenciado en física", entity_id=previo)

    skill = RememberFace(store, CamaraFalsa(), ReconocedorFalso([cara(3)]), identificados)
    result = skill.execute({"name": "Sadid"})

    assert result.data["entity_id"] == previo, "creó una entidad duplicada"
    assert store.stats()["entity"] == 1
    # Y el hecho anterior sigue asociado a la misma persona.
    assert store.facts_about("Sadid")


def test_tras_enrolar_ya_reconoce(store, identificados):
    rostro = cara(11)
    RememberFace(store, CamaraFalsa(), ReconocedorFalso([rostro]), identificados).execute(
        {"name": "Caronte"}
    )

    result = WhoIsThis(store, CamaraFalsa(), ReconocedorFalso([rostro]), identificados).execute({})

    assert result.data["person"] == "Caronte"
    assert result.data["confidence"] >= UMBRAL


def test_no_se_enrola_sin_cara_delante(store, identificados):
    skill = RememberFace(store, CamaraFalsa(), ReconocedorFalso([]), identificados)
    result = skill.execute({"name": "Alguien"})

    assert not result.ok
    assert store.stats()["entity"] == 0


def test_no_se_enrola_con_un_nombre_vacio(store, identificados):
    skill = RememberFace(store, CamaraFalsa(), ReconocedorFalso([cara(5)]), identificados)
    assert not skill.execute({"name": " "}).ok
    assert store.stats()["entity"] == 0


def test_forget_borra_tambien_el_vector_facial(store, identificados):
    """Borrar a una persona debe llevarse su biometría, no sólo sus hechos."""
    RememberFace(store, CamaraFalsa(), ReconocedorFalso([cara(13)]), identificados).execute(
        {"name": "Temporal"}
    )
    store.forget_entity(EntityKind.PERSONA, "Temporal")

    assert store.find_entity_by_embedding(cara(13).embedding, EntityKind.PERSONA) is None


# ─────────────────────────── mirar ────────────────────────────────────────────


def test_look_describe_posiciones_en_palabras(store):
    """El modelo no debe interpretar coordenadas: ahí es donde se equivoca."""
    skill = LookAround(CamaraFalsa(), ReconocedorFalso([cara(1, x=100), cara(2, x=1100)]))
    result = skill.execute({})

    assert result.data["faces"] == 2
    assert "izquierda" in result.output and "derecha" in result.output
    assert "px" not in result.output and "bbox" not in result.output


def test_look_sin_nadie(store):
    result = LookAround(CamaraFalsa(), ReconocedorFalso([])).execute({})
    assert result.ok
    assert result.data["faces"] == 0


# ─────────────────────────── privacidad ───────────────────────────────────────


def test_un_embedding_facial_NUNCA_llega_a_la_traza(tmp_path, monkeypatch):
    """Un embedding facial es el rostro: hay reconstrucción demostrada vía difusión.

    Las trazas se conservan, así que una fuga aquí es permanente.
    """
    monkeypatch.setenv("ARGOS_PATHS__TRACES", str(tmp_path / "traces"))
    tracer = Tracer(settings=Settings())
    vector = cara(42).embedding

    tracer.emit(
        Event.SKILL_RESULT,
        skill="who_is_this",
        embedding=vector.tolist(),
        face_embedding=vector.tolist(),
        person="Sadid",
        confidence=0.96,
    )

    crudo = tracer.path.read_text(encoding="utf-8")
    assert f"{vector[0]:.6f}"[:8] not in crudo, "el vector facial acabó en la traza"

    registro = tracer.read_all()[-1]
    assert registro["embedding"] == "«redactado»"
    assert registro["face_embedding"] == "«redactado»"
    # El nombre y la confianza sí deben quedar: son lo auditable.
    assert registro["person"] == "Sadid"
    assert registro["confidence"] == 0.96

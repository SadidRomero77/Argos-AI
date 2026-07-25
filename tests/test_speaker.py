"""Identificación del interlocutor: hoy por nombre, mañana por rostro.

El diseño clave que prueban estos tests es que `identify_speaker` es el **único**
punto donde se ata una conversación a una persona. En Fase 2, el reconocimiento
facial llamará exactamente aquí con el `entity_id` que resuelva InsightFace, y
todo lo de aguas abajo —episodios asociados, perfil recuperado— seguirá igual.
"""

from __future__ import annotations

import pytest

from argos.memory.embeddings import LexicalEmbedder
from argos.memory.store import EntityKind, MemoryStore
from argos.skills.memory import Forget, IdentifySpeaker, Recall, Remember


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "m.db", embedder=LexicalEmbedder()) as s:
        yield s


@pytest.fixture
def identificado():
    """Captura las llamadas al callback, como haría el gateway."""
    return []


# ───────────────────────── identificar al hablante ────────────────────────────


def test_identificar_a_alguien_nuevo_crea_la_entidad(store, identificado):
    skill = IdentifySpeaker(store, lambda n, i: identificado.append((n, i)))

    result = skill.execute({"name": "Lady"})

    assert result.ok
    assert result.data["known"] is False
    assert store.get_entity(EntityKind.PERSONA, "Lady") is not None
    # El gateway se entera para asociar los episodios siguientes.
    assert identificado == [("Lady", result.data["entity_id"])]


def test_identificar_a_alguien_conocido_devuelve_lo_que_sabe(store, identificado):
    entidad_id = store.upsert_entity(EntityKind.PERSONA, "Lady")
    store.remember_fact("Lady", "es veterinaria", entity_id=entidad_id)
    skill = IdentifySpeaker(store, lambda n, i: identificado.append((n, i)))

    result = skill.execute({"name": "Lady"})

    assert result.data["known"] is True
    assert "veterinaria" in result.output


def test_identificar_dos_veces_no_duplica_la_entidad(store, identificado):
    skill = IdentifySpeaker(store, lambda n, i: identificado.append((n, i)))
    primero = skill.execute({"name": "Lady"})
    segundo = skill.execute({"name": "Lady"})

    assert primero.data["entity_id"] == segundo.data["entity_id"]
    assert store.stats()["entity"] == 1


def test_un_nombre_vacio_se_rechaza(store, identificado):
    skill = IdentifySpeaker(store, lambda n, i: identificado.append((n, i)))
    result = skill.execute({"name": " "})

    assert not result.ok
    assert not identificado, "no debe reasignarse el interlocutor con un nombre inválido"


def test_el_entity_id_devuelto_sirve_para_asociar_episodios(store, identificado):
    """El puente completo: identificar → asociar → recuperar filtrando por persona."""
    skill = IdentifySpeaker(store, lambda n, i: identificado.append((n, i)))
    lady_id = skill.execute({"name": "Lady"}).data["entity_id"]
    otro_id = store.upsert_entity(EntityKind.PERSONA, "Sadid")

    store.remember_episode("s1", "user", "hablamos de gatos y perros", entity_id=lady_id)
    store.remember_episode("s1", "user", "hablamos de gatos y perros", entity_id=otro_id)

    de_lady = store.recall_episodes("gatos y perros", entity_id=lady_id, min_similarity=-1.0)
    assert len(de_lady) == 1
    assert de_lady[0].entity_id == lady_id


# ───────────────────────── skills de memoria ──────────────────────────────────


def test_recall_dice_explicitamente_que_no_hay_nada(store):
    """Que lo diga evita que el modelo rellene el hueco inventando."""
    result = Recall(store).execute({"query": "el color favorito de alguien"})

    assert result.ok
    assert result.data["found"] == 0
    assert "No hay nada" in result.output


def test_remember_y_luego_recall_lo_encuentra(store):
    Remember(store).execute({"subject": "Sadid", "fact": "su color favorito es el azul cobalto"})

    result = Recall(store).execute({"query": "color favorito azul cobalto de Sadid"})

    assert result.data["found"] >= 1
    assert "azul cobalto" in result.output


def test_remember_rechaza_un_dato_vacio(store):
    assert not Remember(store).execute({"subject": "Sadid", "fact": "  "}).ok


def test_remember_con_un_tipo_desconocido_no_falla(store):
    """Un tipo mal escrito por el modelo no debe impedir que se guarde el dato."""
    result = Remember(store).execute(
        {"subject": "cuántica", "fact": "le interesa a Sadid", "kind": "inventado"}
    )
    assert result.ok


def test_forget_borra_todo_lo_asociado(store):
    entidad_id = store.upsert_entity(EntityKind.PERSONA, "Lady")
    store.remember_fact("Lady", "es veterinaria", entity_id=entidad_id)
    store.remember_episode("s1", "user", "conversación con Lady", entity_id=entidad_id)

    assert Forget(store).execute({"subject": "Lady"}).ok

    assert store.stats() == {"entity": 0, "episodic": 0, "semantic": 0, "procedural": 0}


def test_forget_de_alguien_inexistente_no_es_un_error(store):
    result = Forget(store).execute({"subject": "Nadie"})
    assert result.ok
    assert "No había nada" in result.output

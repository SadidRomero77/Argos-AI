"""Memoria: los tres niveles y la tabla de entidades, sin servidor.

`FakeEmbedder` produce vectores deterministas, así que toda la lógica se prueba en
milisegundos. Las pruebas contra el modelo real de embeddings viven en
`test_integracion_local.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from argos.memory.embeddings import FakeEmbedder, normalize
from argos.memory.store import EntityKind, MemoryStore


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "memoria.db", embedder=FakeEmbedder()) as s:
        yield s


# ───────────────────────────── embeddings ─────────────────────────────────────


def test_normalize_deja_norma_unitaria():
    normalizado = normalize(np.array([[3.0, 4.0]]))
    assert np.isclose(np.linalg.norm(normalizado[0]), 1.0)


def test_normalize_no_produce_nan_con_un_vector_nulo():
    """Dividir por cero envenenaría todas las búsquedas posteriores en silencio."""
    resultado = normalize(np.zeros((1, 8)))
    assert not np.isnan(resultado).any()


def test_el_fake_embedder_es_determinista_entre_llamadas():
    a, b = FakeEmbedder(), FakeEmbedder()
    assert np.allclose(a.embed(["hola"]), b.embed(["hola"]))
    assert not np.allclose(a.embed(["hola"]), a.embed(["adiós"]))


def test_el_fake_embedder_acepta_lista_vacia():
    assert FakeEmbedder(dimensions=16).embed([]).shape == (0, 16)


# ────────────────────────────── entidades ─────────────────────────────────────


def test_upsert_crea_y_luego_actualiza_sin_duplicar(store):
    primero = store.upsert_entity(EntityKind.PERSONA, "Sadid")
    segundo = store.upsert_entity(EntityKind.PERSONA, "Sadid")

    assert primero == segundo
    assert store.stats()["entity"] == 1


def test_el_mismo_nombre_con_distinto_tipo_son_entidades_distintas(store):
    """'Taller' puede ser un lugar y un tema a la vez."""
    lugar = store.upsert_entity(EntityKind.LUGAR, "taller")
    tema = store.upsert_entity(EntityKind.TEMA, "taller")

    assert lugar != tema
    assert store.stats()["entity"] == 2


def test_upsert_fusiona_metadatos(store):
    store.upsert_entity(EntityKind.PERSONA, "Ana", rol="ingeniera")
    store.upsert_entity(EntityKind.PERSONA, "Ana", ciudad="Bogotá")

    entidad = store.get_entity(EntityKind.PERSONA, "Ana")
    assert entidad.meta == {"rol": "ingeniera", "ciudad": "Bogotá"}


def test_busqueda_por_embedding_encuentra_la_entidad(store):
    """Base del reconocimiento facial de Fase 2, hoy probada con vectores sintéticos."""
    emb = FakeEmbedder()
    vector = emb.embed(["rostro-de-ana"])[0]
    store.upsert_entity(EntityKind.PERSONA, "Ana", embedding=vector)

    encontrado = store.find_entity_by_embedding(vector, EntityKind.PERSONA)

    assert encontrado is not None
    entidad, score = encontrado
    assert entidad.label == "Ana"
    assert score > 0.99


def test_un_vector_distinto_NO_confunde_identidades(store):
    """Crear una identidad nueva es preferible a confundir a dos personas."""
    emb = FakeEmbedder()
    store.upsert_entity(EntityKind.PERSONA, "Ana", embedding=emb.embed(["rostro-ana"])[0])

    otro = emb.embed(["rostro-de-alguien-completamente-distinto"])[0]
    assert store.find_entity_by_embedding(otro, EntityKind.PERSONA, threshold=0.75) is None


def test_forget_entity_borra_en_cascada_todo_lo_asociado(store):
    """Borrar datos personales no debe dejar rastros sueltos."""
    entidad_id = store.upsert_entity(EntityKind.PERSONA, "Ana")
    store.remember_episode("s1", "user", "Ana dijo hola", entity_id=entidad_id)
    store.remember_fact("Ana", "prefiere el café sin azúcar", entity_id=entidad_id)

    assert store.forget_entity(EntityKind.PERSONA, "Ana") == 1

    assert store.stats()["entity"] == 0
    assert store.stats()["episodic"] == 0, "quedaron episodios huérfanos"
    assert store.stats()["semantic"] == 0, "quedaron hechos huérfanos"


# ──────────────────────────── memoria episódica ───────────────────────────────


def test_recuerda_y_recupera_por_similitud(store):
    store.remember_episode("s1", "user", "el robot usa una Raspberry Pi 5")
    store.remember_episode("s1", "user", "hoy comí arepas")

    resultados = store.recall_episodes("el robot usa una Raspberry Pi 5", limit=1)

    assert len(resultados) == 1
    assert "Raspberry" in resultados[0].content


def test_la_recencia_desempata(store):
    """Sin esto, el agente desentierra lo de hace meses ignorando lo de hace un minuto."""
    for i in range(5):
        store.remember_episode("s1", "user", f"nota número {i}")

    # Con todo el peso en recencia, lo último guardado debe encabezar.
    resultados = store.recall_episodes("cualquier cosa", limit=1, recency_weight=1.0)
    assert resultados[0].content == "nota número 4"


def test_se_puede_filtrar_por_entidad(store):
    ana = store.upsert_entity(EntityKind.PERSONA, "Ana")
    beto = store.upsert_entity(EntityKind.PERSONA, "Beto")
    store.remember_episode("s1", "user", "conversación con Ana", entity_id=ana)
    store.remember_episode("s1", "user", "conversación con Beto", entity_id=beto)

    resultados = store.recall_episodes("conversación", entity_id=ana)

    assert len(resultados) == 1
    assert "Ana" in resultados[0].content


def test_recall_sobre_memoria_vacia_devuelve_lista_vacia(store):
    assert store.recall_episodes("lo que sea") == []


def test_persiste_al_reabrir_la_base(tmp_path):
    """La prueba que de verdad importa: recordar entre ejecuciones distintas."""
    ruta = tmp_path / "memoria.db"
    with MemoryStore(ruta, embedder=FakeEmbedder()) as s:
        s.remember_episode("s1", "user", "el proyecto se llama ARGOS")

    with MemoryStore(ruta, embedder=FakeEmbedder()) as s:
        resultados = s.recall_episodes("el proyecto se llama ARGOS", limit=1)
        assert "ARGOS" in resultados[0].content


# ──────────────────────────── memoria semántica ───────────────────────────────


def test_un_hecho_repetido_no_se_duplica(store):
    primero = store.remember_fact("Sadid", "prefiere respuestas cortas")
    segundo = store.remember_fact("Sadid", "prefiere respuestas cortas")

    assert primero == segundo
    assert store.stats()["semantic"] == 1


def test_recupera_hechos_por_sujeto(store):
    store.remember_fact("Sadid", "trabaja en WSL2")
    store.remember_fact("Ana", "vive en Bogotá")

    resultados = store.recall_facts("dónde", subject="Ana")

    assert len(resultados) == 1
    assert resultados[0].fact == "vive en Bogotá"


def test_forget_fact(store):
    fact_id = store.remember_fact("Sadid", "dato equivocado")
    assert store.forget_fact(fact_id)
    assert store.stats()["semantic"] == 0


# ────────────────────────── memoria procedimental ─────────────────────────────


def test_una_regla_repetida_incrementa_su_uso_en_vez_de_duplicarse(store):
    store.learn_rule("el usuario pregunte por archivos", "usar glob en vez de run_command")
    store.learn_rule("el usuario pregunte por archivos", "usar glob en vez de run_command")

    reglas = store.rules()
    assert len(reglas) == 1
    assert reglas[0].times_applied == 1


def test_las_reglas_mas_usadas_encabezan(store):
    store.learn_rule("A", "hacer a")
    store.learn_rule("B", "hacer b")
    for _ in range(3):
        store.learn_rule("B", "hacer b")

    assert store.rules()[0].trigger == "B"


def test_una_regla_se_renderiza_para_el_prompt(store):
    store.learn_rule("pregunten la hora", "responder en formato 24h")
    assert store.rules()[0].render() == "Cuando pregunten la hora, responder en formato 24h."


# ────────────────────────────── robustez ──────────────────────────────────────


def test_cambiar_de_modelo_de_embeddings_no_rompe_el_arranque(tmp_path):
    """Los recuerdos viejos quedan inaccesibles, pero el agente sigue funcionando.

    Reventar aquí dejaría al agente sin arrancar tras cambiar de modelo, que es un
    fallo mucho peor que perder acceso temporal a recuerdos antiguos.
    """
    ruta = tmp_path / "memoria.db"
    with MemoryStore(ruta, embedder=FakeEmbedder(dimensions=64)) as s:
        s.remember_episode("s1", "user", "recuerdo con vectores de 64 dimensiones")

    with MemoryStore(ruta, embedder=FakeEmbedder(dimensions=128)) as s:
        assert s.recall_episodes("recuerdo") == []  # inaccesible, pero sin excepción
        s.remember_episode("s2", "user", "recuerdo nuevo de 128")
        assert len(s.recall_episodes("recuerdo nuevo de 128", limit=1)) == 1


def test_stats_cuenta_las_cuatro_tablas(store):
    store.upsert_entity(EntityKind.PERSONA, "Ana")
    store.remember_episode("s1", "user", "hola")
    store.remember_fact("Ana", "existe")
    store.learn_rule("pase algo", "hacer algo")

    assert store.stats() == {"entity": 1, "episodic": 1, "semantic": 1, "procedural": 1}


def test_el_fake_embedder_distingue_textos_con_prefijo_comun():
    """Regresión: la semilla dependía sólo de los 4 primeros caracteres.

    'rostro-ana' y 'rostro-de-otro' producían vectores IDÉNTICOS, lo que hacía
    pasar cualquier prueba de similitud sin comprobar nada real.
    """
    emb = FakeEmbedder()
    a = emb.embed(["rostro-ana"])[0]
    b = emb.embed(["rostro-de-alguien-completamente-distinto"])[0]

    assert float(np.dot(a, b)) < 0.5, "textos distintos con prefijo común dan el mismo vector"


def test_la_recencia_no_aplasta_una_similitud_claramente_mejor():
    """Regresión de un fallo de calibración observado con bge-m3.

    Mezclar similitud (banda estrecha ~0,4 a 0,9) con recencia (0 a 1 completo) hacía
    que lo último guardado ganara aunque fuera irrelevante: ante "¿qué tarjeta
    gráfica tengo?" el primer resultado era "me gustan las respuestas cortas".
    La similitud se re-escala al rango observado antes de mezclar.
    """

    class SimuladaEmbedder:
        """Reproduce la banda estrecha de un embedder real."""

        dimensions = 2

        def embed(self, texts):
            # Ángulos elegidos para que las similitudes caigan en ~0,55 a 0,95.
            mapa = {"consulta": 0.0, "muy relevante": 0.32, "nada que ver": 0.98}
            angulos = [mapa.get(t, 1.4) for t in texts]
            return np.array([[np.cos(a), np.sin(a)] for a in angulos], dtype=np.float32)

    import tempfile

    with (
        tempfile.TemporaryDirectory() as d,
        MemoryStore(f"{d}/m.db", embedder=SimuladaEmbedder()) as s,
    ):
        s.remember_episode("s1", "user", "muy relevante")
        s.remember_episode("s1", "user", "nada que ver")  # el más reciente

        top = s.recall_episodes("consulta", limit=1)[0]

    assert top.content == "muy relevante", "lo reciente e irrelevante le ganó a lo pertinente"

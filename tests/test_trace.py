"""La traza es append-only y jamás escribe un secreto en disco.

Estas pruebas son de seguridad, no de formato: una API key o un embedding
biométrico en un JSONL es una fuga permanente, porque las trazas se conservan.
"""

from __future__ import annotations

import json

import pytest

from argos.config import Settings
from argos.trace import Event, Tracer, redact


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-clave-larguisima-de-prueba-123456")
    monkeypatch.setenv("ARGOS_PATHS__TRACES", str(tmp_path / "traces"))
    return Settings()


def test_emite_jsonl_con_secuencia_y_timestamp(settings):
    tracer = Tracer(settings=settings)
    tracer.emit(Event.TURN_START, turno=1)

    lineas = tracer.read_all()
    assert [linea["event"] for linea in lineas] == ["session_start", "turn_start"]
    assert [linea["seq"] for linea in lineas] == [1, 2]
    assert lineas[1]["turno"] == 1
    assert lineas[0]["session"] == tracer.session_id


def test_es_append_only(settings):
    """Emitir nunca reescribe: cada evento sólo añade una línea."""
    tracer = Tracer(settings=settings)
    for i in range(5):
        tracer.emit(Event.SKILL_CALL, i=i)
    assert len(tracer.read_all()) == 6  # 5 + session_start


def test_redacta_por_nombre_de_clave(settings):
    tracer = Tracer(settings=settings)
    tracer.emit(Event.MODEL_CALL, api_key="lo-que-sea", modelo="claude-haiku-4-5")

    registro = tracer.read_all()[-1]
    assert registro["api_key"] == "«redactado»"
    assert registro["modelo"] == "claude-haiku-4-5"  # lo no sensible pasa intacto


def test_redacta_el_valor_literal_del_secreto_aunque_venga_incrustado(settings):
    """El caso real: la clave aparece dentro del texto de un error de la API."""
    tracer = Tracer(settings=settings)
    tracer.emit(
        Event.ERROR,
        mensaje="401 unauthorized usando sk-ant-clave-larguisima-de-prueba-123456 en el header",
    )

    crudo = tracer.path.read_text(encoding="utf-8")
    assert "sk-ant-clave-larguisima-de-prueba-123456" not in crudo
    assert "«redactado»" in crudo


def test_redacta_embeddings_biometricos(settings):
    """Un embedding facial es el rostro. Nunca en una traza."""
    tracer = Tracer(settings=settings)
    tracer.emit(Event.SKILL_CALL, skill="who_is_this", embedding=[0.1] * 512, persona="ana")

    registro = tracer.read_all()[-1]
    assert registro["embedding"] == "«redactado»"
    assert registro["persona"] == "ana"


def test_redacta_en_estructuras_anidadas(settings):
    tracer = Tracer(settings=settings)
    tracer.emit(Event.SKILL_CALL, payload={"headers": [{"authorization": "Bearer xyz"}]})

    registro = tracer.read_all()[-1]
    assert registro["payload"]["headers"][0]["authorization"] == "«redactado»"


def test_trunca_valores_enormes(settings):
    tracer = Tracer(settings=settings)
    tracer.emit(Event.SKILL_RESULT, salida="x" * 10_000)

    salida = tracer.read_all()[-1]["salida"]
    assert len(salida) < 5_000
    assert "truncado" in salida


def test_redact_no_se_cuelga_con_referencias_ciclicas():
    ciclico: dict = {"a": 1}
    ciclico["yo"] = ciclico
    assert redact(ciclico, []) is not None  # termina en vez de desbordar la pila


def test_cada_linea_es_json_valido(settings):
    tracer = Tracer(settings=settings)
    tracer.emit(Event.TURN_START, texto='comillas " y \n saltos y emoji 🤖')

    for linea in tracer.path.read_text(encoding="utf-8").splitlines():
        json.loads(linea)  # lanza si alguna línea está corrupta

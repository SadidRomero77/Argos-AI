"""El CLI: sólo cableado, pero es la superficie que el usuario toca de verdad."""

from __future__ import annotations

import time

from argos.cli import build_parser, main
from argos.harness.permissions import AlwaysAllowApprover, ConsoleApprover, DenyAllApprover


def test_el_default_de_aprobacion_es_denegar():
    """Ejecutar `argos "..."` sin flags no debe poder auto-aprobarse nada."""
    assert build_parser().parse_args(["hola"]).approve == "deny"


def test_los_tres_modos_de_aprobacion_estan_cableados():
    from argos.cli import _APROBADORES

    assert _APROBADORES["deny"] is DenyAllApprover
    assert _APROBADORES["ask"] is ConsoleApprover
    assert _APROBADORES["yes"] is AlwaysAllowApprover


def test_el_kind_por_defecto_es_la_tarea_barata():
    assert build_parser().parse_args(["hola"]).kind == "routine"


def test_sin_credenciales_sale_con_codigo_2_y_mensaje_accionable(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("ARGOS_PATHS__TRACES", str(tmp_path / "traces"))

    codigo = main(["hola"])  # el conftest garantiza que no hay API key

    assert codigo == 2
    error = capsys.readouterr().err
    assert ".env" in error and "LOCAL_ONLY" in error


def test_local_sin_servidor_falla_rapido_y_no_va_a_la_nube(capsys, tmp_path, monkeypatch):
    """Dos garantías: no degrada a la API externa, y avisa ANTES de gastar el turno.

    Antes descubría el problema a mitad del primer turno, esperando el timeout
    completo contra un puerto sin nadie escuchando.
    """
    monkeypatch.setenv("ARGOS_PATHS__TRACES", str(tmp_path / "traces"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-cualquier-cosa-larga")
    # Puerto reservado a "descarte": garantiza connection refused inmediato.
    monkeypatch.setenv("ARGOS_MODELS__LOCAL_URL", "http://127.0.0.1:9/v1")

    inicio = time.monotonic()
    codigo = main(["--local", "hola"])
    transcurrido = time.monotonic() - inicio

    assert codigo == 2
    assert transcurrido < 10, f"tardó {transcurrido:.1f}s en avisar; debe fallar rápido"
    assert "no hay servidor local" in capsys.readouterr().err

"""El CLI: sólo cableado, pero es la superficie que el usuario toca de verdad."""

from __future__ import annotations

import pytest

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


def test_local_sin_proveedor_tambien_falla_en_vez_de_ir_a_la_nube(tmp_path, monkeypatch):
    """El modo local nunca debe degradar silenciosamente a una API externa."""
    monkeypatch.setenv("ARGOS_PATHS__TRACES", str(tmp_path / "traces"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-cualquier-cosa-larga")

    from argos.models.router import LocalProviderUnavailable

    with pytest.raises(LocalProviderUnavailable):
        main(["--local", "hola"])

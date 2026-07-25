"""El PermissionGate falla cerrado. Estas pruebas son el contrato de seguridad."""

from __future__ import annotations

import pytest

from argos.config import Settings
from argos.harness.permissions import (
    AlwaysAllowApprover,
    Decision,
    DenyAllApprover,
    PermissionGate,
    PermissionPolicy,
    SkillRule,
)
from argos.trace import Tracer


@pytest.fixture
def policy() -> PermissionPolicy:
    return PermissionPolicy(
        default=Decision.ASK,
        phase=1,
        skills={
            "read_file": SkillRule(decision=Decision.ALLOW, min_phase=1),
            "run_command": SkillRule(decision=Decision.ASK, min_phase=1, reason="ejecuta código"),
            "nuke": SkillRule(decision=Decision.DENY, min_phase=1, reason="jamás"),
            "move": SkillRule(decision=Decision.ALLOW, min_phase=4),
        },
    )


def test_allow_se_permite(policy):
    assert PermissionGate(policy=policy).evaluate("read_file").allowed


def test_deny_se_deniega_y_explica(policy):
    outcome = PermissionGate(policy=policy).evaluate("nuke")
    assert not outcome.allowed
    assert outcome.decision is Decision.DENY
    assert "jamás" in outcome.explanation


def test_skill_desconocida_cae_al_default_ask_y_sin_aprobador_se_deniega(policy):
    """Una skill nueva no obtiene permisos por el hecho de existir."""
    outcome = PermissionGate(policy=policy).evaluate("skill_inventada")
    assert not outcome.allowed
    assert outcome.decision is Decision.ASK


def test_ask_sin_aprobador_falla_cerrado(policy):
    """El default DenyAllApprover impide que un agente headless se auto-apruebe."""
    gate = PermissionGate(policy=policy, approver=DenyAllApprover())
    assert not gate.evaluate("run_command", {"cmd": "ls"}).allowed


def test_ask_con_aprobador_que_acepta(policy):
    gate = PermissionGate(policy=policy, approver=AlwaysAllowApprover())
    assert gate.evaluate("run_command", {"cmd": "ls"}).allowed


def test_la_fase_gana_sobre_un_allow(policy):
    """El control clave: 'move' dice ALLOW pero es de Fase 4 y estamos en Fase 1."""
    outcome = PermissionGate(policy=policy).evaluate("move", {"distance": 1.0})
    assert not outcome.allowed
    assert outcome.decision is Decision.DENY
    assert "fase 4" in outcome.explanation


def test_al_subir_la_fase_la_skill_se_habilita(policy):
    policy.phase = 4
    assert PermissionGate(policy=policy).evaluate("move", {"distance": 1.0}).allowed


def test_la_fase_no_convierte_un_deny_en_allow(policy):
    """Subir de fase habilita, pero nunca anula una denegación explícita."""
    policy.phase = 9
    assert not PermissionGate(policy=policy).evaluate("nuke").allowed


def test_un_aprobador_que_lanza_excepcion_no_abre_la_puerta(policy):
    class AprobadorRoto:
        def ask(self, request):
            raise RuntimeError("terminal desconectado")

    outcome = PermissionGate(policy=policy, approver=AprobadorRoto()).evaluate("run_command")
    assert not outcome.allowed
    assert "falló" in outcome.explanation


def test_outcome_es_evaluable_como_booleano(policy):
    """Permite escribir `if not gate.evaluate(...)` sin olvidar `.allowed`."""
    gate = PermissionGate(policy=policy)
    assert bool(gate.evaluate("read_file"))
    assert not bool(gate.evaluate("nuke"))


def test_cada_decision_queda_en_la_traza(tmp_path, monkeypatch, policy):
    """Auditabilidad: no debe existir una decisión de permiso sin registro."""
    monkeypatch.setenv("ARGOS_PATHS__TRACES", str(tmp_path / "traces"))
    tracer = Tracer(settings=Settings())
    gate = PermissionGate(policy=policy, tracer=tracer)

    gate.evaluate("read_file")
    gate.evaluate("nuke")

    permisos = [linea for linea in tracer.read_all() if linea["event"] == "permission"]
    assert [p["skill"] for p in permisos] == ["read_file", "nuke"]
    assert [p["allowed"] for p in permisos] == [True, False]


def test_los_parametros_sensibles_se_redactan_en_la_traza(tmp_path, monkeypatch, policy):
    monkeypatch.setenv("ARGOS_PATHS__TRACES", str(tmp_path / "traces"))
    tracer = Tracer(settings=Settings())
    gate = PermissionGate(policy=policy, tracer=tracer)

    gate.evaluate("read_file", {"path": "/tmp/x", "token": "super-secreto"})

    registro = [linea for linea in tracer.read_all() if linea["event"] == "permission"][-1]
    assert registro["params"]["token"] == "«redactado»"
    assert registro["params"]["path"] == "/tmp/x"


def test_carga_la_politica_real_del_repo():
    """La política versionada debe ser válida y coherente con el plan."""
    gate = PermissionGate()
    assert gate.policy.default is Decision.ASK, "el default debe ser 'ask'"
    # 'stop' nunca debe pedir permiso: frenar es siempre seguro.
    assert gate.policy.skills["stop"].decision is Decision.ALLOW
    # Las skills de hardware no deben ser invocables en la fase actual.
    assert gate.policy.skills["move"].min_phase >= 4
    assert not gate.evaluate("move", {"distance": 1.0}).allowed

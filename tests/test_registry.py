"""El registro es el único punto por el que el LLM puede actuar.

La prueba central de este archivo es que **no existe camino de despacho que salte
el PermissionGate**. Si alguna vez alguien añade uno, esa prueba falla.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from argos.config import Settings
from argos.harness.permissions import (
    AlwaysAllowApprover,
    Decision,
    PermissionGate,
    PermissionPolicy,
    SkillRule,
)
from argos.skills.base import Skill, SkillResult
from argos.tools.registry import SkillRegistry, default_registry
from argos.trace import Tracer


class Contador(Skill):
    """Skill de prueba que registra si llegó a ejecutarse."""

    name = "contador"
    description = "Incrementa un contador. Úsala sólo en pruebas."

    class Params(BaseModel):
        paso: int = 1

    def __init__(self) -> None:
        self.veces = 0

    def run(self, params) -> SkillResult:
        self.veces += params.paso
        return SkillResult.success(f"contador = {self.veces}", valor=self.veces)


def gate_con(decision: Decision, **kwargs) -> PermissionGate:
    policy = PermissionPolicy(
        default=Decision.DENY, phase=1, skills={"contador": SkillRule(decision=decision)}
    )
    return PermissionGate(policy=policy, **kwargs)


def test_despacha_cuando_la_politica_permite():
    skill = Contador()
    registry = SkillRegistry(gate=gate_con(Decision.ALLOW))
    registry.register(skill)

    assert registry.dispatch("contador", {"paso": 3}).ok
    assert skill.veces == 3


def test_una_skill_denegada_NO_se_ejecuta():
    """El control crítico: denegar debe impedir la ejecución, no sólo reportarla."""
    skill = Contador()
    registry = SkillRegistry(gate=gate_con(Decision.DENY))
    registry.register(skill)

    result = registry.dispatch("contador", {"paso": 3})
    assert not result.ok
    assert "permiso denegado" in result.error
    assert skill.veces == 0, "la skill se ejecutó pese a estar denegada"


def test_ask_denegado_por_el_aprobador_tampoco_ejecuta():
    skill = Contador()
    registry = SkillRegistry(gate=gate_con(Decision.ASK))  # DenyAllApprover por defecto
    registry.register(skill)

    assert not registry.dispatch("contador").ok
    assert skill.veces == 0


def test_ask_aprobado_si_ejecuta():
    skill = Contador()
    registry = SkillRegistry(gate=gate_con(Decision.ASK, approver=AlwaysAllowApprover()))
    registry.register(skill)

    assert registry.dispatch("contador").ok
    assert skill.veces == 1


def test_skill_desconocida_lista_las_disponibles():
    registry = SkillRegistry(gate=gate_con(Decision.ALLOW))
    registry.register(Contador())

    result = registry.dispatch("no_existe")
    assert not result.ok
    assert "contador" in result.error  # le decimos al modelo qué sí tiene


def test_no_se_puede_registrar_dos_veces_el_mismo_nombre():
    registry = SkillRegistry(gate=gate_con(Decision.ALLOW))
    registry.register(Contador())
    with pytest.raises(ValueError, match="ya está registrada"):
        registry.register(Contador())


def test_las_definiciones_van_en_orden_estable():
    """Reordenarlas invalidaría la caché de prompt entera: se renderizan al inicio."""
    registry = default_registry(gate=gate_con(Decision.ALLOW))
    nombres = [d["name"] for d in registry.tool_definitions()]
    assert nombres == sorted(nombres)
    assert nombres == [d["name"] for d in registry.tool_definitions()]


def test_el_registro_por_defecto_trae_las_skills_de_fase_1():
    registry = default_registry(gate=gate_con(Decision.ALLOW))
    assert set(registry.names) == {
        "glob",
        "read_file",
        "write_note",
        "run_command",
        "weather",
        "search_web",
        "fetch_url",
    }


def test_sin_memoria_no_se_registran_las_skills_de_memoria():
    """El agente debe arrancar aunque no haya embeddings: sin continuidad, pero vivo."""
    registry = default_registry(gate=gate_con(Decision.ALLOW), memory=None)
    assert not {"remember", "recall", "forget", "identify_speaker"} & set(registry.names)


def test_las_skills_que_traen_contenido_de_terceros_piden_permiso():
    """search_web y fetch_url meten texto ajeno en el razonamiento: nunca en `allow`.

    El riesgo no es el coste sino la inyección indirecta — una página puede
    contener instrucciones dirigidas al agente.
    """
    policy = PermissionGate().policy
    assert policy.skills["search_web"].decision is Decision.ASK
    assert policy.skills["fetch_url"].decision is Decision.ASK
    # weather sí puede ir en allow: devuelve números de una API, no texto libre.
    assert policy.skills["weather"].decision is Decision.ALLOW


def test_toda_ejecucion_queda_en_la_traza(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGOS_PATHS__TRACES", str(tmp_path / "traces"))
    tracer = Tracer(settings=Settings())
    registry = SkillRegistry(gate=gate_con(Decision.ALLOW, tracer=tracer), tracer=tracer)
    registry.register(Contador())

    registry.dispatch("contador", {"paso": 2})

    eventos = [linea["event"] for linea in tracer.read_all()]
    # permiso -> llamada -> resultado, en ese orden y sin huecos
    assert eventos == ["session_start", "permission", "skill_call", "skill_result"]


def test_una_denegacion_no_emite_skill_call(tmp_path, monkeypatch):
    """La traza debe distinguir 'se pidió' de 'se ejecutó'."""
    monkeypatch.setenv("ARGOS_PATHS__TRACES", str(tmp_path / "traces"))
    tracer = Tracer(settings=Settings())
    registry = SkillRegistry(gate=gate_con(Decision.DENY, tracer=tracer), tracer=tracer)
    registry.register(Contador())

    registry.dispatch("contador")

    eventos = [linea["event"] for linea in tracer.read_all()]
    assert "skill_call" not in eventos
    assert "permission" in eventos


def test_las_skills_reales_pasan_por_la_puerta_con_la_politica_del_repo():
    """Integración con config/permissions.yaml versionado: 'move' sigue bloqueada."""
    registry = default_registry()
    result = registry.dispatch("move", {"distance": 1.0})
    assert not result.ok
    # No está registrada en Fase 1, y aunque lo estuviera la fase la bloquearía.
    assert "no existe la skill" in result.error or "permiso denegado" in result.error

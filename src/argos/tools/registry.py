"""Registro de skills: lo único que el LLM puede invocar.

El registro expone **skills**, nunca primitivas de bajo nivel. Es donde se aplica
el principio rector: el modelo elige un nombre de la lista y pasa parámetros que
serán validados; no puede construir una acción arbitraria.

Todo despacho pasa obligatoriamente por el `PermissionGate` antes de ejecutar.
No existe un camino que salte la puerta.
"""

from __future__ import annotations

from typing import Any

from argos.harness.permissions import PermissionGate
from argos.skills.base import Skill, SkillResult
from argos.trace import Event, Tracer


class SkillRegistry:
    """Colección de skills disponibles, con despacho controlado por permisos."""

    def __init__(self, gate: PermissionGate | None = None, tracer: Tracer | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        self.gate = gate or PermissionGate(tracer=tracer)
        self.tracer = tracer

    def register(self, skill: Skill) -> Skill:
        if skill.name in self._skills:
            raise ValueError(f"la skill '{skill.name}' ya está registrada")
        self._skills[skill.name] = skill
        return skill

    def register_all(self, *skills: Skill) -> None:
        for skill in skills:
            self.register(skill)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._skills)

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Definiciones para la API, en orden estable.

        El orden importa: las definiciones de herramientas se renderizan al inicio
        del prompt, así que reordenarlas invalidaría la caché de prompt entera.
        """
        return [self._skills[name].tool_definition() for name in self.names]

    def dispatch(self, name: str, params: dict[str, Any] | None = None) -> SkillResult:
        """Ejecuta una skill por nombre. Único punto de entrada desde el agente."""
        params = params or {}

        skill = self._skills.get(name)
        if skill is None:
            disponibles = ", ".join(self.names) or "(ninguna)"
            return SkillResult.fail(f"no existe la skill '{name}'. Disponibles: {disponibles}")

        outcome = self.gate.evaluate(name, params)
        if not outcome.allowed:
            # El motivo vuelve al modelo para que pueda replantear en vez de reintentar.
            return SkillResult.fail(f"permiso denegado: {outcome.explanation}")

        if self.tracer is not None:
            self.tracer.emit(Event.SKILL_CALL, skill=name, params=params)

        result = skill.execute(params)

        if self.tracer is not None:
            self.tracer.emit(
                Event.SKILL_RESULT,
                skill=name,
                ok=result.ok,
                error=result.error,
                output=result.output,
            )
        return result


def default_registry(
    gate: PermissionGate | None = None,
    tracer: Tracer | None = None,
    memory: Any = None,
    on_identified: Any = None,
) -> SkillRegistry:
    """Registro con las skills de Fase 1 (sin hardware).

    `memory` es opcional: sin almacén no se registran las skills de memoria, y el
    agente sigue funcionando — sin continuidad, pero funcionando.

    `on_identified(nombre, entity_id)` se invoca cuando el agente descubre con
    quién habla. Lo usa el gateway para asociar los episodios a esa persona; en
    Fase 2 lo llamará también el reconocimiento facial.
    """
    from argos.skills.fs import Glob, ReadFile, WriteNote
    from argos.skills.shell import RunCommand

    registry = SkillRegistry(gate=gate, tracer=tracer)
    registry.register_all(Glob(), ReadFile(), WriteNote(), RunCommand())

    if memory is not None:
        from argos.skills.memory import Forget, IdentifySpeaker, Recall, Remember

        registry.register_all(Remember(memory), Recall(memory), Forget(memory))
        if on_identified is not None:
            registry.register(IdentifySpeaker(memory, on_identified))
    return registry

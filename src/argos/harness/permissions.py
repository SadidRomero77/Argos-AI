"""PermissionGate: decide si una skill puede ejecutarse, ANTES de ejecutarla.

Evalúa siempre en el orden `deny -> ask -> allow`. Dos controles independientes,
y ambos deben pasar:

1. **Fase** — una skill cuya `min_phase` supera la fase habilitada queda denegada
   aunque su regla diga `allow`. Es la aplicación literal del principio
   "Least Agency": la autonomía se gana por fase, no viene de fábrica.
2. **Decisión por skill** — `allow` / `ask` / `deny` desde `config/permissions.yaml`.

El comportamiento ante cualquier ambigüedad es **fallar cerrado**: sin política
cargada se deniega; con `ask` y sin aprobador se deniega; ante un error al evaluar
se deniega. Nunca se abre por defecto.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, Field

from argos.config import Settings, get_settings
from argos.trace import Event, Tracer


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class SkillRule(BaseModel):
    decision: Decision = Decision.ASK
    min_phase: int = 1
    reason: str | None = None


class PermissionPolicy(BaseModel):
    """Política declarativa. Se carga de YAML y no se muta en runtime."""

    default: Decision = Decision.ASK
    phase: int = 1
    skills: dict[str, SkillRule] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> PermissionPolicy:
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh) or {})


class PermissionRequest(BaseModel):
    """Lo que se le muestra al humano cuando hay que preguntar."""

    skill: str
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class Outcome(BaseModel):
    """Resultado de la evaluación. `allowed` es lo único que debe consultar el llamador."""

    allowed: bool
    decision: Decision
    skill: str
    explanation: str

    def __bool__(self) -> bool:
        return self.allowed


class Approver(Protocol):
    """Quien resuelve un `ask`. En Fase 3 lo implementa el botón físico del ESP32."""

    def ask(self, request: PermissionRequest) -> bool: ...


class DenyAllApprover:
    """Aprobador por defecto: deniega todo `ask`.

    Es deliberado. Un agente headless sin nadie mirando no debe poder auto-aprobarse
    acciones que la política marcó como necesitadas de supervisión.
    """

    def ask(self, request: PermissionRequest) -> bool:
        return False


class AlwaysAllowApprover:
    """Aprueba todo `ask`. SÓLO para pruebas y desarrollo local supervisado."""

    def ask(self, request: PermissionRequest) -> bool:
        return True


class ConsoleApprover:
    """Pregunta por stdin. La vía interactiva mientras no exista el terminal físico."""

    def ask(self, request: PermissionRequest) -> bool:
        print(f"\n  El agente quiere ejecutar: {request.skill}")
        if request.params:
            for key, value in request.params.items():
                print(f"     {key} = {value!r}")
        if request.reason:
            print(f"     motivo: {request.reason}")
        try:
            return input("  ¿Permitir? [s/N] ").strip().lower() in {"s", "si", "sí", "y", "yes"}
        except (EOFError, KeyboardInterrupt):
            print("  (sin respuesta -> denegado)")
            return False


class PermissionGate:
    """Puerta de permisos. Se consulta antes de CADA ejecución de skill."""

    def __init__(
        self,
        policy: PermissionPolicy | None = None,
        approver: Approver | None = None,
        tracer: Tracer | None = None,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or get_settings()
        if policy is None:
            path = settings.paths.resolved("permissions")
            policy = PermissionPolicy.from_yaml(path) if path.is_file() else PermissionPolicy()
        self.policy = policy
        # Fallar cerrado: sin aprobador explícito, todo `ask` se deniega.
        self.approver: Approver = approver or DenyAllApprover()
        self.tracer = tracer

    def evaluate(self, skill: str, params: dict[str, Any] | None = None) -> Outcome:
        params = params or {}
        outcome = self._decide(skill, params)
        if self.tracer is not None:
            self.tracer.emit(
                Event.PERMISSION,
                skill=skill,
                params=params,
                decision=str(outcome.decision),
                allowed=outcome.allowed,
                explanation=outcome.explanation,
                phase=self.policy.phase,
            )
        return outcome

    def _decide(self, skill: str, params: dict[str, Any]) -> Outcome:
        rule = self.policy.skills.get(skill)

        # Control 1 — la fase manda sobre la regla. Una skill de hardware no es
        # invocable en Fase 1 aunque alguien le ponga `allow` por descuido.
        if rule is not None and rule.min_phase > self.policy.phase:
            return Outcome(
                allowed=False,
                decision=Decision.DENY,
                skill=skill,
                explanation=(
                    f"'{skill}' requiere la fase {rule.min_phase} y la fase habilitada "
                    f"es la {self.policy.phase}"
                ),
            )

        decision = rule.decision if rule is not None else self.policy.default

        if decision is Decision.DENY:
            motivo = rule.reason if rule and rule.reason else "denegada por política"
            return Outcome(
                allowed=False, decision=decision, skill=skill, explanation=f"'{skill}': {motivo}"
            )

        if decision is Decision.ALLOW:
            return Outcome(
                allowed=True,
                decision=decision,
                skill=skill,
                explanation=f"'{skill}' permitida por política",
            )

        # ASK — la decisión la toma el humano; cualquier fallo se resuelve denegando.
        request = PermissionRequest(
            skill=skill, params=params, reason=rule.reason if rule else None
        )
        try:
            aprobado = bool(self.approver.ask(request))
        except Exception as exc:  # el aprobador puede colgarse o fallar; se cierra
            return Outcome(
                allowed=False,
                decision=Decision.ASK,
                skill=skill,
                explanation=f"'{skill}': el aprobador falló ({exc!r}) -> denegado",
            )

        return Outcome(
            allowed=aprobado,
            decision=Decision.ASK,
            skill=skill,
            explanation=f"'{skill}': {'aprobada' if aprobado else 'denegada'} por el usuario",
        )

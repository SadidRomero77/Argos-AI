"""AgentLoop: percibir -> pensar -> actuar, con presupuesto y límites duros.

Detalles que no son cosméticos y por los que el bucle está escrito así:

- **Todos los `tool_result` van en UN solo mensaje de usuario.** Repartirlos en varios
  le enseña al modelo a dejar de pedir herramientas en paralelo, y el sistema se vuelve
  secuencial sin que nadie sepa por qué.
- **Cada `tool_use` recibe su `tool_result`, incluso si falló.** Omitir uno hace que la
  API rechace el turno siguiente.
- **El contenido de las skills se envuelve en `<datos_externos>`.** Es el control contra
  inyección indirecta: el prompt de sistema declara que ahí dentro hay datos, nunca
  instrucciones.
- **El presupuesto se comprueba antes de cada iteración**, no después de gastar.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from argos.config import Settings, get_settings
from argos.models.base import LLMResponse, TaskKind, Usage
from argos.models.router import BudgetExceeded, ModelRouter
from argos.tools.registry import SkillRegistry
from argos.trace import Event, Tracer


class StopReason(StrEnum):
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    TOKEN_BUDGET = "token_budget"
    USD_BUDGET = "usd_budget"
    REFUSED = "refused"


class TurnResult(BaseModel):
    text: str = ""
    iterations: int = 0
    usage: Usage = Field(default_factory=Usage)
    usd: float = 0.0
    stopped_because: StopReason = StopReason.COMPLETED
    skill_calls: list[str] = Field(default_factory=list)
    # Veces que hubo que corregir al modelo por describir una llamada en vez
    # de emitirla. Es una métrica de calidad del modelo, no un error del turno.
    nudges: int = 0

    @property
    def ok(self) -> bool:
        return self.stopped_because is StopReason.COMPLETED


# Señales de que el modelo está *describiendo* una llamada en vez de emitirla.
_SENALES_LLAMADA = ("arguments", "tool_call", "parameters", '"tool"', "input_schema")


def looks_like_text_tool_call(text: str, skill_names: list[str]) -> str | None:
    """Detecta una llamada a skill escrita como texto. Devuelve el nombre, o None.

    Modo de fallo habitual en modelos pequeños: en vez de usar el canal de
    function-calling, escriben algo como

        ```json
        {"tool": "run_command", "arguments": ["ls"]}
        ```

    Sin esto el bucle lo toma por la respuesta final y cierra el turno con la
    tarea sin hacer — que es exactamente lo que se observó con qwen3:4b.

    La detección es deliberadamente estrecha: exige el nombre de una skill
    registrada, una llave de apertura y una palabra propia de una invocación.
    Hablar de una skill en prosa no debe dispararla.
    """
    if "{" not in text:
        return None
    bajo = text.lower()
    if not any(senal in bajo for senal in _SENALES_LLAMADA):
        return None
    for nombre in skill_names:
        if nombre in text:
            return nombre
    return None


def wrap_external(skill: str, content: str) -> str:
    """Marca contenido externo como datos, nunca como instrucciones.

    Un archivo o una página web pueden contener texto diseñado para redirigir al
    agente. El marcado no lo neutraliza por sí solo — lo hace junto con la regla
    del prompt de sistema — pero sin él ni siquiera hay frontera que señalar.
    """
    return f'<datos_externos skill="{skill}">\n{content}\n</datos_externos>'


class AgentLoop:
    """Un turno del agente: recibe una petición y trabaja hasta terminar o topar."""

    def __init__(
        self,
        router: ModelRouter,
        registry: SkillRegistry,
        tracer: Tracer | None = None,
        settings: Settings | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.router = router
        self.registry = registry
        self.tracer = tracer
        self.settings = settings or get_settings()
        self._system_prompt = system_prompt

    @property
    def system_prompt(self) -> str:
        """Se lee de disco una vez. Debe ser estable byte a byte: es el prefijo cacheado."""
        if self._system_prompt is None:
            path = self.settings.paths.resolved("system_prompt")
            self._system_prompt = path.read_text(encoding="utf-8") if path.is_file() else ""
        return self._system_prompt

    def run(
        self,
        user_message: str,
        kind: TaskKind = TaskKind.ROUTINE,
        history: list[dict[str, Any]] | None = None,
    ) -> TurnResult:
        messages: list[dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": user_message})

        result = TurnResult()
        nudged = False
        tools = self.registry.tool_definitions()
        max_iter = self.settings.budget.max_iterations
        max_tokens_tarea = self.settings.budget.max_tokens_task

        if self.tracer is not None:
            self.tracer.emit(Event.TURN_START, kind=str(kind), message=user_message)

        for iteracion in range(1, max_iter + 1):
            result.iterations = iteracion

            if result.usage.total_tokens >= max_tokens_tarea:
                result.stopped_because = StopReason.TOKEN_BUDGET
                break

            try:
                response = self.router.complete(
                    kind, messages, system=self.system_prompt, tools=tools
                )
            except BudgetExceeded:
                result.stopped_because = StopReason.USD_BUDGET
                break

            result.usage = result.usage + response.usage
            result.usd = self.router.total_usd
            if response.text:
                result.text = response.text

            if response.refused:
                result.stopped_because = StopReason.REFUSED
                break

            # El contenido crudo se devuelve tal cual: los bloques de thinking llevan
            # firma y la API rechaza los alterados.
            messages.append({"role": "assistant", "content": response.raw_content or response.text})

            if not response.wants_tools:
                # Antes de dar el turno por terminado: ¿escribió la llamada como
                # texto en vez de emitirla? Se corrige UNA vez; insistir con un
                # modelo que no sabe hacerlo sólo quema iteraciones.
                if not nudged and (
                    skill := looks_like_text_tool_call(response.text, self.registry.names)
                ):
                    nudged = True
                    result.nudges += 1
                    if self.tracer is not None:
                        self.tracer.emit(Event.NUDGE, skill=skill, text=response.text[:300])
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Has descrito una llamada a '{skill}' como texto, pero no la "
                                "has ejecutado. Invócala de verdad usando el mecanismo de "
                                "herramientas; no escribas la llamada en tu respuesta."
                            ),
                        }
                    )
                    continue

                result.stopped_because = StopReason.COMPLETED
                break

            messages.append({"role": "user", "content": self._execute_tools(response, result)})
        else:
            # El for terminó sin break: se agotaron las iteraciones.
            result.stopped_because = StopReason.MAX_ITERATIONS

        self._emit_end(result, max_iter)
        return result

    def _execute_tools(self, response: LLMResponse, result: TurnResult) -> list[dict[str, Any]]:
        """Ejecuta las skills pedidas y devuelve TODOS los resultados en un bloque."""
        bloques: list[dict[str, Any]] = []

        for call in response.tool_calls:
            result.skill_calls.append(call.name)
            skill_result = self.registry.dispatch(call.name, call.params)

            cuerpo = skill_result.output if skill_result.ok else (skill_result.error or "falló")
            bloques.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": wrap_external(call.name, cuerpo),
                    # Un error marcado deja que el modelo corrija en vez de reintentar igual.
                    "is_error": not skill_result.ok,
                }
            )

        return bloques

    def _emit_end(self, result: TurnResult, max_iter: int) -> None:
        if self.tracer is None:
            return
        self.tracer.emit(
            Event.TURN_END,
            stopped_because=str(result.stopped_because),
            iterations=result.iterations,
            skill_calls=result.skill_calls,
            tokens=result.usage.total_tokens,
            usd=round(result.usd, 6),
        )
        if result.stopped_because is not StopReason.COMPLETED:
            self.tracer.emit(
                Event.BUDGET,
                stopped_because=str(result.stopped_because),
                iterations=f"{result.iterations}/{max_iter}",
                tokens=f"{result.usage.total_tokens}/{self.settings.budget.max_tokens_task}",
                usd=f"{result.usd:.4f}/{self.settings.budget.max_usd_per_day:.2f}",
            )

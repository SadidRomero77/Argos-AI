"""Banco de evaluación de tool-calling contra las skills reales de Fase 1.

Mide lo único que importa para decidir si un modelo local sirve como cerebro:
**¿elige la skill correcta y le pasa parámetros válidos?** No perplexity, no
benchmarks genéricos. Cada caso tiene un resultado verificable por código.

El agente puede llegar al resultado correcto por caminos distintos, así que se
puntúan tres cosas por separado:

- `eligió_skill`  — invocó la skill esperada al menos una vez
- `parámetros_ok` — los parámetros pasaron la validación de la skill
- `respuesta_ok`  — la respuesta final contiene lo que debía

Un modelo que acierta la skill pero falla los parámetros necesita mejores
descripciones; uno que ni elige la skill necesita otro modelo. La distinción
importa y por eso no se colapsa en una sola nota.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from argos.config import Settings
from argos.harness.loop import AgentLoop, TurnResult
from argos.harness.permissions import AlwaysAllowApprover, PermissionGate
from argos.models.base import LLMProvider, TaskKind
from argos.models.router import ModelRouter
from argos.tools.registry import default_registry


class Case(BaseModel):
    """Un caso de evaluación con criterio de éxito verificable por código."""

    nombre: str
    prompt: str
    skill_esperada: str | None = None
    # Recibe la respuesta final en minúsculas; decide si es correcta.
    verifica: Callable[[str], bool] | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}


class CaseResult(BaseModel):
    nombre: str
    eligio_skill: bool
    parametros_ok: bool
    respuesta_ok: bool
    iteraciones: int
    segundos: float
    skills_llamadas: list[str]
    texto: str

    @property
    def aprobado(self) -> bool:
        return self.eligio_skill and self.parametros_ok and self.respuesta_ok


class Report(BaseModel):
    modelo: str
    resultados: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.resultados)

    def _tasa(self, campo: str) -> float:
        if not self.resultados:
            return 0.0
        return sum(getattr(r, campo) for r in self.resultados) / self.total

    @property
    def tasa_skill(self) -> float:
        return self._tasa("eligio_skill")

    @property
    def tasa_parametros(self) -> float:
        return self._tasa("parametros_ok")

    @property
    def tasa_respuesta(self) -> float:
        return self._tasa("respuesta_ok")

    @property
    def tasa_aprobado(self) -> float:
        return self._tasa("aprobado")

    @property
    def segundos_medios(self) -> float:
        if not self.resultados:
            return 0.0
        return sum(r.segundos for r in self.resultados) / self.total


def casos_fase1() -> list[Case]:
    """Casos contra `read_file`, `write_note` y `run_command`.

    Van de trivial a compuesto a propósito: un modelo pequeño suele acertar los
    primeros y romperse en los que exigen encadenar o elegir entre skills.
    """
    return [
        Case(
            nombre="lee_un_archivo",
            prompt="¿Qué dice la primera línea del archivo config/settings.toml?",
            skill_esperada="read_file",
            verifica=lambda t: "argos" in t or "configuración" in t or "config" in t,
        ),
        Case(
            nombre="cuenta_con_comando",
            prompt="Usa un comando para decirme cuántos archivos .py hay en el directorio src.",
            skill_esperada="run_command",
            verifica=lambda t: any(str(n) in t for n in range(5, 40)),
        ),
        Case(
            nombre="escribe_una_nota",
            prompt="Guarda una nota titulada 'prueba eval' con el texto 'el arnés funciona'.",
            skill_esperada="write_note",
            verifica=lambda t: "nota" in t or "guard" in t,
        ),
        Case(
            nombre="elige_entre_skills",
            prompt="Dime el estado de git de este repositorio.",
            skill_esperada="run_command",
            verifica=lambda t: len(t) > 10,
        ),
        Case(
            nombre="encadena_dos_skills",
            prompt=(
                "Lee el archivo pyproject.toml y guarda una nota titulada 'deps' "
                "que liste sus dependencias."
            ),
            skill_esperada="read_file",
            verifica=lambda t: "nota" in t or "guard" in t or "pydantic" in t,
        ),
        Case(
            nombre="no_inventa_si_no_existe",
            prompt="Lee el archivo inexistente_xyz.txt y dime qué contiene.",
            skill_esperada="read_file",
            # Lo correcto es reconocer que no existe, no inventarse el contenido.
            verifica=lambda t: "no existe" in t or "no encontr" in t or "no pude" in t,
        ),
        Case(
            nombre="responde_sin_herramientas",
            prompt="¿Cuánto es 17 por 3? Responde sólo con el número.",
            skill_esperada=None,  # usar una skill aquí sería un falso positivo
            verifica=lambda t: "51" in t,
        ),
    ]


def evaluar(
    provider: LLMProvider,
    casos: list[Case] | None = None,
    settings: Settings | None = None,
    verbose: bool = True,
) -> Report:
    """Corre los casos contra un proveedor y devuelve el informe."""
    casos = casos or casos_fase1()
    settings = settings or Settings()
    resultados: list[CaseResult] = []

    for caso in casos:
        # Registro limpio por caso: sin estado compartido entre pruebas.
        gate = PermissionGate(approver=AlwaysAllowApprover(), settings=settings)
        registry = default_registry(gate=gate)
        router = ModelRouter(
            providers=dict.fromkeys(TaskKind, provider),
            local_provider=provider,
            settings=settings,
        )
        loop = AgentLoop(router=router, registry=registry, settings=settings)

        inicio = time.monotonic()
        try:
            turno: TurnResult = loop.run(caso.prompt)
            texto = turno.text
            llamadas = turno.skill_calls
            iteraciones = turno.iterations
        except Exception as exc:  # un modelo local puede devolver cualquier cosa
            texto = f"(excepción: {type(exc).__name__}: {exc})"
            llamadas, iteraciones = [], 0
        segundos = time.monotonic() - inicio

        if caso.skill_esperada is None:
            eligio = not llamadas  # el acierto es NO usar herramientas
            params_ok = eligio
        else:
            eligio = caso.skill_esperada in llamadas
            # Si la skill se ejecutó y el turno produjo texto útil, los parámetros
            # pasaron la validación Pydantic; si no, habría vuelto un error.
            params_ok = eligio and "parámetros inválidos" not in texto.lower()

        respuesta_ok = bool(caso.verifica(texto.lower())) if caso.verifica else bool(texto)

        resultado = CaseResult(
            nombre=caso.nombre,
            eligio_skill=eligio,
            parametros_ok=params_ok,
            respuesta_ok=respuesta_ok,
            iteraciones=iteraciones,
            segundos=segundos,
            skills_llamadas=llamadas,
            texto=texto[:200],
        )
        resultados.append(resultado)

        if verbose:
            marca = "✓" if resultado.aprobado else "✗"
            print(
                f"  {marca} {caso.nombre:26} {segundos:5.1f}s  "
                f"skill={'sí' if eligio else 'NO':3} "
                f"params={'sí' if params_ok else 'NO':3} "
                f"resp={'sí' if respuesta_ok else 'NO':3} "
                f"[{', '.join(llamadas) or '—'}]",
                flush=True,
            )

    return Report(modelo=provider.model, resultados=resultados)


def imprimir_comparativa(informes: list[Report]) -> None:
    """Tabla final. Es lo que decide qué modelo se queda."""
    print(f"\n{'modelo':22} {'aprobado':>9} {'skill':>7} {'params':>7} {'resp':>7} {'s/caso':>8}")
    print("─" * 66)
    for informe in informes:
        print(
            f"{informe.modelo:22} "
            f"{informe.tasa_aprobado:8.0%} "
            f"{informe.tasa_skill:6.0%} "
            f"{informe.tasa_parametros:6.0%} "
            f"{informe.tasa_respuesta:6.0%} "
            f"{informe.segundos_medios:7.1f}s"
        )

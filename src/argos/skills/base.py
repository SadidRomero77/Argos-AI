"""Base de la biblioteca de skills — el 80% determinista del sistema.

Una skill es una capacidad **cerrada y verificable**: esquema de parámetros tipado,
precondiciones explícitas, ejecución determinista y criterio de éxito. El LLM elige
qué skill invocar y con qué parámetros; nunca genera la acción en sí.

Consecuencia de diseño: **toda skill debe poder probarse sin modelo**. Si una prueba
necesita un LLM, la skill está mal diseñada — la lógica se filtró al prompt.

Orden de ejecución, y ninguna etapa se salta:

    validar parámetros -> precondición -> ejecutar -> criterio de éxito
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field, ValidationError


class ConfinementError(ValueError):
    """Una ruta intentó salirse de su raíz permitida."""


def resolve_confined(candidate: str | Path, root: Path) -> Path:
    """Resuelve `candidate` garantizando que queda dentro de `root`.

    `Path.resolve()` normaliza `..` **y** sigue symlinks, así que una sola
    comprobación posterior cubre travesía por sintaxis y por enlace simbólico.
    Se resuelve primero y se compara después: al revés sería trivial de burlar.
    """
    root = root.resolve()
    candidate = Path(candidate)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if target != root and root not in target.parents:
        raise ConfinementError(f"la ruta '{candidate}' queda fuera de la raíz permitida '{root}'")
    return target


class SkillResult(BaseModel):
    """Resultado de una skill. `ok` es el criterio de éxito ya evaluado."""

    ok: bool
    output: str = ""
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok

    @classmethod
    def fail(cls, error: str) -> SkillResult:
        return cls(ok=False, error=error)

    @classmethod
    def success(cls, output: str = "", **data: Any) -> SkillResult:
        return cls(ok=True, output=output, data=data)


class Skill(ABC):
    """Capacidad determinista invocable por el agente.

    Subclases obligatorias: `name`, `description`, `Params` y `run()`.

    `description` se le muestra al modelo y es parte de la arquitectura, no
    documentación: debe decir **cuándo** invocar la skill, no sólo qué hace. Las
    descripciones prescriptivas elevan mucho la tasa de invocación correcta.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    Params: ClassVar[type[BaseModel]]

    def precondition(self, params: BaseModel) -> str | None:
        """Comprueba el estado del mundo. Devuelve el motivo si no se cumple.

        Distinto de la validación de parámetros: aquí van condiciones de runtime
        (el archivo existe, los motores están armados, hay conexión).
        """
        return None

    @abstractmethod
    def run(self, params: BaseModel) -> SkillResult:
        """Ejecuta. Sólo se llama con parámetros ya validados y precondición cumplida."""

    def execute(self, raw_params: dict[str, Any] | None = None) -> SkillResult:
        """Punto de entrada único. Nunca lanza: los fallos vuelven como SkillResult."""
        raw_params = raw_params or {}
        try:
            params = self.Params.model_validate(raw_params)
        except ValidationError as exc:
            detalles = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '(raíz)'}: {e['msg']}"
                for e in exc.errors()
            )
            return SkillResult.fail(f"parámetros inválidos para '{self.name}' -> {detalles}")

        if (motivo := self.precondition(params)) is not None:
            return SkillResult.fail(f"precondición no cumplida para '{self.name}': {motivo}")

        try:
            return self.run(params)
        except Exception as exc:
            # Una skill que revienta no debe tumbar el bucle del agente.
            return SkillResult.fail(f"'{self.name}' falló: {type(exc).__name__}: {exc}")

    @classmethod
    def tool_definition(cls) -> dict[str, Any]:
        """Definición en el formato de tool use de la API."""
        schema = cls.Params.model_json_schema()
        schema.pop("title", None)
        return {"name": cls.name, "description": cls.description, "input_schema": schema}

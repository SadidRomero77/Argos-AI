"""Ejecución de comandos con allowlist. Nunca blocklist.

Tres barreras independientes, y el fallo de cualquiera bloquea:

1. **Forma argv, no cadena.** Los parámetros son una lista `["ls", "-la"]`, no
   `"ls -la"`. Con `shell=False` no hay intérprete que pueda expandir metacaracteres:
   la inyección deja de ser posible por construcción, no por filtrado.
2. **Allowlist de ejecutables.** Lo que no está listado no corre. Una blocklist es
   inauditable: siempre falta un binario.
3. **Timeout y truncado de salida.** Un comando colgado no bloquea al agente.

La aprobación del `PermissionGate` es una cuarta barrera, aguas arriba de esta skill.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from argos.config import ROOT
from argos.skills.base import Skill, SkillResult

# Sólo lectura e inspección. Nada que escriba, borre, instale o salga a la red.
# Ampliar esta lista es una decisión de seguridad consciente, no un trámite.
DEFAULT_ALLOWLIST: frozenset[str] = frozenset(
    {"ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "file", "stat", "git", "python3"}
)

# Diagnóstico, no seguridad. Con shell=False estos caracteres llegan como literales
# a execve y son inofensivos. Sólo se miran cuando argv tiene UN solo elemento, que
# es la señal inequívoca de que el modelo escribió una línea de shell entera en vez
# de una lista. Aplicarlos a cada argumento daría falsos positivos constantes:
# `grep 'a|b'`, `python3 -c 'x; y'` y `find -name '*.py'` son todos legítimos.
_SHELL_METACHARS = ("&&", "||", "|", ";", "`", "$(", ">", "<")

_MAX_OUTPUT = 20_000


class RunCommandParams(BaseModel):
    argv: list[str] = Field(
        min_length=1,
        description=(
            "Comando como lista de argumentos, ej. ['git', 'status', '--short']. "
            "NO es una cadena de shell: las tuberías, redirecciones y '&&' no funcionan."
        ),
    )
    timeout_s: float = Field(default=20.0, gt=0, le=120, description="Segundos antes de abortar")


class RunCommand(Skill):
    name = "run_command"
    description = (
        "Ejecuta UN comando de sólo lectura y devuelve su salida. Úsala para "
        "inspeccionar el estado real del sistema (git, procesos, metadatos) en vez "
        "de suponerlo. El comando va como lista de argumentos: ['git', 'status']. "
        "NO hay shell, así que las tuberías, redirecciones y '&&' no funcionan — si "
        "necesitabas 'find ... | wc -l', usa la herramienta glob, que ya devuelve el "
        "conteo. Sólo se permiten ejecutables de una lista blanca."
    )
    Params = RunCommandParams

    allowlist: ClassVar[frozenset[str]] = DEFAULT_ALLOWLIST

    def __init__(self, allowlist: frozenset[str] | None = None, cwd: Path | None = None) -> None:
        self.allowlist = allowlist if allowlist is not None else DEFAULT_ALLOWLIST
        self.cwd = cwd or ROOT

    def precondition(self, params: RunCommandParams) -> str | None:
        ejecutable = params.argv[0]

        # Primero el diagnóstico más específico. Un único elemento con espacios o
        # metacaracteres = el modelo escribió una línea de shell entera. No es
        # explotable (shell=False), pero decirle "no está en la lista blanca" sería
        # un error opaco que le haría reintentar lo mismo; esto le dice cómo arreglarlo.
        if len(params.argv) == 1:
            unico = ejecutable
            if any(meta in unico for meta in _SHELL_METACHARS) or " " in unico.strip():
                return (
                    f"{unico!r} parece una línea de shell completa. Esta skill no usa "
                    "shell: pásame el comando ya separado, ej. ['git', 'status']. Las "
                    "tuberías y redirecciones no están disponibles; usa una llamada por "
                    "comando."
                )

        if "/" in ejecutable or "\\" in ejecutable:
            return (
                f"usa el nombre del ejecutable, no una ruta ('{ejecutable}'). "
                "Una ruta esquivaría la lista blanca."
            )

        if ejecutable not in self.allowlist:
            permitidos = ", ".join(sorted(self.allowlist))
            return f"'{ejecutable}' no está en la lista blanca. Permitidos: {permitidos}"

        if shutil.which(ejecutable) is None:
            return f"'{ejecutable}' está permitido pero no se encuentra en el sistema"

        return None

    def run(self, params: RunCommandParams) -> SkillResult:
        try:
            proc = subprocess.run(
                params.argv,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=params.timeout_s,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SkillResult.fail(f"'{params.argv[0]}' superó el timeout de {params.timeout_s}s")

        salida = (proc.stdout or "") + (f"\n[stderr]\n{proc.stderr}" if proc.stderr.strip() else "")
        if len(salida) > _MAX_OUTPUT:
            salida = salida[:_MAX_OUTPUT] + f"\n…«truncado, {len(salida)} chars en total»"

        # Un exit code != 0 es información, no una excepción: el agente decide qué hacer.
        return SkillResult(
            ok=proc.returncode == 0,
            output=salida.strip(),
            error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
            data={"returncode": proc.returncode, "argv": params.argv},
        )

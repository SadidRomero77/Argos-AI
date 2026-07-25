"""Skills de sistema de archivos, confinadas a una raíz.

El confinamiento es estructural, no una comprobación de cortesía: `resolve_confined`
resuelve la ruta (normalizando `..` y siguiendo symlinks) y sólo entonces verifica
que sigue dentro de la raíz. Ninguna ruta llega a `open()` sin pasar por ahí.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from argos.config import ROOT, Settings, get_settings
from argos.skills.base import ConfinementError, Skill, SkillResult, resolve_confined

_MAX_READ_BYTES = 200_000  # el contexto es finito; leer un binario de 2 GB no ayuda


class ReadFileParams(BaseModel):
    path: str = Field(description="Ruta relativa a la raíz del proyecto, ej. 'docs/PLAN.md'")


class ReadFile(Skill):
    name = "read_file"
    description = (
        "Lee un archivo de texto del proyecto. Úsala cuando necesites el contenido "
        "exacto de un archivo antes de responder o de modificar algo — no supongas "
        "lo que contiene. Confinada a la raíz del proyecto."
    )
    Params = ReadFileParams

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ROOT

    def precondition(self, params: ReadFileParams) -> str | None:
        try:
            target = resolve_confined(params.path, self.root)
        except ConfinementError as exc:
            return str(exc)
        if not target.exists():
            return f"'{params.path}' no existe"
        if not target.is_file():
            return f"'{params.path}' no es un archivo"
        return None

    def run(self, params: ReadFileParams) -> SkillResult:
        target = resolve_confined(params.path, self.root)
        data = target.read_bytes()
        truncado = len(data) > _MAX_READ_BYTES
        texto = data[:_MAX_READ_BYTES].decode("utf-8", errors="replace")
        if truncado:
            texto += f"\n…«truncado, el archivo tiene {len(data)} bytes»"
        return SkillResult.success(texto, path=str(target), bytes=len(data), truncated=truncado)


class WriteNoteParams(BaseModel):
    title: str = Field(description="Título corto; se convierte en el nombre del archivo")
    content: str = Field(description="Contenido de la nota, en Markdown")
    append: bool = Field(default=False, description="Añadir al final en vez de sobrescribir")


class WriteNote(Skill):
    name = "write_note"
    description = (
        "Guarda una nota persistente en el cuaderno del agente. Úsala para registrar "
        "algo que necesitarás en una sesión futura y que no se deduce del código ni "
        "del historial. No la uses como memoria de trabajo dentro de una misma tarea."
    )
    Params = WriteNoteParams

    def __init__(self, notes_dir: Path | None = None, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.notes_dir = notes_dir or settings.paths.resolved("var") / "notes"

    @staticmethod
    def _slug(title: str) -> str:
        """Nombre de archivo seguro. Evita que el título controle la ruta."""
        limpio = "".join(c if c.isalnum() or c in "-_ " else "-" for c in title.strip().lower())
        slug = "-".join(limpio.split())[:80]
        return slug or "nota"

    def precondition(self, params: WriteNoteParams) -> str | None:
        if not params.content.strip():
            return "la nota está vacía"
        return None

    def run(self, params: WriteNoteParams) -> SkillResult:
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        # El slug se calcula aquí y el confinamiento se verifica igual: defensa en
        # profundidad por si el slug dejara pasar algo inesperado.
        target = resolve_confined(f"{self._slug(params.title)}.md", self.notes_dir)

        modo = "a" if params.append else "w"
        with target.open(modo, encoding="utf-8") as fh:
            if params.append and target.stat().st_size > 0:
                fh.write("\n\n")
            fh.write(params.content.rstrip() + "\n")

        verbo = "añadida a" if params.append else "guardada en"
        return SkillResult.success(f"Nota {verbo} {target.name}", path=str(target))

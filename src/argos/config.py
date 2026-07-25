"""Configuración por capas: settings.toml -> .env -> variables de entorno.

Precedencia (gana el último): archivo TOML, luego `.env`, luego el entorno real.
Los secretos viven sólo en `.env` / entorno y nunca en el TOML versionado.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# Raíz del repo: config.py está en src/argos/, así que subimos dos niveles.
ROOT = Path(__file__).resolve().parents[2]
SETTINGS_TOML = ROOT / "config" / "settings.toml"


class ModelSettings(BaseModel):
    routine: str = "claude-haiku-4-5"
    reasoning: str = "claude-sonnet-5"
    planning: str = "claude-opus-5"
    local_only: bool = False
    local_url: str = "http://localhost:11434/v1"
    local_model: str = "qwen3:4b"


class AgentSettings(BaseModel):
    """Con quién habla el agente por defecto."""

    # Su perfil se inyecta SIEMPRE en el contexto, no por búsqueda vectorial.
    user: str = "Sadid"
    # true = voces neuronales de Edge (mejor calidad, el texto sale del equipo).
    # false = síntesis del navegador (peor, pero nada sale de la máquina).
    voice: bool = True


class BudgetSettings(BaseModel):
    """Límites por tarea. El agente los ve y se autolimita en vez de ser cortado."""

    max_iterations: int = 12
    max_tokens_task: int = 60_000
    max_usd_per_day: float = 2.0


class PathSettings(BaseModel):
    var: Path = Path("var")
    traces: Path = Path("var/traces")
    memory_db: Path = Path("var/memory.db")
    permissions: Path = Path("config/permissions.yaml")
    system_prompt: Path = Path("config/prompts/system.md")
    identity_prompt: Path = Path("config/prompts/identity.md")

    def resolved(self, field: str) -> Path:
        """Ruta absoluta, anclada a la raíz del repo si venía relativa."""
        value: Path = getattr(self, field)
        return value if value.is_absolute() else ROOT / value


def _toml_source(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


class TomlSource(PydanticBaseSettingsSource):
    """Lee config/settings.toml como capa de menor precedencia."""

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        raise NotImplementedError  # no se usa: sobrescribimos __call__

    def __call__(self) -> dict[str, Any]:
        return _toml_source(SETTINGS_TOML)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARGOS_",
        env_nested_delimiter="__",
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    models: ModelSettings = Field(default_factory=ModelSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    paths: PathSettings = Field(default_factory=PathSettings)

    # Secretos. `SecretStr` evita que aparezcan en repr(), logs o trazas por accidente.
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    bus_key: SecretStr | None = Field(default=None, alias="ARGOS_BUS_KEY")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Orden = precedencia descendente: lo primero gana.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Instancia única. Cacheada para que la config no cambie a mitad de un turno."""
    return Settings()

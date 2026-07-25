"""Integración real contra un servidor local. Excluida del bucle rápido.

    uv run pytest -m needs_llm

Estas son las únicas pruebas del repo que necesitan infraestructura. Se saltan
solas si no hay servidor, para que `uv run pytest` siga funcionando en cualquier
máquina y en CI.

Cubren lo que los dobles de prueba no pueden: que la traducción de formatos
funcione contra un servidor de verdad, y que los modos de fallo observados en
modelos pequeños sigan estando cubiertos por el harness.
"""

from __future__ import annotations

import httpx
import pytest

from argos.config import Settings
from argos.harness.loop import AgentLoop
from argos.harness.permissions import AlwaysAllowApprover, PermissionGate
from argos.models.base import TaskKind
from argos.models.local import LocalProvider
from argos.models.router import ModelRouter
from argos.tools.registry import default_registry

MODELO = "qwen3:4b"
URL = "http://localhost:11434/v1"


def _hay_servidor() -> bool:
    try:
        httpx.get(f"{URL}/models", timeout=2.0).raise_for_status()
    except httpx.HTTPError:
        return False
    return True


pytestmark = [
    pytest.mark.needs_llm,
    pytest.mark.skipif(not _hay_servidor(), reason="no hay servidor local escuchando"),
]


@pytest.fixture
def loop():
    settings = Settings()
    provider = LocalProvider(MODELO, base_url=URL, num_ctx=16384)
    registry = default_registry(
        gate=PermissionGate(approver=AlwaysAllowApprover(), settings=settings)
    )
    router = ModelRouter(
        providers=dict.fromkeys(TaskKind, provider),
        local_provider=provider,
        settings=settings,
    )
    return AgentLoop(router=router, registry=registry, settings=settings)


def test_el_modelo_declara_soporte_de_herramientas():
    """Sin `tools` en capabilities no hay agente posible: se detecta antes de nada."""
    provider = LocalProvider(MODELO, base_url=URL)
    ok, mensaje = provider.health()
    assert ok, mensaje


def test_ciclo_completo_con_una_skill(loop):
    """La prueba de integración mínima: pedir, elegir skill, ejecutar, responder."""
    result = loop.run("¿Cuántos archivos .py hay en el directorio src?")

    assert result.ok
    assert result.skill_calls, "no invocó ninguna skill"
    assert any(s in result.skill_calls for s in ("glob", "run_command"))
    # 27 hoy; la prueba sólo exige que dé un número plausible, no uno exacto,
    # para no romperse cada vez que se añade un archivo.
    assert any(str(n) in result.text for n in range(10, 200))


def test_una_respuesta_vacia_no_pasa_por_turno_completado(loop):
    """Regresión del bug que más costó encontrar.

    Qwen3 puede gastar todo su presupuesto en `reasoning` y devolver content="".
    Antes eso se contaba como turno terminado con éxito, con la tarea sin hacer.
    """
    result = loop.run("Lee el archivo config/settings.toml y dime qué modelo local usa.")

    # O bien produjo texto, o bien el harness lo corrigió. Lo inaceptable es
    # terminar en silencio sin ninguna de las dos cosas.
    assert result.text.strip() or result.nudges > 0

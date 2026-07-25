"""Aislamiento del entorno para toda la suite.

Sin esto, las pruebas leerían el `.env` y las variables `ARGOS_*` de la máquina
donde corren: pasarían en el portátil de quien las escribió y fallarían en CI, o
al revés. Peor aún, una prueba podría gastar dinero real si encontrara una API key.

Cada prueba arranca con: sin variables de entorno de ARGOS, sin `.env`, y con la
caché de configuración limpia. Lo que necesite valores concretos los pone con
`monkeypatch.setenv`, que se aplica después de este fixture.
"""

from __future__ import annotations

import os

import pytest

from argos.config import Settings, get_settings

_PREFIJOS = ("ARGOS_", "ANTHROPIC_", "GEMINI_")


@pytest.fixture(autouse=True)
def entorno_hermetico(monkeypatch, tmp_path):
    for var in list(os.environ):
        if var.startswith(_PREFIJOS):
            monkeypatch.delenv(var, raising=False)

    # Apunta el env_file a un archivo inexistente: el .env real queda fuera.
    monkeypatch.setitem(Settings.model_config, "env_file", tmp_path / "sin-env")

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

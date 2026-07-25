"""La config carga en el orden correcto y los secretos no se filtran por repr()."""

from __future__ import annotations

import pytest

from argos.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_carga_defaults_del_toml():
    s = Settings()
    assert s.models.routine == "claude-haiku-4-5"
    assert s.budget.max_iterations == 12


def test_variable_de_entorno_gana_sobre_toml(monkeypatch):
    monkeypatch.setenv("ARGOS_MODELS__ROUTINE", "modelo-de-prueba")
    monkeypatch.setenv("ARGOS_BUDGET__MAX_ITERATIONS", "3")
    s = Settings()
    assert s.models.routine == "modelo-de-prueba"
    assert s.budget.max_iterations == 3


def test_local_only_es_conmutable_por_entorno(monkeypatch):
    """El interruptor de 'nunca llamar a una API externa' debe ser trivial de activar."""
    monkeypatch.setenv("ARGOS_MODELS__LOCAL_ONLY", "true")
    assert Settings().models.local_only is True


def test_rutas_relativas_se_anclan_a_la_raiz_del_repo():
    resolved = Settings().paths.resolved("traces")
    assert resolved.is_absolute()
    assert resolved.name == "traces"


def test_el_secreto_no_aparece_en_repr_ni_en_str(monkeypatch):
    """SecretStr es la primera línea de defensa contra fugas en logs."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-valor-secreto-de-prueba")
    s = Settings()
    assert "sk-ant-valor-secreto-de-prueba" not in repr(s)
    assert "sk-ant-valor-secreto-de-prueba" not in str(s.anthropic_api_key)
    # ...pero sigue siendo accesible de forma explícita
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-valor-secreto-de-prueba"

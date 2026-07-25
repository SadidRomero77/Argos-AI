"""Catálogo de proveedores: cambiar de LLM sin tocar código ni reiniciar.

La prueba que más importa aquí es de seguridad: **una clave de API nunca debe
acabar en el YAML versionado ni en lo que la interfaz muestra**.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from argos.config import Settings
from argos.models.providers import (
    DEFAULT_PROVIDERS,
    ProviderCatalog,
    ProviderKind,
    ProviderSpec,
    SecretStore,
)


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGOS_PATHS__VAR", str(tmp_path / "var"))
    return ProviderCatalog(settings=Settings(), path=tmp_path / "providers.yaml")


# ───────────────────────────── almacén de claves ──────────────────────────────


def test_la_clave_se_guarda_con_permisos_restrictivos(tmp_path):
    """0600. Crearlo abierto y ajustarlo después deja una ventana de lectura."""
    almacen = SecretStore(tmp_path / "secrets.json")
    almacen.set("ANTHROPIC_API_KEY", "sk-ant-secreta")

    modo = stat.S_IMODE(os.stat(almacen.path).st_mode)
    assert modo == 0o600, f"permisos {oct(modo)}; otro usuario podría leer la clave"


def test_recupera_y_borra_una_clave(tmp_path):
    almacen = SecretStore(tmp_path / "secrets.json")
    almacen.set("K", "valor")
    assert almacen.get("K") == "valor"
    assert almacen.delete("K")
    assert almacen.get("K") is None


def test_el_entorno_gana_sobre_el_archivo(tmp_path, monkeypatch):
    """Permite inyectar la clave en CI o en un contenedor sin escribir en disco."""
    almacen = SecretStore(tmp_path / "secrets.json")
    almacen.set("K", "del-archivo")
    monkeypatch.setenv("K", "del-entorno")
    assert almacen.get("K") == "del-entorno"


def test_known_keys_devuelve_nombres_pero_nunca_valores(tmp_path):
    almacen = SecretStore(tmp_path / "secrets.json")
    almacen.set("ANTHROPIC_API_KEY", "sk-ant-secretisima")

    assert almacen.known_keys() == ["ANTHROPIC_API_KEY"]
    assert "sk-ant-secretisima" not in json.dumps(almacen.known_keys())


def test_un_archivo_de_secretos_corrupto_no_impide_arrancar(tmp_path):
    ruta = tmp_path / "secrets.json"
    ruta.write_text("{esto no es json", encoding="utf-8")
    assert SecretStore(ruta).get("K") is None


# ──────────────────────────────── catálogo ────────────────────────────────────


def test_arranca_con_los_proveedores_por_defecto(catalog):
    nombres = {spec.name for spec in catalog.specs()}
    assert nombres == {spec.name for spec in DEFAULT_PROVIDERS}
    assert catalog.active_name == "local"


def test_el_yaml_guardado_NO_contiene_claves(catalog, tmp_path):
    """El control de seguridad central: el archivo va a git."""
    catalog.set_secret("ANTHROPIC_API_KEY", "sk-ant-jamas-debe-aparecer")
    catalog.activate("claude-haiku")

    contenido = (tmp_path / "providers.yaml").read_text(encoding="utf-8")

    assert "sk-ant-jamas-debe-aparecer" not in contenido
    assert "ANTHROPIC_API_KEY" in contenido  # el NOMBRE sí, el valor no


def test_status_expone_si_falta_la_clave_pero_no_la_clave(catalog):
    estado = {p["name"]: p for p in catalog.status()}

    assert estado["local"]["ready"] is True  # local no necesita clave
    assert estado["claude-haiku"]["ready"] is False  # aún sin clave
    assert estado["claude-haiku"]["needs_secret"] == "ANTHROPIC_API_KEY"
    assert "sk-" not in json.dumps(catalog.status())


def test_activar_un_proveedor_sin_clave_falla_con_mensaje_accionable(catalog):
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        catalog.activate("claude-haiku")


def test_tras_poner_la_clave_ya_se_puede_activar(catalog):
    catalog.set_secret("ANTHROPIC_API_KEY", "sk-ant-valida")
    spec = catalog.activate("claude-haiku")

    assert spec.model == "claude-haiku-4-5"
    assert catalog.active_name == "claude-haiku"


def test_activar_uno_inexistente_lista_los_disponibles(catalog):
    with pytest.raises(KeyError, match="local"):
        catalog.activate("no-existe")


def test_la_seleccion_persiste_entre_reinicios(tmp_path, monkeypatch):
    """Elegir proveedor en la interfaz debe sobrevivir a cerrar el programa."""
    monkeypatch.setenv("ARGOS_PATHS__VAR", str(tmp_path / "var"))
    ruta = tmp_path / "providers.yaml"

    primero = ProviderCatalog(settings=Settings(), path=ruta)
    primero.activate("local-8b")

    segundo = ProviderCatalog(settings=Settings(), path=ruta)
    assert segundo.active_name == "local-8b"


def test_se_puede_anadir_un_proveedor_openai_compatible(catalog):
    """Un solo tipo cubre Groq, OpenRouter, DeepSeek, vLLM, LM Studio…"""
    catalog.set_secret("GROQ_API_KEY", "gsk-loquesea")
    catalog.add(
        ProviderSpec(
            name="groq",
            kind=ProviderKind.OPENAI,
            model="llama-3.3-70b",
            base_url="https://api.groq.com/openai/v1",
            secret_key="GROQ_API_KEY",
            label="Groq · llama 3.3 70B",
        )
    )
    spec = catalog.activate("groq")

    assert spec.is_local is False
    provider = catalog.build("groq")
    # La clave viaja como cabecera Authorization, no en la URL ni en el cuerpo.
    assert provider.headers["Authorization"] == "Bearer gsk-loquesea"


def test_un_proveedor_local_no_lleva_cabecera_de_autorizacion(catalog):
    provider = catalog.build("local")
    assert provider.headers == {}
    assert provider.model == "qwen3:4b"


def test_is_local_distingue_correctamente(catalog):
    assert catalog.get("local").is_local is True
    assert catalog.get("claude-haiku").is_local is False


def test_eliminar_el_proveedor_activo_reasigna_otro(catalog):
    """La interfaz debe poder arrancar siempre, aunque borres el que estaba activo."""
    catalog.remove("local")
    assert catalog.active_name != "local"
    assert catalog.active_name in {s.name for s in catalog.specs()}


def test_si_el_activo_desaparece_del_archivo_se_cae_al_primero(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGOS_PATHS__VAR", str(tmp_path / "var"))
    ruta = tmp_path / "providers.yaml"
    ruta.write_text(
        "active: fantasma\nproviders:\n"
        "  - name: local\n    kind: openai\n    model: qwen3:4b\n"
        "    base_url: http://localhost:11434/v1\n",
        encoding="utf-8",
    )

    catalog = ProviderCatalog(settings=Settings(), path=ruta)
    assert catalog.active_name == "local"

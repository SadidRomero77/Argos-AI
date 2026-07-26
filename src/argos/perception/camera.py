"""Cliente del puente de cámara. Vive en WSL; el puente vive en Windows.

## Cómo se descubre el puente, y por qué así

WSL2 no ve la webcam, así que la cámara corre como **nodo publicador** en Windows
(`bridge/camera_bridge.py`). El agente lo consume por HTTP.

Encontrarlo tiene un matiz que costó tiempo: en este equipo, con red *mirrored*,
`localhost` **no** conecta WSL con Windows en ninguna dirección. Lo que sí
funciona es la interfaz de Tailscale. Por eso no se asume una dirección: se
prueban varias y se recuerda la que respondió. Así el mismo código sirve si mañana
el puente corre en la Raspberry Pi o en otra máquina de la red.

## Autenticación

Una webcam servida por HTTP sin autenticar es una cámara de vigilancia abierta
para cualquiera del WiFi. El puente exige un token que genera solo y guarda en el
perfil de usuario de Windows; WSL lo lee del mismo archivo a través de `/mnt/c`.
Ninguno de los dos lados necesita configuración manual y el token nunca entra en
el repositorio.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import httpx

PUERTO = 8710

# Rutas candidatas del token, en orden. La primera es la que usa el puente en
# Windows; la segunda permite forzarlo en pruebas o en otra máquina.
_RUTAS_TOKEN = (
    pathlib.Path("/mnt/c/Users") / os.environ.get("WIN_USER", "sadid") / ".argos-bridge-token",
    pathlib.Path.home() / ".argos-bridge-token",
)


def leer_token() -> str:
    for ruta in _RUTAS_TOKEN:
        try:
            if ruta.is_file() and (valor := ruta.read_text(encoding="utf-8").strip()):
                return valor
        except OSError:
            continue
    return ""


def _candidatas() -> list[str]:
    """Direcciones donde puede estar el puente, de la más probable a la menos."""
    vistas: list[str] = []
    if fijado := os.environ.get("ARGOS_CAMERA_URL"):
        vistas.append(fijado.rstrip("/"))

    # Todas las IPv4 que ve WSL: en modo mirrored incluyen las de Windows, y una
    # de ellas es la que funciona. Cuál, depende del equipo — por eso se prueban.
    try:
        import socket
        import subprocess

        salida = subprocess.run(
            ["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=3, check=False
        ).stdout
        for linea in salida.splitlines():
            partes = linea.split()
            if len(partes) > 3 and partes[2] == "inet":
                ip = partes[3].split("/")[0]
                if not ip.startswith("127."):
                    vistas.append(f"http://{ip}:{PUERTO}")
        socket  # noqa: B018  importado por claridad del bloque
    except Exception:
        pass

    vistas.append(f"http://127.0.0.1:{PUERTO}")
    # Sin duplicados, conservando el orden.
    return list(dict.fromkeys(vistas))


@dataclass(slots=True)
class Frame:
    jpeg: bytes
    width: int = 0
    height: int = 0
    source: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.jpeg


class CameraBridge:
    """Cliente HTTP del puente. Descubre la dirección una vez y la reutiliza."""

    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.token = token if token is not None else leer_token()
        self._client = httpx.Client(timeout=httpx.Timeout(8.0, connect=2.0))

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-Argos-Token": self.token} if self.token else {}

    def discover(self) -> str | None:
        """Encuentra el puente y memoriza dónde estaba."""
        if self.base_url is not None:
            return self.base_url
        for candidata in _candidatas():
            try:
                r = self._client.get(f"{candidata}/health", headers=self._headers, timeout=2.0)
                if r.status_code == 200:
                    self.base_url = candidata
                    return candidata
                if r.status_code == 401:
                    # Está ahí, pero el token no vale. Distinguirlo de "no está"
                    # ahorra mucho tiempo de diagnóstico.
                    self.base_url = candidata
                    return candidata
            except httpx.HTTPError:
                continue
        return None

    def health(self) -> tuple[bool, str]:
        base = self.discover()
        if base is None:
            return False, (
                "no encuentro el puente de cámara. Arráncalo en Windows con "
                "`python argos-bridge\\camera_bridge.py`."
            )
        try:
            r = self._client.get(f"{base}/health", headers=self._headers)
        except httpx.HTTPError as exc:
            return False, f"el puente no responde en {base} ({type(exc).__name__})"

        if r.status_code == 401:
            return False, (
                f"el puente responde en {base} pero rechaza el token. Comprueba que "
                f"{_RUTAS_TOKEN[0]} existe y que el puente se reinició después de crearlo."
            )
        datos = r.json()
        if not datos.get("ok"):
            return False, f"el puente aún no tiene imagen: {datos.get('error') or 'iniciando'}"
        return True, (
            f"cámara {datos.get('width')}x{datos.get('height')} @ {datos.get('fps')} fps en {base}"
        )

    def grab(self) -> Frame:
        base = self.discover()
        if base is None:
            return Frame(jpeg=b"")
        r = self._client.get(f"{base}/frame.jpg", headers=self._headers)
        r.raise_for_status()
        return Frame(jpeg=r.content, source=base)

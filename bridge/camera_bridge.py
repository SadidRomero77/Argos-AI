"""Nodo sensor de cámara. **Corre en Windows, no en WSL.**

WSL2 no ve la webcam: no hay `/dev/video*` y pasar el USB exige `usbipd-win` con
permisos de administrador. En vez de pelearse con eso, la cámara se convierte en
un **nodo que publica** — que es exactamente la arquitectura que la Fase 4 va a
necesitar para la Raspberry Pi. Lo que aquí resuelve un problema de WSL, allí
será el diseño normal.

Arranque (desde Windows, sin administrador):

    python bridge\\camera_bridge.py

Queda escuchando en `http://127.0.0.1:8710` con dos rutas:

    GET /frame.jpg   el fotograma más reciente
    GET /health      estado, resolución y fotogramas por segundo

Como WSL está en modo de red *mirrored*, el agente lo alcanza en ese mismo
`localhost` sin configurar nada.

## Por qué un hilo captura y el servidor sólo sirve

Abrir la cámara en cada petición tarda entre uno y dos segundos: el agente
pediría una imagen y llegaría vieja. Aquí un hilo mantiene el dispositivo abierto
y sobrescribe siempre el último fotograma; el servidor entrega esa copia. La
consecuencia buscada es que **nunca se sirve una imagen atrasada**: si el agente
pregunta qué ve, ve el ahora, no una cola de fotogramas acumulados.

Sin dependencias fuera de `opencv-python`; el servidor es de la biblioteca estándar.
"""

from __future__ import annotations

import argparse
import hmac
import os
import pathlib
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_ultimo: bytes | None = None
_lock = threading.Lock()
_estado: dict[str, object] = {"fps": 0.0, "width": 0, "height": 0, "frames": 0, "error": None}
_TOKEN = ""

# Token compartido entre este puente (Windows) y el agente (WSL). Se guarda en el
# perfil del usuario, que WSL lee montado en /mnt/c: así ninguno de los dos lados
# necesita configuración manual y el token nunca pasa por el repositorio.
RUTA_TOKEN = pathlib.Path(os.path.expanduser("~")) / ".argos-bridge-token"


def cargar_token() -> str:
    """Lee el token, o lo crea si es el primer arranque."""
    if RUTA_TOKEN.is_file() and (guardado := RUTA_TOKEN.read_text(encoding="utf-8").strip()):
        return guardado
    nuevo = secrets.token_urlsafe(32)
    RUTA_TOKEN.write_text(nuevo, encoding="utf-8")
    return nuevo


def capturar(indice: int, ancho: int, alto: int, calidad: int) -> None:
    """Mantiene la cámara abierta y guarda siempre el fotograma más reciente."""
    global _ultimo

    # CAP_DSHOW evita el arranque lento de Media Foundation en Windows.
    camara = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    # Búfer de 1: sin esto la cámara acumula fotogramas y el agente vería el
    # pasado. Preferimos perder imágenes antes que servir una atrasada.
    camara.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camara.isOpened():
        _estado["error"] = f"no se pudo abrir la cámara {indice}"
        return

    marca, contador = time.monotonic(), 0
    try:
        while True:
            ok, fotograma = camara.read()
            if not ok:
                time.sleep(0.1)
                continue

            ok, jpeg = cv2.imencode(".jpg", fotograma, [cv2.IMWRITE_JPEG_QUALITY, calidad])
            if not ok:
                continue

            with _lock:
                _ultimo = jpeg.tobytes()

            contador += 1
            if (ahora := time.monotonic()) - marca >= 1.0:
                _estado.update(
                    fps=round(contador / (ahora - marca), 1),
                    height=fotograma.shape[0],
                    width=fotograma.shape[1],
                    frames=int(_estado["frames"]) + contador,
                )
                marca, contador = ahora, 0
    finally:
        camara.release()


class Handler(BaseHTTPRequestHandler):
    def _autorizado(self) -> bool:
        """Compara en tiempo constante. Sin token válido no se sirve la cámara."""
        recibido = self.headers.get("X-Argos-Token", "")
        if not recibido:
            # También por query, para poder abrirlo en el navegador y comprobar.
            _, _, cola = self.path.partition("?")
            for par in cola.split("&"):
                clave, _, valor = par.partition("=")
                if clave == "token":
                    recibido = valor
        return hmac.compare_digest(recibido, _TOKEN)

    def do_GET(self) -> None:
        if not self._autorizado():
            # Una webcam sin autenticar en la red es una cámara de vigilancia
            # para cualquiera. El token es lo único que separa una cosa de la otra.
            self._json(401, {"error": "falta o es inválido X-Argos-Token"})
            return

        if self.path.startswith("/frame"):
            with _lock:
                datos = _ultimo
            if datos is None:
                self._json(503, {"error": _estado["error"] or "aún no hay imagen"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(datos)))
            # Sin caché: cada petición debe traer el instante actual.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(datos)

        elif self.path.startswith("/health"):
            with _lock:
                listo = _ultimo is not None
            self._json(200, {"ok": listo, **_estado})

        else:
            self._json(404, {"error": "usa /frame.jpg o /health"})

    def _json(self, codigo: int, cuerpo: dict) -> None:
        import json

        datos = json.dumps(cuerpo).encode()
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def log_message(self, *_: object) -> None:
        """Silencio. Una línea por fotograma haría el log inservible."""


def main() -> int:
    parser = argparse.ArgumentParser(description="Puente de cámara de ARGOS (Windows)")
    parser.add_argument("--camera", type=int, default=0, help="índice de la cámara")
    parser.add_argument("--port", type=int, default=8710)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--quality", type=int, default=80, help="calidad JPEG, 1-100")
    # Por defecto sólo escucha en localhost. Abrirlo a la red expondría tu
    # webcam a cualquiera del WiFi: que sea una decisión explícita.
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="interfaz de escucha; el acceso lo controla el token, no esto",
    )
    args = parser.parse_args()

    global _TOKEN
    _TOKEN = cargar_token()

    hilo = threading.Thread(
        target=capturar,
        args=(args.camera, args.width, args.height, args.quality),
        daemon=True,
    )
    hilo.start()

    for _ in range(50):  # hasta 5 s esperando el primer fotograma
        with _lock:
            if _ultimo is not None:
                break
        if _estado["error"]:
            print(f"error: {_estado['error']}")
            return 1
        time.sleep(0.1)

    print(f"camara  : {_estado['width']}x{_estado['height']} @ {_estado['fps']} fps")
    print(f"sirviendo: puerto {args.port} · token en {RUTA_TOKEN}")
    print("Ctrl+C para parar")

    servidor = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\ndetenido")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

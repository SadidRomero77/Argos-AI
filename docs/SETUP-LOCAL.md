# Modelos locales (sin API key)

Ollama está instalado **en espacio de usuario**, en `~/.local`. No requiere sudo y se
desinstala borrando tres rutas (ver abajo).

## Arrancar el servidor

No hay servicio systemd, así que **no arranca solo tras un reinicio**:

```bash
nohup ollama serve > ~/.local/ollama-dl/serve.log 2>&1 &
```

Comprobar que responde y que ve la GPU:

```bash
curl -s http://localhost:11434/api/version && grep -i "inference compute" ~/.local/ollama-dl/serve.log | tail -1
```

Debe aparecer `library=CUDA ... NVIDIA GeForce RTX 3060 Laptop GPU`. Si dice `library=cpu`,
la inferencia irá diez veces más lenta.

## Usar el agente en local

```bash
argos --local --approve ask "¿qué dice el archivo config/settings.toml?"
```

El CLI comprueba el servidor **antes** de empezar el turno; si no hay nada escuchando lo
dice en tres segundos en vez de esperar el timeout.

Para fijarlo como modo por defecto, en `config/settings.toml`:

```toml
[models]
local_only = true
local_model = "qwen3:4b"
```

## Ejecutar el banco de evaluación

Es lo que decide si un modelo sirve como cerebro:

```bash
uv run python -m argos.evals qwen3:4b qwen3:8b
```

## Restricciones reales de esta máquina

| Dato | Valor |
|---|---|
| VRAM total / libre | 6,0 GiB / ~5,0 GiB (Windows consume el resto) |
| `qwen3:4b` (Q4_K_M) | 2,5 GB — entra entera en VRAM |
| `qwen3:8b` (Q4_K_M) | 5,2 GB — **no cabe**, parte va a CPU |

**`num_ctx` es la trampa silenciosa.** Ollama arranca en **4096 tokens**, que un agente
agota de inmediato entre el prompt de sistema y las definiciones de skills. Al desbordar,
el servidor **trunca por el principio y el modelo pierde sus instrucciones sin emitir
ningún error**: se manifiesta como un agente que "se vuelve tonto" a mitad de la tarea.
El `LocalProvider` pide 16384 por defecto.

**Qwen3 razona antes de responder.** En una prueba, contestar "51" a la pregunta `17*3`
consumió 307 tokens de salida. Eso multiplica la latencia de cada turno y es inherente a
la familia, no un fallo de configuración.

## Desinstalar

```bash
rm -rf ~/.local/bin/ollama ~/.local/lib/ollama ~/.ollama ~/.local/ollama-dl
```

Los modelos descargados (~7,7 GB) están en `~/.ollama/models`.

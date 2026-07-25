# Estado del proyecto

Actualizado: 2026-07-25 · 15 commits · 3.681 líneas · 144 pruebas rápidas + 3 de integración

## Mapa de fases

| Fase | Qué entrega | Estado |
|---|---|---|
| **0 — Fundación** | Repo, config, trazas | ✅ **terminada** |
| **1 — El cerebro** | Agente de escritorio autónomo | 🔵 **~60%** |
| 2 — Visión e identidad | El agente ve y reconoce personas | ⬜ sin empezar |
| 3 — Voz y rostro | Terminal físico ESP32-S3 | ⬜ sin empezar |
| 4 — El cuerpo | Robot móvil con reflejos | ⬜ sin empezar |
| 5 — Navegación | Objetivos en lenguaje natural | ⬜ sin empezar |
| 6 — Autonomía real | Metas propias, curiosidad | ⬜ sin empezar |

## Fase 1 en detalle

| Sub-fase | Qué es | Estado |
|---|---|---|
| 1.1 Harness | `AgentLoop` + `PermissionGate` | ✅ hecho |
| 1.2 Router de modelos | Claude + local (Ollama) | ✅ hecho |
| 1.4 Biblioteca de skills | `glob`, `read_file`, `write_note`, `run_command` | ✅ hecho |
| — Banco de evaluación | Mide tool-calling contra skills reales | ✅ hecho |
| **1.3 Memoria** | **Episódica, semántica, procedimental + entidades** | ⬜ **siguiente** |
| 1.5 Cliente MCP | Conectar servidores MCP externos | ⬜ pendiente |
| 1.6 Autonomía | `GoalManager`, `Scheduler`, `Reflector` | ⬜ pendiente |
| — `ContextCurator` | Compactación cuando el contexto se llena | ⬜ pendiente |

**Lo que falta para cerrar la Fase 1** es, en orden: memoria (el bloque grande, sin
ella el agente no tiene continuidad entre sesiones), autonomía (sin ella sólo
reacciona, no persigue metas propias), y luego MCP y compactación de contexto.

## Qué funciona hoy, comprobado

- Agente que razona, elige skills y las ejecuta — **sin API key**, con modelo local
- Permisos `deny → ask → allow` con control por fase; falla cerrado en todos los caminos
- Tres topes de presupuesto: iteraciones, tokens por tarea, dólares por día
- Traza JSONL de auditoría con redacción de secretos
- Banco de evaluación: **qwen3:4b saca 100%** en los 7 casos, 26,7 s de media

## Qué NO funciona / no existe todavía

- **El agente no recuerda nada entre ejecuciones.** Cada invocación arranca en blanco.
- **No tiene metas propias.** Sólo responde cuando se le pregunta; no despierta solo.
- **Nunca ha hablado con la API de Claude.** Todo el código de `ClaudeProvider` está
  probado contra dobles, no contra el servicio real.
- No hay visión, ni voz, ni hardware.
- El servidor de Ollama **no arranca solo tras reiniciar** (ver `SETUP-LOCAL.md`).

## Decisiones tomadas y dónde están

| Decisión | Dónde |
|---|---|
| El LLM nunca cierra un lazo de control (80/20) | `CLAUDE.md`, `docs/PLAN.md` |
| `qwen3:4b` como modelo local por defecto | `docs/adr/002-modelo-local.md` |
| Bus propio ligero en vez de ROS 2 | `docs/PLAN.md` |
| LAFVIN ESP32-S3 = terminal fijo de escritorio | `docs/PLAN.md` |
| Dónde vive el cerebro con el robot móvil | **pendiente — se decide en Fase 2 con datos** |

## Los cuatro fallos que encontró el banco de evaluación

Ninguno lo habría visto una prueba unitaria, porque dependen de cómo se comporta un
modelo pequeño de verdad:

1. **`num_ctx=4096`** — Ollama trunca por el principio al desbordar y el modelo pierde
   su prompt de sistema **sin emitir error**.
2. **Campo `reasoning` ignorado** — el modelo agotaba su presupuesto pensando, devolvía
   `content=""` y el turno se daba por completado **con la tarea sin hacer**.
3. **Nombres alucinados** — descripciones en español + nombres en inglés llevaron al
   modelo a invocar una función `busca_archivos` que no existe.
4. **Sustituyó una respuesta correcta por una peor** — tenía el conteo bueno de `glob`
   y lo descartó tras leer un número de inodo de `find -ls` como si fuera un tamaño.

Los cuatro están arreglados. El patrón de (3) y (4): **el fallo era del diseño, no del
modelo**. La solución nunca fue un modelo más listo sino skills mejor formadas.

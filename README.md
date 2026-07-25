# ARGOS

Agente de IA autónomo, encarnado progresivamente en hardware hasta llegar a un robot
móvil que toma sus propias decisiones. Plan completo por fases en [`docs/PLAN.md`](docs/PLAN.md).

**Estado: Fase 1 en curso.** Hay un agente de escritorio funcional que razona, invoca
skills, respeta permisos y presupuestos, y corre **sin API key** contra un modelo local.

## Empezar

```bash
uv sync
uv run pytest            # 145 pruebas; sin API key, GPU ni servidor
```

Con un modelo local (ver [`docs/SETUP-LOCAL.md`](docs/SETUP-LOCAL.md)):

```bash
argos --local --approve ask "¿cuántos archivos .py hay en src?"
```

Con la API de Claude — copiar `.env.example` a `.env` y poner la clave:

```bash
argos --approve ask "resume el estado del repositorio"
```

## La idea central

> **El LLM nunca cierra un lazo de control.**

El modelo **elige una skill y sus parámetros**; el código determinista **ejecuta y
verifica**. Nunca genera trayectorias, comandos de motor ni acciones arbitrarias. La
proporción código/IA no es fija: crece a favor del código cuanto más cerca se está del
actuador — 100/0 en los reflejos de seguridad, ~20/80 en planificación.

Tres consecuencias prácticas, que son la razón del diseño:

- **Testeable sin modelo.** Cada skill tiene su prueba determinista. Es la única forma de
  depurar un robot sin quemar tokens en cada iteración.
- **Acotado por construcción.** Una alucinación produce como máximo *una skill mal
  elegida*, nunca un comando arbitrario.
- **Degradable.** Si el cerebro cae, las skills y los reflejos siguen funcionando.

## Qué hay construido

| Módulo | |
|---|---|
| `harness/permissions.py` | `deny → ask → allow` + control por fase. Falla cerrado en todos los caminos |
| `harness/loop.py` | Bucle con tres topes (iteraciones, tokens, dólares) y marcado de contenido externo |
| `skills/` | `glob`, `read_file`, `write_note`, `run_command` — confinadas y con allowlist |
| `models/` | Router por tipo de tarea; proveedores Claude y local (Ollama / llama.cpp) |
| `evals/` | Banco de tool-calling contra las skills reales |
| `trace.py` | JSONL append-only con redacción de secretos y biométricos |

## Evaluación

Lo que decide si un modelo sirve como cerebro no es perplexity, es si elige bien la skill
y le pasa parámetros válidos:

```bash
uv run python -m argos.evals qwen3:4b qwen3:8b
```

## Seguridad

Alineado con OWASP Top 10 for Agentic Applications 2026. El principio operativo es
**Least Agency**: la autonomía se gana por fase, no viene de fábrica. Una skill de
hardware no es invocable hoy aunque alguien le ponga `allow` en el YAML por descuido.

Detalles en [`CLAUDE.md`](CLAUDE.md) y en la sección de ciberseguridad del plan.

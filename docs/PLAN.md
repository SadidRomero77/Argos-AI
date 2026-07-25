# Proyecto ARGOS — Agente de IA Física Autónomo

## Contexto

El objetivo es construir un agente de IA autónomo capaz de tomar decisiones propias, encarnado
progresivamente en hardware que ya se posee, hasta llegar a un robot móvil no teleoperado.

El problema explícito a resolver **no es técnico sino de método**: de los proyectos anteriores sólo
se concluyó el de la Raspberry Pi. Por eso este plan está estructurado para que **cada fase termine
en un artefacto funcional, demostrable y usable por sí mismo**, aunque el resto del proyecto nunca
se construya. Si el proyecto se detiene en la Fase 1, queda un agente de escritorio útil. Si se
detiene en la Fase 3, queda un asistente de voz físico. Nada queda a medias.

Directorio de trabajo: `/home/sadid/proyectos/Phisycs IA` (vacío, proyecto desde cero).

---

## Hardware disponible — evaluación

| Componente | Rol asignado | Prioridad |
|---|---|---|
| **Laptop** i7-12700H, 16 GB RAM, **RTX 3060 6 GB VRAM**, 900 GB libres, WSL2 | **Cerebro**: LLM local, VLM, STT, entrenamiento, simulación | Crítico |
| **Raspberry Pi 5** | **Cuerpo**: control de motores, sensores, reflejos de seguridad, cámara | Crítico (Fase 4) |
| **LAFVIN ESP32-S3 AIChatBot** (codec de audio, altavoz, micrófono, TFT 2.0", 4 botones) | **Terminal fijo de escritorio**: rostro animado + voz + UI de permisos. **Nunca se monta en el robot** (decisión tomada) | Alto (Fase 3) |
| **IdeaSpark ESP32-WROOM-32** (TFT 1.14" ST7789, 2 botones, batería) | Nodo sensor con batería o indicador físico de "motores armados". **No es un HUD** | Bajo — degradado |
| Motores con llantas + servo | Locomoción y pan/tilt de cámara | Alto (Fase 4) |
| Protoboards, LEDs, botones, switch | Prototipado, indicadores de estado, corte de emergencia físico | Alto (seguridad) |
| **Arduino Uno R3 + Sensor Shield v5.0** | Sensores analógicos si la Pi 5 se queda sin GPIO. **No necesario al inicio** | Bajo |

> ⚠️ **El ESP32-S3 no es un cerebro.** Tiene ~512 KB de SRAM (más PSRAM si el módulo la trae). No
> puede correr un LLM, un VLM ni Whisper — ni cuantizado ni recortado. El kit se vende como "AI
> Chatbot" pero funciona **enviando el audio a un servidor**; ese es el modelo de `xiaozhi-esp32`.
> Es un **terminal**: wake word local, codec Opus, I2S, WiFi y pantalla. Eso es mucho, pero es
> periferia. Tratarlo como cerebro lleva a meter lógica de decisión en C sobre un microcontrolador
> en vez de en Python sobre la RTX 3060.

**Recomendación**: no usar el Arduino Uno en las primeras fases. La Pi 5 + ESP32-S3 cubren todo lo
que hace, con WiFi y sin el cuello de botella de 16 MHz. Reservarlo como breakout analógico futuro.

### Consecuencia de fijar la LAFVIN en el escritorio

El robot móvil **no puede llevarse el terminal encima**, así que necesita su propio audio:
micrófono + altavoz USB en la Pi 5 (~$10–15). Esto crea dos rutas de audio, y ese es el riesgo real
de esta decisión: **duplicar el pipeline de voz es exactamente cómo mueren estos proyectos.**

Mitigación obligatoria: una única interfaz `AudioTerminal` en el bus, con dos implementaciones
(`Esp32Terminal` vía WebSocket+Opus, `LocalAudioTerminal` vía ALSA/PortAudio en la Pi). **El agente
no sabe cuál está hablando.** Toda la lógica de voz vive una sola vez, en la laptop.

Ventajas que compra esta decisión: cero trabajo de montaje y alimentación en el kit (la protoboard
se queda en la mesa, sin vibración ni ruido eléctrico de los motores en el codec de audio), y el
agente conserva presencia en el escritorio aunque el robot esté apagado o en otro cuarto.

**Compras futuras** (todas opcionales salvo la primera):
- **Micrófono + altavoz USB para la Pi 5 (~$10–15)** → requerido en Fase 4 si el robot debe hablar
- Sensores HC-SR04 / VL53L0X ToF (~$5–15) → obstáculos, mínimo indispensable Fase 4
- LiDAR RPLIDAR A1 (~$100) → SLAM y navegación real (Fase 5+)
- AI HAT+ Hailo-8L (~$70) → sólo si se decide autonomía total en la Pi

---

## Hallazgos de la investigación

### Modelos de lenguaje (cerebro)

- **Local en 6 GB VRAM**: `Qwen3.6-35B-A3B` (MoE, 35B totales / ~3B activos) corre a ~30 tok/s en 6 GB
  con llama.cpp y tiene tool-calling notablemente mejorado — es la mejor opción local disponible.
  Alternativa densa: cualquier modelo 8B a `Q4_K_M` (~6 GB).
- **API económica**: `claude-haiku-4-5` ($1/$5 por MTok, 200K contexto) como caballo de batalla;
  `claude-sonnet-5` o `claude-opus-5` sólo para planificación de largo horizonte.
- El **tool-calling y la planificación multi-paso** son donde los modelos locales pequeños se rompen.
  Esto justifica el router híbrido elegido.

### VLM (visión)

- `Moondream 2` (1.9B, <4 GB VRAM) — el más rápido; en pruebas de inspección real describe escenas
  y localiza objetos relativos a puntos de referencia. Ideal para el bucle de percepción del robot.
- `Qwen3-VL 8B` / `MiniCPM-V 4.5` — mayor precisión, cabe en 6 GB, más lento. Para consultas puntuales.
- `SmolVLM` (2B) — SOTA para su huella de memoria, opción de respaldo en la Pi.

### VLA (Vision-Language-Action) — Fase 6, aspiracional

- **`SmolVLA` (450M, HuggingFace/LeRobot)** — la única realista para este hardware. Diseñada
  explícitamente para entrenarse en una sola GPU de consumo y desplegarse en GPU de consumo o CPU.
  Se entrena con datasets comunitarios de LeRobot.
- `π0` (~3.3B: 3B VLM + 300M difusión, 50 Hz), `OpenVLA` (7B, Llama2 + LoRA), `GR00T N1` (2B, abierto
  vía Isaac-GR00T, 120 Hz), `Octo` (93M, abierto) — todos integran con LeRobot; sólo Octo y GR00T N1
  son viables sin cuantización agresiva en 6 GB.
- Cerrados / sin pesos: RT-2 (55B), Helix (Figure AI).
- **Ninguna VLA es necesaria hasta la Fase 6.** Un robot móvil con navegación reactiva no las requiere.

### Gemini Robotics — separar tres productos que se confunden

| Modelo | Qué hace | Acceso real (jul 2026) |
|---|---|---|
| **`gemini-robotics-er-1.6-preview`** | **VLM de razonamiento encarnado.** Devuelve texto, **puntos y bounding boxes en coordenadas**, trayectorias, JSON estructurado; soporta function calling y ejecución de código. 1M de contexto | **Público vía Gemini API. $0.30 / $2.50 por MTok** |
| Gemini Robotics 1.5 (la VLA) | Emite acciones de motor | Acceso restringido, sin pesos públicos |
| Gemini Robotics On-Device | VLA local en el robot | Trusted testers |

**El único usable hoy es ER**, y es el que conviene: no es un cerebro, es un **proveedor de visión
espacial**. Para la consulta que de verdad necesita un robot ("¿dónde está la puerta? dame el punto
en píxeles") supera a un VLM genérico, y a $0.30/MTok de entrada cuesta menos que Haiku.

**Decisión**: se adopta en Fase 2 como una implementación más de `VisionProvider`, junto a Moondream
(rápido/local) y Qwen3-VL (preciso/local). Encaja exactamente en el principio 80/20: **ER devuelve
coordenadas; el código determinista convierte coordenadas en movimiento.** Nunca genera la acción.
Advertencia: es *preview* y la versión 1.5 ya está marcada para deprecación — por eso va detrás de
la interfaz, sin dependencia dura, y el sistema debe degradar a Moondream si no está disponible.

### World models — observar, no construir

`V-JEPA 2` (Meta) permite planificación robótica zero-shot en entornos nuevos; `Cosmos Predict 2.5`
(NVIDIA) genera datos sintéticos y evalúa políticas en simulación. Ambos son de escala de
investigación. **Decisión: no incluirlos en el plan ejecutable.** Se registran como vía futura
para la Fase 6 si se consigue más VRAM.

### Arquitectura de agentes — el *harness*

El patrón validado en 2026 (Claude Agent SDK, Agno) es tratar el harness como un **sistema operativo
del agente**, no como un prompt:

- **Curación de contexto** — decidir qué información entra, no meter todo
- **Secuencia de arranque** — system prompt, hooks, skills disponibles
- **Drivers estándar** — las descripciones de herramientas *son* parte de la arquitectura
- **Chequeo de permisos por cada llamada a herramienta** — `deny → ask → allow`, en ese orden
- Subagentes con contexto propio; sesiones persistentes; soporte MCP de primera clase

Este es el patrón que adopta este proyecto.

### Frameworks multi-agente

- **`GradientHQ/symphony`** — framework multi-agente descentralizado que orquesta modelos ligeros en
  hardware heterogéneo de consumo (RTX, Jetson, edge) con enrutamiento por *beacon* y votación
  Chain-of-Thought. **Es el "Symphony" relevante para este proyecto** (laptop + Pi + ESP32).
- `openai/symphony` (marzo 2026) es Elixir/BEAM para orquestar agentes de *código* — no aplica.
- `OS-Symphony` (ACL 2026) — agentes de uso de computadora, no encarnados.

**Decisión**: no adoptar ningún framework multi-agente en Fase 1. Se estudia `GradientHQ/symphony`
como referencia de diseño para el enrutamiento entre laptop/Pi/ESP32 en Fase 4.

### MCP en robótica

El ecosistema MCP robótico supera los 50 servidores en 2026 (puentes ROS/ROS2, brazos, domótica).
Referencias directas: `LCAS/ros2_mcp` (introspección y control ROS2 vía MCP), `ROSBag MCP Server`
(análisis de datos de robot con LLM), y el *Robot Context Protocol* (RCP) como propuesta agnóstica
de runtime. El patrón clave: **el agente ve por las cámaras vía MCP resources y actúa vía MCP tools**,
cerrando el bucle percepción-acción.

### Proyectos GitHub de referencia (tipo Jarvis)

| Proyecto | Qué aporta |
|---|---|
| `PanPenek/JarvisAi` | Offline total: Whisper STT + Kokoro TTS + Ollama. 31 herramientas, encadenamiento agéntico hasta 15 llamadas. **Mejor referencia de pipeline de voz local.** |
| `vierisid/jarvis` | Daemon autónomo *always-on* con jerarquía multi-agente y **límites de autoridad definidos** — el modelo de permisos a copiar. |
| `bionorthtech/JarvisAI` | ChromaDB para memoria, "segundo cerebro", bots autónomos con *emotional drives*. |
| `78/xiaozhi-esp32` | **Firmware ESP32 basado en MCP.** WebSocket con JSON de control + frames Opus binarios, wake word local, despacho MCP. Soporta 15+ placas incluyendo ESP32-S3. **Es la base directa para el terminal de voz LAFVIN.** |
| `keon/awesome-physical-ai` | Índice curado de VLA, world models y robot foundation models. |
| `huggingface/lerobot` | Framework de aprendizaje robótico end-to-end + brazo SO-101 imprimible (~$130). Vía de expansión Fase 6. |

### Voz (Fase 3)

- **STT**: `faster-whisper` — el mejor para pipelines Python con GPU NVIDIA. En una RTX 3060 la
  latencia extremo a extremo (STT+LLM+TTS) es de **1–2 s**; en una Pi 5 sola es de **5–8 s**.
  Esto refuerza poner el cerebro en la laptop.
- **TTS**: `Kokoro-82M` (82M parámetros, 2–3 GB VRAM, corre incluso en CPU, calidad muy superior a
  su tamaño). **`Piper` fue archivado en octubre de 2025** — no usarlo en proyecto nuevo.
  `XTTS v2` sólo si se requiere clonación de voz.

### Memoria del agente

El ecosistema convergió en una taxonomía de tres niveles que replica la ciencia cognitiva:
**episódica** (interacciones pasadas), **semántica** (hechos y preferencias), **procedimental**
(comportamientos y reglas aprendidas). Ninguna arquitectura pura basta: los sistemas de producción
usan **híbridos vector + grafo**.

- `Mem0` (~48k estrellas) — capa semántica más desplegada; scopes user/session/agent.
- `Letta`/`MemGPT` — paginación de contexto estilo SO, para agentes de larga duración.
- `Zep`/`Graphiti` — razonamiento sobre cómo cambian los hechos en el tiempo.

**Decisión**: implementar los tres niveles con SQLite + `sqlite-vec` propio en Fase 1 (control total,
sin dependencia de servicio), con interfaz que permita cambiar a Mem0 después.

### Memoria por identidad — embeddings de rostro, objeto y lugar

Patrón validado y barato: `InsightFace`/`ArcFace` produce vectores de **512 dimensiones**, corre
sobrado en la RTX 3060 y **no requiere LLM**. Los stacks de referencia combinan InsightFace para el
embedding con búsqueda vectorial (FAISS / pgvector / Qdrant).

Aplicado a este proyecto: la identidad se convierte en la **clave de scope de la memoria episódica**.
El robot ve una cara → obtiene el embedding → busca el vecino más cercano → si coincide, recupera
todo el historial de interacciones con esa persona; si no, crea una identidad nueva. **El mismo
mecanismo generaliza a objetos y lugares** sin código adicional: es la misma tabla vectorial con
distinto tipo de entidad.

**Advertencia de privacidad, no negociable**: un embedding facial **no es un dato anónimo, es el
rostro**. Trabajo de 2026 demuestra reconstrucción facial realista a partir del embedding mediante
modelos de difusión. Es un identificador biométrico persistente y se trata como tal (ver sección de
ciberseguridad).

### Ciberseguridad — OWASP Top 10 for Agentic Applications 2026

Publicado en diciembre de 2025, es el marco de referencia. Riesgos que aplican directamente:

- **Prompt injection indirecta** — manipulación vía documentos o datos externos que el agente lee
- **Agent Control/Takeover** — el atacante secuestra el proceso de decisión
- **Supply Chain / Skill Poisoning** — se documentaron skills maliciosas alcanzando 26.000+ agentes
  vía marketplaces confiables, y dependencias secuestrables afectando 134K agentes
- **Principio rector: "Least Agency"** — *la autonomía es una capacidad que se gana, no un default*

Este principio se aplica literalmente: el agente arranca con permisos mínimos y los gana por fase.

---

## Decisiones de arquitectura (confirmadas)

| Decisión | Elección | Razón |
|---|---|---|
| **Cerebro** | Router híbrido | Modelo local para reflejos y tareas rutinarias; API Claude sólo para planificación difícil. Se puede forzar 100% local con una variable de entorno. |
| **Middleware** | Bus propio ligero (ZeroMQ + Pydantic) | ROS 2 en WSL2 tiene problemas de multicast DDS y una curva de aprendizaje que ha hundido proyectos anteriores. Se deja un puerto de salida a ROS 2 si Fase 5 necesita Nav2/SLAM. |
| **Ubicación del cómputo** | **A decidir tras Fase 2** | Se construye con la capa de transporte abstracta y se mide latencia real antes de comprometerse. La Fase 2 termina con un benchmark que produce esta decisión con datos. |
| **Lenguaje** | Python 3.12 + `uv` | Ya instalado; ecosistema de IA y robótica. |
| **Firmware ESP32** | ESP-IDF, base `xiaozhi-esp32` | Protocolo WebSocket + Opus ya resuelto y probado en ESP32-S3. |

---

## Principio rector: el reparto 80/20 entre código e IA

El principio más importante del proyecto, y el que más determina si funciona o no:

> **El LLM nunca cierra un lazo de control.**

El LLM **decide y parametriza**; el código determinista **ejecuta y verifica**. Concretamente, el
agente no genera trayectorias, comandos PWM ni secuencias de motor: **selecciona una skill de una
biblioteca determinista y le pasa parámetros validados**.

```
LLM (el 20%)                          Código tradicional (el 80%)
─────────────                         ───────────────────────────
"acércate a la puerta"      ──────▶   goto(x=1.2, y=0.4)   ← PID, rampas, encoders
"mira a la izquierda"       ──────▶   look_at(pan=-45)     ← límites mecánicos, suavizado
"¿qué hay en la mesa?"      ──────▶   VisionProvider       ← coordenadas, no acciones
                                      ▲
                            REFLEJOS ─┘ (0% LLM: parada, obstáculo, timeout, corriente)
```

**La proporción no es fija: varía con la distancia al actuador.**

| Capa | Código / IA | Por qué |
|---|---|---|
| Reflejos (parada, obstáculo, timeout) | **100 / 0** | Tiene deadline duro y consecuencia física |
| Skills de movimiento (`goto`, `turn`, `look_at`) | **95 / 5** | La IA sólo elige la skill y sus parámetros |
| Navegación y percepción | **80 / 20** | La IA interpreta la escena; el código planifica la ruta |
| Planificación y diálogo | **40 / 60** | Aquí la IA aporta valor real |
| Metas y reflexión | **20 / 80** | Territorio propio del LLM |

**Beneficios concretos, no filosóficos:**

1. **Testeable sin LLM** — cada skill tiene su prueba unitaria determinista. Es la única forma de
   depurar un robot sin quemar tokens en cada iteración.
2. **Barato** — el 80% del cómputo no llama a ningún modelo.
3. **Seguro** — una alucinación del LLM produce como máximo *una skill mal elegida*, nunca un
   comando de motor arbitrario. El espacio de acciones está acotado por construcción.
4. **Degradable** — si el cerebro cae, las skills y los reflejos siguen funcionando.

**Componente que esto añade a la arquitectura**: `src/argos/skills/` — biblioteca de skills
tipadas con Pydantic, cada una con precondiciones, validación de parámetros, criterio de éxito y
prueba unitaria propia. El `ToolRegistry` del harness expone las skills al LLM; no expone motores.

---

## Arquitectura del sistema

```
┌─────────────────────────── LAPTOP (cerebro) ──────────────────────────┐
│                                                                       │
│  ┌─────────────────────────── HARNESS ────────────────────────────┐  │
│  │  Boot: system prompt + hooks + skills disponibles              │  │
│  │  Context curator ──▶ decide qué entra al contexto              │  │
│  │  Tool registry ────▶ descripciones = parte del prompt          │  │
│  │  Permission gate ──▶ deny → ask → allow  (Least Agency)        │  │
│  │  Agent loop ───────▶ percibir → pensar → actuar → reflexionar  │  │
│  └───────────────┬──────────────────┬──────────────────┬──────────┘  │
│                  │                  │                  │             │
│         ┌────────▼──────┐   ┌───────▼──────┐   ┌───────▼────────┐    │
│         │ MODEL ROUTER  │   │   MEMORIA    │   │ SKILLS (80%)   │    │
│         │ local ⇄ API   │   │ episódica    │   │ deterministas, │    │
│         │ por costo y   │   │ semántica    │   │ tipadas,       │    │
│         │ dificultad    │   │ procedimental│   │ testeables     │    │
│         └───────────────┘   │ identidades  │   ├────────────────┤    │
│                             │ (embeddings) │   │ TOOLS / MCP    │    │
│                             └──────────────┘   └────────────────┘    │
│                                                                       │
│   Percepción: faster-whisper (STT) · InsightFace (identidad)          │
│               Moondream / Qwen3-VL / Gemini-ER (visión espacial)      │
│   Salida:     Kokoro-82M (TTS)                                        │
└───────────────────────────────┬───────────────────────────────────────┘
                                │  BUS (ZeroMQ, mensajes Pydantic)
                                │  transporte abstracto: inproc | tcp
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐   ┌──────────▼─────────┐   ┌─────────▼──────────┐
│ ESP32-S3 LAFVIN│   │  RASPBERRY PI 5    │   │ ESP32 IdeaSpark    │
│ TERMINAL FIJO  │   │  Cuerpo (móvil)    │   │ (degradado)        │
│ rostro animado │   │  motores, servo,   │   │ nodo sensor o      │
│ mic+codec+spk  │   │  cámara, sensores  │   │ testigo físico de  │
│ WS + Opus      │   │  mic+altavoz USB   │   │ "motores armados"  │
│ 4 botones      │   │  REFLEJOS locales  │   │                    │
└───────┬────────┘   └──────────┬─────────┘   └────────────────────┘
        │                       │
        └──── AudioTerminal ────┘   una sola interfaz, dos implementaciones.
              (Esp32 | Local)       El agente no sabe cuál está hablando.
```

**Regla de seguridad transversal**: los *reflejos* (parada de emergencia, evasión de obstáculo,
límite de corriente) viven **siempre en el nodo más cercano al actuador**, nunca en el LLM. El LLM
decide *intenciones*; el nodo local decide si esa intención es físicamente segura de ejecutar.

---

## Plan por fases

### Fase 0 — Fundación (2–3 días)

Objetivo: que el repositorio sea imposible de abandonar por fricción.

- `git init`, estructura `src/argos/{harness,models,memory,tools,bus,perception,body}/`
- `uv` + `pyproject.toml`, `ruff`, `pytest`, `pre-commit`
- Configuración por capas: `settings.toml` + `.env` (secretos) + override por variable de entorno.
  `.env` en `.gitignore` desde el primer commit.
- **Logging estructurado + trazas del agente desde el día 1** (JSONL: cada turno, cada llamada a
  herramienta, cada decisión del router, tokens y costo). Sin esto no se puede depurar un agente.
- `CLAUDE.md` con las convenciones del repo.

**Entregable**: repo que arranca, testea y formatea con un comando.

---

### Fase 1 — El cerebro (2–3 semanas) ⭐ el corazón del proyecto

Objetivo: un agente de escritorio completamente funcional, autónomo en decisiones, sin hardware.

**1.1 — Harness**
- `Harness`: boot sequence, system prompt versionado en archivo (no hardcodeado), registro de hooks
- `AgentLoop`: bucle percibir → pensar → actuar → reflexionar, con límite de iteraciones y
  presupuesto de tokens por tarea
- `ContextCurator`: ventana deslizante + compactación por resumen cuando se acerca al límite
- `PermissionGate`: evalúa `deny → ask → allow` **antes** de cada ejecución de herramienta.
  Política declarativa en YAML por herramienta. Default: `ask`.

**1.2 — Router de modelos**
- Interfaz `LLMProvider` única. Implementaciones: `LocalProvider` (llama.cpp server / Ollama),
  `ClaudeProvider` (SDK `anthropic`, `claude-haiku-4-5` por defecto, `claude-sonnet-5` para planificación)
- `ModelRouter`: clasifica la tarea (reflejo / rutina / razonamiento / planificación) y enruta.
  Registra costo real por decisión. Variable `ARGOS_LOCAL_ONLY=1` fuerza 100% local.
- Descarga y evaluación de `Qwen3.6-35B-A3B` GGUF (Q2/Q4) vs un denso 8B `Q4_K_M` en la RTX 3060.
  Criterio de selección: **tasa de éxito en tool-calling**, no perplexity.

**1.3 — Memoria de tres niveles + identidades**
- SQLite + `sqlite-vec` para embeddings locales
- **Episódica**: log de interacciones con recuperación por similitud + recencia
- **Semántica**: hechos y preferencias extraídos, con deduplicación
- **Procedimental**: reglas aprendidas ("cuando X, hacer Y") que se inyectan en el system prompt
- **Tabla de entidades desde el día 1**: `entity(id, tipo, embedding, etiqueta, creado, visto_por_última_vez)`
  con `tipo ∈ {persona, objeto, lugar}`. Toda memoria episódica lleva `entity_id` opcional.
  En Fase 1 sólo se puebla con entidades de texto; en Fase 2 llega el embedding facial y **no hay
  que migrar el esquema**. Esta es la razón de definirla ahora y no después.
- Interfaz `MemoryStore` desacoplada para poder migrar a Mem0/Zep sin tocar el harness

**1.4 — Biblioteca de skills (el 80%)**
- `src/argos/skills/`: cada skill es una clase con esquema Pydantic de parámetros, precondiciones,
  ejecución determinista, criterio de éxito verificable y prueba unitaria propia
- Skills de Fase 1 (sin hardware): `search_web`, `read_file`, `write_note`, `set_reminder`,
  `summarize`, `run_command` — todas con validación de parámetros antes de ejecutar
- El `ToolRegistry` expone **skills**, no primitivas. Establece el patrón que la Fase 4 va a usar
  para los motores: el LLM nunca verá una API de bajo nivel.

**1.5 — Herramientas y MCP**
- Herramientas nativas: sistema de archivos (con raíz confinada), ejecución de shell (allowlist),
  búsqueda web, temporizadores, notas
- Cliente MCP: el agente puede conectarse a servidores MCP externos. Cada servidor MCP nuevo entra
  con permisos `ask` por defecto.
- **Diseño de herramientas**: descripciones prescriptivas de *cuándo* llamar, no sólo qué hacen —
  es lo que más eleva la tasa de invocación correcta.

**1.6 — Autonomía**
- `GoalManager`: metas persistentes con estado, prioridad y criterio de completitud
- `Scheduler`: el agente despierta solo (cron/intervalos), revisa metas y decide si actuar
- `Reflector`: al cerrar una tarea, evalúa qué salió bien/mal y escribe memoria procedimental

**Entregable**: agente de escritorio que corre como daemon, tiene metas propias, decide cuándo
actuar, recuerda entre sesiones y pide permiso antes de acciones destructivas. **Usable aunque el
proyecto termine aquí.**

---

### Fase 2 — Sentidos digitales + identidad + decisión de cómputo (2 semanas)

Objetivo: el agente ve, reconoce a quién ve, y se toma la decisión de arquitectura pendiente **con datos**.

**2.1 — Visión con tres proveedores tras una interfaz**
- `VisionProvider` con tres implementaciones y selección automática por tipo de consulta:
  - `MoondreamProvider` (1.9B local) — bucle de percepción continuo, barato y rápido
  - `QwenVLProvider` (8B local) — consultas puntuales que exigen precisión
  - `GeminiERProvider` (`gemini-robotics-er-1.6-preview`) — **consultas espaciales**: puntos,
    bounding boxes, trayectorias. Es el que se usará para "¿dónde está X?" en Fase 5.
- **Degradación obligatoria**: si Gemini-ER no responde o el presupuesto se agota, cae a Moondream.
  ER es *preview*; el sistema no puede depender de él.
- Skills: `see_image(path)`, `see_screen()`, `describe_scene()`, `locate_object(desc) → (x, y)`.
  Nótese que `locate_object` devuelve **coordenadas**, no una acción — es el 80/20 aplicado.

**2.2 — Reconocimiento de identidad**
- `InsightFace` (ArcFace, 512-d) sobre la webcam: detectar rostro → embedding → buscar en la tabla
  `entity` creada en Fase 1 → si el vecino más cercano supera el umbral, es la misma persona.
- Al reconocer, el harness inyecta en contexto el historial episódico de esa persona.
- Enrolamiento **explícito y con consentimiento**: el agente pregunta "¿cómo te llamas?" y sólo
  entonces persiste la identidad. Nunca enrola en silencio.
- Skills: `who_is_this()`, `remember_person(nombre)`, `forget_person(nombre)` — la última borra
  embedding e historial asociado, y debe existir desde el primer día.
- El mismo pipeline se reutiliza para objetos y lugares en Fase 5 sin código nuevo.

**2.3 — Benchmark de decisión** (el entregable clave de esta fase)

Mide en hardware real la latencia y calidad de (a) visión en laptop, (b) visión en Pi 5,
(c) round-trip laptop↔Pi por WiFi, (d) pipeline de voz completo en cada sitio, (e) latencia y costo
real de Gemini-ER frente a Moondream en consultas de localización.
El resultado **decide** dónde vive el cerebro cuando el robot esté móvil, y queda escrito en
`docs/adr/001-ubicacion-computo.md`.

**Entregable**: agente que ve, reconoce personas y recuerda su historial + ADR con la decisión de
arquitectura tomada sobre mediciones.

---

### Fase 3 — Voz, rostro y presencia física (2.5–3 semanas)

Objetivo: el agente escucha, habla y tiene cara. Primer hardware real.

> **Nota de riesgo, decidida por el usuario.** Propuse partir esta fase en 3a (voz en laptop) y 3b
> (firmware ESP32) para tener un entregable intermedio; se decidió mantenerla como una sola fase.
> Se acepta la decisión y se mitiga con **un orden de trabajo interno estricto y un checkpoint a
> mitad**: nada de ESP-IDF hasta que la voz funcione en la laptop y esté medida.

**Orden de trabajo obligatorio** (no reordenar):

**3.1 — Pipeline de voz en la laptop** *(2–3 días)*
- `faster-whisper` en GPU → agente → `Kokoro-82M`, usando micrófono y altavoces existentes
- Interfaz `AudioTerminal` definida aquí, con `LocalAudioTerminal` como primera implementación
- **Checkpoint duro**: latencia extremo a extremo medida (<2 s objetivo). Si no se cumple, se
  arregla aquí — **no se avanza a ESP-IDF con un pipeline lento**, porque el ESP32 sólo puede
  añadir latencia, nunca quitarla.

**3.2 — Bus de mensajes** *(2 días)*
- ZeroMQ + esquemas Pydantic, transporte configurable (`inproc` en desarrollo, `tcp` distribuido)
- Autenticación con clave precompartida desde el principio

**3.3 — Verificación de hardware del kit** *(medio día, antes de escribir firmware)*
- Flashear un sketch mínimo y **medir**: ¿el módulo tiene PSRAM? (`esp_get_free_heap_size()` /
  `heap_caps_get_total_size(MALLOC_CAP_SPIRAM)`), resolución y controlador reales de la TFT,
  modelo del codec de audio, pines de SPI e I2S.
- **Este paso decide si el rostro es cómodo o angustioso.** Si no hay PSRAM, se recorta el alcance
  del rostro a 3 estados con sprites mínimos, y se documenta en la ADR.

**3.4 — Firmware del terminal ESP32-S3** *(1–1.5 semanas)*
- Base: `78/xiaozhi-esp32` (WebSocket con JSON de control + frames Opus binarios, wake word local)
- `Esp32Terminal` como segunda implementación de `AudioTerminal` — **cero lógica de voz nueva**
- Los 4 botones: push-to-talk, interrumpir, confirmar/denegar permiso, mute
- **Aprobación por voz**: el `PermissionGate` de Fase 1 ahora pregunta en voz alta y acepta la
  respuesta por botón físico (el botón es la vía confiable; la voz puede fallar en reconocimiento)

**3.5 — Rostro animado** *(3–4 días)*

Restricciones de diseño, no negociables:

| Restricción | Razón (con números) |
|---|---|
| **Sprites de región parcial, nunca frame completo** | 240×320 RGB565 = **153 KB/frame**; a 40 MHz SPI son ~31 ms → ~32 fps teóricos *antes* de que el I2S robe ciclos. Ojos+boca como sprites = **~7 KB/frame**, 20× menos, sobran 60 fps |
| **El audio siempre gana** | El I2S tiene deadline duro. Un frame de audio perdido **se oye**; uno de cara no lo nota nadie. La animación corre a la prioridad más baja y es interrumpible |
| **Nunca enviar frames por WiFi** | Competirían con el stream Opus en la misma radio. La laptop manda **estado** (`escuchando`), el ESP32 **renderiza local** |
| **Alcance duro v1: 6 estados** | reposo, escuchando, pensando, hablando, error, dormido. **Sin lip-sync, sin emociones, sin animaciones de reposo elaboradas.** El rostro es una trampa de alcance adictiva: es fácil gastar tres semanas ahí |

Lip-sync desde la envolvente de amplitud del Opus queda registrado como v2 — barato, pero v2.

**Fuera de alcance de esta fase (degradado a propósito)**: el HUD del IdeaSpark. Un dashboard web
en la laptop da más información con una fracción del esfuerzo. El IdeaSpark se reserva para nodo
sensor con batería o testigo físico de "motores armados" en Fase 4.

**Entregable**: asistente de voz físico con rostro, capaz de funcionar offline.
**Usable aunque el proyecto termine aquí.**

---

### Fase 4 — El cuerpo (2–3 semanas)

Objetivo: el agente se mueve y decide sus movimientos.

- **Nodo Pi 5**: proceso Python que se suscribe al bus, expone control de motores y servo,
  publica telemetría (encoders, batería, sensores).
- **Capa de reflejos (independiente del LLM)**: bucle de control en la Pi con parada por obstáculo,
  timeout de comando (si el enlace cae, se detiene), límites de velocidad. **Corre sin el cerebro.**
- **Corte físico**: el switch de encendido cablead0 en serie con la alimentación de motores.
  El LED indica "motores armados". No es opcional.
- **Audio propio del robot**: micrófono + altavoz USB en la Pi 5, expuestos como una segunda
  instancia de `LocalAudioTerminal`. Cero lógica de voz nueva — es la interfaz de Fase 3.
- Sensores mínimos: ultrasónico HC-SR04 o ToF VL53L0X frontal + encoders de rueda
- Herramientas del agente: `move(distance)`, `turn(angle)`, `look_at(pan, tilt)`, `stop()`,
  `read_sensors()` — todas pasando por el `PermissionGate` y validadas por la capa de reflejos
- Aplicación de la decisión de la ADR de Fase 2 sobre dónde corre qué

**Entregable**: robot móvil que ejecuta intenciones del agente con seguridad física garantizada.

---

### Fase 5 — Percepción encarnada y navegación (2–3 semanas)

Objetivo: el robot entiende dónde está y qué hay alrededor.

- Cámara en la Pi 5 con streaming al servicio de visión (según ADR)
- Bucle de percepción continuo: VLM describiendo la escena a intervalo adaptativo
- Mapa de ocupación simple construido desde odometría + sensores de distancia
- Navegación reactiva: llegar a un objetivo descrito en lenguaje natural ("ve a la puerta").
  Pipeline 80/20 completo: `GeminiERProvider.locate_object("puerta") → (x, y)` → conversión
  determinista a pose → `goto()` con PID y reflejos. **La IA nunca toca el motor.**
- Memoria espacial: reutilizar la tabla `entity` para lugares y objetos con su última posición conocida

**Motor de curiosidad (con presupuesto)**

Se implementa la versión barata y correcta: **novedad por distancia de embedding**. Cada percepción
se compara con el vecino más cercano en la tabla `entity`; si la distancia supera un umbral, es algo
nuevo y genera un candidato a investigar. No requiere world model.

La versión cara (curiosidad por error de predicción) sí lo requiere y queda fuera de alcance.

**Restricciones, que son la parte importante:** curiosidad ilimitada + robot físico + tokens de API
es un robot que se distrae, gasta y se mete en problemas. Por eso la curiosidad es un *drive con
cuota*, no un permiso:
- Sólo se activa cuando **no hay meta activa** — nunca interrumpe una tarea
- Cuota configurable (p. ej. N investigaciones por hora) con presupuesto de tokens propio
- Pasa por el **mismo `PermissionGate`** que todo lo demás: investigar puede requerir moverse, y
  moverse sigue necesitando aprobación bajo la política vigente
- Toda investigación escribe en memoria episódica, se reconozca o no el objeto — si no, la
  curiosidad se repite en bucle sobre lo mismo

**Punto de decisión**: si se necesita SLAM real → comprar RPLIDAR A1 y añadir el puente ROS 2
(`LCAS/ros2_mcp` como referencia de integración MCP↔ROS2). Si la navegación reactiva basta,
no se añade ROS 2.

**Entregable**: robot que navega a objetivos en lenguaje natural y explora lo desconocido con
presupuesto acotado.

---

### Fase 6 — Autonomía real (abierta)

Objetivo: el robot persigue metas propias en el tiempo. Sin fecha fija; se aborda cuando 0–5 estén sólidas.

- Metas persistentes encarnadas ("patrulla la casa cada mañana y reporta cambios")
- Memoria espacial: el robot recuerda dónde estaban las cosas y nota diferencias
- **Vía de expansión A — manipulación**: brazo SO-101 imprimible (~$130) + `SmolVLA` fine-tuneada
  con LeRobot en la RTX 3060. Es la única VLA realista para este hardware.
- **Vía de expansión B — world models**: `V-JEPA 2` para planificación zero-shot. Requiere más VRAM;
  evaluar sólo si se actualiza el hardware.
- Estudio de `GradientHQ/symphony` para orquestación multi-agente entre laptop/Pi/ESP32

---

## Ciberseguridad — aplicada, no aspiracional

Alineado con **OWASP Top 10 for Agentic Applications 2026**. Cada control se implementa en la fase indicada.

| Riesgo OWASP | Control | Fase |
|---|---|---|
| Prompt injection indirecta | Todo contenido externo (web, archivos, resultados MCP) se marca como *no confiable* en el contexto y nunca se trata como instrucción | 1 |
| Agent Control / Takeover | `PermissionGate` con default `ask`; acciones destructivas y físicas siempre requieren confirmación | 1, 4 |
| Least Agency | Permisos se ganan por fase. La Fase 1 no puede tocar hardware. La Fase 4 arranca con `move` en `ask`. | Todas |
| Skill/Supply chain poisoning | Dependencias fijadas con hash (`uv.lock`); servidores MCP externos entran en `ask` y se auditan antes de promover a `allow` | 0, 1 |
| Exfiltración de datos | Herramienta de archivos con raíz confinada y validación de path canónico (rechaza `..`, symlinks, rutas absolutas fuera de raíz) | 1 |
| Ejecución de código | Shell con **allowlist** de ejecutables, rechazo de operadores (`&&`, `\|`, `;`, backticks, `$()`), timeouts y límites de recursos. Nunca blocklist. | 1 |
| Secretos | Nunca en prompts, memoria ni logs. `.env` fuera de git. Redacción automática en trazas. | 0 |
| Espacio de acciones acotado | El LLM sólo puede invocar skills tipadas con parámetros validados; nunca primitivas de motor. Una alucinación produce como máximo una skill mal elegida | 1, 4 |
| **Datos biométricos** | Los embeddings faciales **no son anónimos** (hay reconstrucción facial demostrada desde el embedding vía difusión). Por tanto: cifrados en reposo, **nunca salen del dispositivo**, **nunca entran en un prompt a una API externa**, enrolamiento sólo con consentimiento explícito, y `forget_person()` disponible desde el primer día | 2 |
| Curiosidad acotada | Cuota por hora, presupuesto de tokens propio, sólo sin meta activa, sujeta al `PermissionGate` | 5 |
| Seguridad física | Reflejos independientes del LLM + corte físico de motores + timeout de enlace | 4 |
| Superficie de red | ESP32 y Pi en VLAN/red aislada; bus autenticado con clave precompartida; sin puertos expuestos a internet | 3, 4 |
| Auditabilidad | Traza JSONL completa e inmutable de cada decisión, herramienta y costo | 0 |

---

## Buenas prácticas de implementación de IA

- **Evals antes que features**: cada fase define su suite de evaluación *antes* de implementar.
  Para el agente: tasa de éxito en tool-calling, no perplexity.
- **Descripciones de herramientas prescriptivas**: decir *cuándo* invocar, no sólo qué hace.
- **Prompt caching** en el proveedor Claude: prompt de sistema congelado (nada de timestamps ni IDs
  interpolados) y contenido volátil al final. Reduce el costo ~90% en el prefijo.
- **Presupuesto por tarea**: límite de tokens e iteraciones por objetivo; el agente lo sabe y se
  autolimita en lugar de ser cortado.
- **Contexto curado, no acumulado**: compactación por resumen y limpieza de resultados de
  herramientas obsoletos.
- **Determinismo donde importa**: el enrutamiento, los permisos y los reflejos son código, no prompts.
- **Un artefacto usable por fase** — la regla que define este plan.

---

## Archivos críticos a crear

| Ruta | Contenido |
|---|---|
| `src/argos/harness/loop.py` | Bucle del agente, presupuestos, compactación |
| `src/argos/harness/permissions.py` | `PermissionGate` (deny→ask→allow) |
| `src/argos/harness/context.py` | `ContextCurator` |
| `src/argos/models/router.py` | `ModelRouter` y clasificación de tareas |
| `src/argos/models/{local,claude}.py` | Implementaciones de `LLMProvider` |
| `src/argos/memory/store.py` | Memoria episódica/semántica/procedimental + tabla `entity` sobre SQLite |
| **`src/argos/skills/`** | **Biblioteca de skills deterministas (el 80%)** — tipadas, con precondiciones y prueba unitaria |
| `src/argos/tools/registry.py` | Registro de skills expuestas al LLM + cliente MCP |
| `src/argos/bus/` | Bus ZeroMQ + esquemas Pydantic, transporte abstracto |
| `src/argos/perception/vision/{moondream,qwen_vl,gemini_er}.py` | Implementaciones de `VisionProvider` con degradación |
| `src/argos/perception/identity.py` | InsightFace + búsqueda vectorial + consentimiento y borrado |
| `src/argos/perception/{stt,tts}.py` | faster-whisper, Kokoro-82M |
| `src/argos/audio/terminal.py` | Interfaz `AudioTerminal` + `LocalAudioTerminal` / `Esp32Terminal` |
| `src/argos/drives/curiosity.py` | Novedad por embedding, cuota y presupuesto |
| `src/argos/body/` | Nodo Pi 5, reflejos, drivers de motor (nunca expuestos al LLM) |
| `firmware/desk-terminal/` | ESP-IDF para LAFVIN ESP32-S3: audio (base xiaozhi) + rostro por sprites |
| `firmware/desk-terminal/face/` | 6 estados como sprites, render de región parcial, prioridad baja |
| `config/prompts/system.md` | System prompt versionado |
| `config/permissions.yaml` | Política declarativa por herramienta |
| `docs/adr/` | Decisiones de arquitectura, empezando por la de Fase 2 |

---

## Verificación

**Fase 0**: `uv run pytest` pasa, `ruff check` limpio, `.env` no aparece en `git status`.

**Fase 1**:
- Suite de evals de tool-calling: ≥20 tareas con resultado verificable, medida en modelo local y en
  Claude; el router debe elegir correctamente en ≥90% de los casos.
- Prueba de memoria: reiniciar el daemon y verificar que recuerda hechos y preferencias de la sesión previa.
- Prueba de permisos: intentar `rm -rf` fuera de la raíz confinada → debe ser bloqueado, no preguntado.
- Prueba de inyección: alimentar un documento con instrucciones maliciosas → el agente no las ejecuta.
- **Cobertura de skills sin LLM**: cada skill debe pasar su prueba unitaria con el modelo apagado.
  Si una skill sólo se puede probar invocando al agente, está mal diseñada.
- Correr el daemon 24 h y revisar la traza JSONL: costo, decisiones del router, metas atendidas.

**Fase 2**:
- Benchmark ejecutable que produce la tabla de latencias y escribe la ADR.
- Identidad: enrolar 3 personas, verificar reconocimiento tras reiniciar; verificar que
  `forget_person()` borra embedding **e** historial; confirmar en la traza que **ningún embedding
  facial aparece jamás en una llamada a API externa**.
- Degradación: cortar el acceso a Gemini-ER y verificar que la visión sigue funcionando con Moondream.

**Fase 3**:
- **Checkpoint 3.1 (bloqueante)**: latencia de voz extremo a extremo en laptop <2 s, medida y
  registrada. **No se escribe una línea de ESP-IDF hasta pasar esto.**
- Verificación de hardware 3.3 documentada: PSRAM sí/no, resolución y controlador reales de la TFT.
- Probar denegación de permiso por botón físico y por voz.
- Desconectar internet y verificar que el modo local funciona completo.
- **Prueba de contención audio/vídeo**: con el rostro animando a máxima actividad, grabar 60 s de
  audio y verificar **cero frames Opus perdidos**. Si se pierden, la animación baja de prioridad.
- Confirmar que la interfaz `AudioTerminal` no tiene lógica de voz duplicada: la misma prueba de
  conversación debe pasar contra `LocalAudioTerminal` y `Esp32Terminal` sin cambiar el test.

**Fase 4**: **con el robot elevado sobre bloques y las ruedas al aire**, verificar cada comando.
Después: prueba de corte de enlace (WiFi apagado → el robot debe detenerse en <500 ms), prueba de
obstáculo (bloquear el sensor → parada de reflejo sin intervención del LLM), prueba del corte físico.
Sólo entonces bajar el robot al suelo.

**Fase 5**: recorrido a objetivo en lenguaje natural, con tasa de éxito sobre 10 intentos.

---

## Cronograma realista

| Fase | Duración | Acumulado |
|---|---|---|
| 0 — Fundación | 2–3 días | ~3 días |
| 1 — Cerebro + skills | 2–3 semanas | ~3.5 semanas |
| 2 — Visión, identidad + ADR | 2 semanas | ~5.5 semanas |
| 3 — Voz, rostro y presencia | 2.5–3 semanas | ~8.5 semanas |
| 4 — Cuerpo | 2–3 semanas | ~11.5 semanas |
| 5 — Navegación + curiosidad | 2–3 semanas | ~14.5 semanas |
| 6 — Autonomía real | abierta | — |

La Fase 3 creció de 1.5–2 a 2.5–3 semanas por dos razones: el rostro animado (3–4 días) y la
decisión de mantenerla como fase única, que obliga a incluir el firmware completo en el entregable.

**Costo estimado de hardware adicional hasta Fase 5**: $10–30 (audio USB para la Pi + sensores de
distancia). El LiDAR (~$100) es opcional y sólo si Fase 5 lo justifica.
**Costo de API**: con router híbrido y prompt caching, del orden de unos pocos dólares al mes en
desarrollo; $0 en modo `ARGOS_LOCAL_ONLY=1`. Gemini-ER a $0.30/MTok de entrada es marginal si sólo
se usa para consultas de localización y no en el bucle de percepción continuo.

---

## Empezamos por

Fase 0 completa + el esqueleto del harness de Fase 1: `AgentLoop`, `PermissionGate`, `ModelRouter`
con `ClaudeProvider`, y **dos o tres skills reales en `src/argos/skills/` con sus pruebas unitarias**
— porque el patrón de skills es lo que decide si la Fase 4 será segura, y conviene establecerlo con
el primer código, no retrofitearlo cuando ya haya motores conectados.

Al final de la primera sesión de trabajo: un agente que responde, respeta permisos, y cuyas
capacidades están acotadas por construcción.

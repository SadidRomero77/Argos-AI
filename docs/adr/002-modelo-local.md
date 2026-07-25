# ADR 002 — Modelo local por defecto: `qwen3:4b`

**Fecha:** 2026-07-24 · **Estado:** aceptada · **Contexto:** Fase 1

## Decisión

`qwen3:4b` como modelo local por defecto (`config/settings.toml`). `qwen3:8b` queda
instalado como alternativa, pero **no** se usa ni siquiera para las tareas de
razonamiento.

## Medición

Banco `argos.evals`, 7 casos contra las skills reales de Fase 1, una pasada por modelo,
`num_ctx=16384`, RTX 3060 Laptop con ~5,0 GiB de VRAM libre:

| modelo | aprobado | skill | params | resp | s/caso |
|---|---|---|---|---|---|
| **qwen3:4b** | **100%** | 100% | 100% | 100% | **26,7 s** |
| qwen3:8b | 86% | 86% | 86% | 100% | 56,6 s |

Único fallo del 8b: `no_inventa_si_no_existe`. Afirmó que el archivo no existía **sin
comprobarlo con `read_file`**. Acertó, pero por suposición — y el prompt de sistema dice
explícitamente que compruebe antes de afirmar. Es un fallo legítimo, no un artefacto.

## Por qué el modelo pequeño gana

Contradice la intuición, así que conviene decir por qué:

1. **El 8b no cabe en VRAM.** Ocupa 5,2 GB frente a ~5,0 GB libres, así que Ollama
   descarga parte a CPU. De ahí el 2× de latencia, que es el efecto dominante.
2. **A esta escala, "más listo" se manifiesta como razonar más, no como elegir mejor.**
   Ambos modelos tienen el mismo `resp=100%`: los dos *saben* la respuesta. La diferencia
   está en la disciplina de verificar antes de afirmar, y ahí el tamaño no ayudó.

## Advertencia sobre la confianza estadística

**Son 7 casos y una sola pasada.** La diferencia de "aprobado" (100% vs 86%) es **un solo
caso** y cae dentro del ruido para esta n. Lo que sí es sólido y consistente caso a caso
es **la latencia: el 4b es 2× más rápido en los siete**. La decisión se apoya
principalmente en eso, no en el porcentaje.

Con más presupuesto de tiempo habría que repetir cada caso 3–5 veces. Se deja anotado
como deuda del banco.

## Consecuencia descartada: enrutar por tarea entre ambos

La idea natural —`ROUTINE`→4b, `PLANNING`→8b— **no es viable en este hardware**. Juntos
suman 7,7 GB sobre 5,0 GB disponibles, así que Ollama tendría que descargar y recargar un
modelo en cada cambio de `TaskKind`. El coste de swap supera cualquier ganancia.

Cuando entre la API de Claude, el router recupera su sentido original: local para lo
rutinario, nube para lo difícil. Ahí no hay contención de VRAM.

## Lo que este banco encontró (y ninguna prueba unitaria habría visto)

Los tres fallos siguientes dependen de cómo se comporta un modelo pequeño de verdad, y
los tres se arreglaron antes de fijar esta decisión:

1. **`reasoning` ignorado.** Ollama expone el razonamiento de Qwen3 fuera de `content`.
   Cuando el 8b agotaba su presupuesto pensando, devolvía `content=""` y el bucle daba el
   turno por `COMPLETED`: tarea sin hacer, sin ninguna señal.
2. **Nombres alucinados por mezcla de idiomas.** Descripciones en español y nombres en
   inglés llevaron al 8b a razonar sobre una función `busca_archivos` —traducida de la
   descripción de `glob`— y a no poder invocar nada. Ahora cada descripción empieza por
   su identificador literal.
3. **Faltaba `glob`.** El 8b eligió `run_command` con parámetros válidos tres veces
   seguidas (376 s) sin llegar al resultado: contar archivos exige una tubería y
   `run_command` las prohíbe por diseño. **El fallo era del diseño, no del modelo.** La
   respuesta correcta según el principio 80/20 no fue un modelo más listo sino una skill
   mejor formada.

Tras los tres arreglos, el caso que costaba 143 s al 4b pasó a 21 s.

## Revisar esta decisión si

- Se libera VRAM (cerrar Windows de fondo, GPU dedicada) → reevaluar el 8b sin swap.
- Se amplía el banco a 20+ casos con repeticiones → la diferencia de acierto podría
  invertirse o confirmarse.
- Aparece un modelo MoE que quepa en 5 GB con activación pequeña.

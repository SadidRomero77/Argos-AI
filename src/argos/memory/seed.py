"""Siembra la memoria con lo que el agente debe saber desde el primer arranque.

Estos hechos son el punto de partida, no una lista cerrada: el agente añade los
suyos según conversa. Se guardan como memoria **semántica** —hechos estables sobre
una entidad— y no como episodios, porque no son "algo que pasó" sino "algo que es".

Ejecutar con:

    uv run python -m argos.memory.seed

Es idempotente: los hechos se deduplican por `(sujeto, hecho)`, así que volver a
lanzarlo no crea copias.

## Privacidad

Todo esto son **datos personales de Sadid y su familia**. Viven en `var/memory.db`,
que está fuera de git. Si se activa un proveedor de LLM en la nube, los hechos que
el agente recupere para responder sí viajarán en el prompt — es inherente a usar un
modelo remoto, y por eso el modo local es el predeterminado.
"""

from __future__ import annotations

from argos.config import Settings, get_settings
from argos.memory.embeddings import Embedder, LexicalEmbedder, OllamaEmbedder
from argos.memory.store import EntityKind, MemoryStore

# (tipo, nombre, [hechos])
SEMILLA: list[tuple[EntityKind, str, list[str]]] = [
    (
        EntityKind.PERSONA,
        "Sadid",
        [
            "es mi creador; construyó ARGOS desde cero",
            "es licenciado en física",
            "tiene una maestría en inteligencia artificial",
            "tiene 31 años",
            "le apasiona la tecnología en general",
            "es geek: le gustan el anime, los cómics, los videojuegos y los superhéroes",
            "le interesan la computación cuántica, la ciberseguridad y la robótica",
            "está casado con Lady",
            "trabaja en WSL2 sobre Windows, con una RTX 3060 de 6 GB",
        ],
    ),
    (
        EntityKind.PERSONA,
        "Lady",
        [
            "es la esposa de Sadid",
            "es veterinaria",
            "tiene 31 años",
        ],
    ),
    (
        EntityKind.PERSONA,
        "Caronte",
        [
            "es uno de los dos perros de Sadid y Lady",
            "es un border collie",
        ],
    ),
    (
        EntityKind.PERSONA,
        "Caramelo",
        [
            "es uno de los dos perros de Sadid y Lady",
            "es criollo, de raza mezclada",
        ],
    ),
    (
        EntityKind.TEMA,
        "ARGOS",
        [
            "es este proyecto: un agente de IA autónomo que se encarna en hardware por fases",
            "su principio rector es que el LLM nunca cierra un lazo de control",
            "corre en local con qwen3:4b sobre Ollama, y puede usar Claude si se configura",
        ],
    ),
]


def build_embedder(settings: Settings | None = None) -> tuple[Embedder, str]:
    """Devuelve el mejor embedder disponible y cómo se llama, para poder avisar."""
    settings = settings or get_settings()
    ollama = OllamaEmbedder()
    ok, mensaje = ollama.health()
    if ok:
        return ollama, mensaje
    # Degradar es mejor que fallar: con el léxico la memoria funciona peor pero
    # funciona, y el agente arranca aunque no haya servidor.
    return LexicalEmbedder(), f"sin servidor de embeddings ({mensaje}); usando respaldo léxico"


def seed(store: MemoryStore) -> int:
    """Inserta los hechos. Devuelve cuántos había en total al terminar."""
    for kind, nombre, hechos in SEMILLA:
        entidad_id = store.upsert_entity(kind, nombre)
        for hecho in hechos:
            store.remember_fact(nombre, hecho, entity_id=entidad_id)
    return store.stats()["semantic"]


def main() -> int:
    settings = get_settings()
    embedder, aviso = build_embedder(settings)
    print(f"· {aviso}")

    with MemoryStore(settings.paths.resolved("memory_db"), embedder) as store:
        total = seed(store)
        print(f"· memoria sembrada: {total} hechos, {store.stats()['entity']} entidades")
        for consulta in ("¿quién soy?", "¿cómo se llaman mis perros?", "¿a qué se dedica Lady?"):
            hechos = store.recall_facts(consulta, limit=2)
            resumen = "; ".join(f"{h.subject}: {h.fact}" for h in hechos) or "(nada relevante)"
            print(f"  {consulta:32} → {resumen[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

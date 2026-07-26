"""Skills de memoria: el agente decide qué merece recordarse.

Que el propio agente elija qué guardar es deliberado. La alternativa —extraer
hechos automáticamente de cada turno con otra llamada al modelo— duplica el coste
y llena la memoria de ruido ("el usuario dijo hola"). Aquí sólo se guarda lo que
el agente considera que servirá en el futuro, y el prompt de sistema le dice
cuándo hacerlo.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from argos.memory.store import EntityKind, MemoryStore
from argos.skills.base import Skill, SkillResult


class RememberParams(BaseModel):
    subject: str = Field(
        description="De quién o de qué trata el dato, ej. 'Sadid', 'Lady', 'ARGOS'"
    )
    fact: str = Field(
        description="El dato, en una frase y en tercera persona, ej. 'prefiere respuestas cortas'"
    )
    kind: str = Field(
        default="persona",
        description="Tipo de entidad: persona, objeto, lugar o tema",
    )


class Remember(Skill):
    name = "remember"
    description = (
        "[remember] Guarda un dato duradero sobre alguien o algo. Úsala cuando en la "
        "conversación aparezca información que seguirá siendo cierta mañana: nombres, "
        "gustos, relaciones, preferencias de trabajo, decisiones tomadas. NO la uses "
        "para lo que sólo vale ahora ni para lo que ya está en tu memoria."
    )
    Params = RememberParams

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def precondition(self, params: RememberParams) -> str | None:
        if len(params.fact.strip()) < 4:
            return "el dato está vacío o es demasiado corto para ser útil"
        return None

    def run(self, params: RememberParams) -> SkillResult:
        try:
            kind = EntityKind(params.kind.lower())
        except ValueError:
            kind = EntityKind.TEMA  # un tipo desconocido no debe impedir recordar

        entidad_id = self.store.upsert_entity(kind, params.subject)
        fact_id = self.store.remember_fact(params.subject, params.fact, entity_id=entidad_id)
        return SkillResult.success(
            f"Recordado sobre {params.subject}: {params.fact}", fact_id=fact_id
        )


class RecallParams(BaseModel):
    query: str = Field(description="Qué quieres recordar, en lenguaje natural")
    subject: str | None = Field(
        default=None, description="Limitar a una persona o tema concreto (opcional)"
    )


class Recall(Skill):
    name = "recall"
    description = (
        "[recall] Busca en tu memoria datos sobre alguien o algo. Úsala cuando te "
        "pregunten por información personal que deberías saber y no la tengas ya en "
        "el contexto de esta conversación. Si no encuentra nada, dilo — no inventes."
    )
    Params = RecallParams

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(self, params: RecallParams) -> SkillResult:
        hechos = self.store.recall_facts(params.query, limit=8, subject=params.subject)
        episodios = self.store.recall_episodes(params.query, limit=3)

        if not hechos and not episodios:
            # Decirlo explícitamente evita que el modelo rellene el hueco.
            return SkillResult.success("No hay nada en la memoria sobre eso.", found=0)

        partes = []
        if hechos:
            partes.append("Datos:\n" + "\n".join(f"- {h.subject}: {h.fact}" for h in hechos))
        if episodios:
            partes.append(
                "De conversaciones anteriores:\n" + "\n".join(f"- {e.content}" for e in episodios)
            )
        return SkillResult.success("\n\n".join(partes), found=len(hechos) + len(episodios))


class IdentifyParams(BaseModel):
    name: str = Field(description="Cómo se llama, o cómo prefiere que le llamen")


class IdentifySpeaker(Skill):
    """Ata la conversación en curso a una persona concreta.

    Es el punto de unión entre lo que hay hoy y lo que vendrá: hoy la identidad se
    resuelve **por nombre**, porque es lo único disponible en una conversación de
    texto o voz. En Fase 2 el reconocimiento facial llamará a este mismo mecanismo
    con el `entity_id` que resuelva InsightFace, y todo lo de aguas abajo —
    episodios asociados, perfil recuperado— sigue igual. Por eso la entidad se crea
    aquí y no en un sitio distinto para cada vía.
    """

    name = "identify_speaker"
    description = (
        "[identify_speaker] Registra con quién estás hablando, por su nombre. Úsala "
        "en cuanto alguien te diga cómo se llama o cómo quiere que le llames. A "
        "partir de ese momento todo lo que te cuente queda asociado a esa persona, "
        "y la próxima vez que hable la reconocerás."
    )
    Params = IdentifyParams

    def __init__(self, store: MemoryStore, on_identified: Callable[[str, int], None]) -> None:
        self.store = store
        self.on_identified = on_identified

    def precondition(self, params: IdentifyParams) -> str | None:
        if len(params.name.strip()) < 2:
            return "el nombre está vacío o es demasiado corto"
        return None

    def run(self, params: IdentifyParams) -> SkillResult:
        nombre = params.name.strip()
        conocido = self.store.get_entity(EntityKind.PERSONA, nombre) is not None
        entidad_id = self.store.upsert_entity(EntityKind.PERSONA, nombre)
        self.on_identified(nombre, entidad_id)

        if conocido:
            hechos = self.store.facts_about(nombre, limit=10)
            resumen = "; ".join(h.fact for h in hechos) or "(sin datos guardados)"
            return SkillResult.success(
                f"Ya conocías a {nombre}. Lo que sabes: {resumen}",
                entity_id=entidad_id,
                known=True,
            )
        return SkillResult.success(
            f"Ahora hablas con {nombre}, alguien nuevo. Lo que te cuente se guardará "
            f"asociado a esa persona.",
            entity_id=entidad_id,
            known=False,
        )


class ForgetParams(BaseModel):
    subject: str = Field(description="Persona o tema a borrar por completo")
    kind: str = Field(default="persona", description="persona, objeto, lugar o tema")


class Forget(Skill):
    name = "forget"
    description = (
        "[forget] Borra de tu memoria todo lo relacionado con alguien o algo. Úsala "
        "cuando te lo pidan explícitamente. Es irreversible."
    )
    Params = ForgetParams

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(self, params: ForgetParams) -> SkillResult:
        try:
            kind = EntityKind(params.kind.lower())
        except ValueError:
            kind = EntityKind.TEMA
        borrados = self.store.forget_entity(kind, params.subject)
        if not borrados:
            return SkillResult.success(f"No había nada guardado sobre {params.subject}.")
        return SkillResult.success(f"Borrado todo lo relacionado con {params.subject}.")

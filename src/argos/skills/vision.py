"""Skills de visión: mirar por la cámara y reconocer a quién se ve.

Aquí se cierra el círculo que la memoria dejó abierto desde el primer día. El
vector facial entra en `entity.embedding`, la misma columna que ya usaba la
identificación por nombre, y al reconocer a alguien se llama al **mismo**
`on_identified` que usa `identify_speaker`. Una persona reconocida por su cara y
una que dijo su nombre acaban siendo la misma entidad, con la misma memoria.

Es, literalmente, el perro de Odiseo: reconocer a alguien a través del tiempo.

## El principio 80/20, aquí más claro que en ningún sitio

**InsightFace decide quién es alguien. El LLM sólo lo cuenta.** El modelo nunca ve
el vector ni decide si dos caras coinciden — recibe un nombre y un grado de
confianza ya calculados. Una alucinación no puede convertir a una persona en otra.

## Privacidad

El enrolamiento es **explícito**: ARGOS no guarda la cara de nadie por su cuenta.
Hace falta que alguien diga su nombre. Y `forget` borra el vector junto con todo
lo demás de esa persona, en cascada.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from pydantic import BaseModel, Field

from argos.memory.store import EntityKind, MemoryStore
from argos.perception.camera import CameraBridge
from argos.perception.hands import ExpressionReader, HandCounter
from argos.perception.identity import UMBRAL, FaceRecognizer
from argos.perception.vlm import VisionLanguageModel
from argos.skills.base import Skill, SkillResult


class SinParams(BaseModel):
    pass


class _AvisaInterlocutor:
    """Notifica al gateway quién está delante, sin que un fallo suyo rompa la skill.

    El aviso es una consecuencia de haber reconocido a alguien, no parte de
    reconocerlo: cuando ya se guardó la cara o ya se identificó a la persona,
    reportar fracaso porque el callback reventó sería mentir sobre lo que pasó.
    """

    on_identified: Callable[[str, int], None] | None

    def _avisar(self, nombre: str, entidad_id: int) -> None:
        if self.on_identified is None:
            return
        with contextlib.suppress(Exception):
            self.on_identified(nombre, entidad_id)


class WhoIsThis(Skill, _AvisaInterlocutor):
    name = "who_is_this"
    description = (
        "[who_is_this] Mira por la cámara y dice a quién ve. Úsala cuando pregunten "
        "quién está delante, o cuando quieras saber con quién hablas. Si no reconoce "
        "a la persona lo dice claramente — entonces pregúntale su nombre y usa "
        "`remember_face`. Nunca supongas de quién se trata."
    )
    Params = SinParams

    def __init__(
        self,
        store: MemoryStore,
        camera: CameraBridge,
        recognizer: FaceRecognizer,
        on_identified: Callable[[str, int], None] | None = None,
    ) -> None:
        self.store = store
        self.camera = camera
        self.recognizer = recognizer
        self.on_identified = on_identified

    def precondition(self, params: SinParams) -> str | None:
        ok, mensaje = self.camera.health()
        return None if ok else mensaje

    def run(self, params: SinParams) -> SkillResult:
        cara = self.recognizer.primary(self.camera.grab().jpeg)
        if cara is None:
            return SkillResult.success("No veo ninguna cara delante de la cámara.", faces=0)

        encontrado = self.store.find_entity_by_embedding(
            cara.embedding, EntityKind.PERSONA, threshold=UMBRAL
        )
        if encontrado is None:
            return SkillResult.success(
                "Veo a alguien, pero no lo reconozco. Pregúntale cómo se llama y usa "
                "remember_face para recordarlo.",
                faces=1,
                known=False,
            )

        entidad, similitud = encontrado
        # Verlo cuenta como interacción: actualiza `last_seen_at`.
        self.store.upsert_entity(EntityKind.PERSONA, entidad.label)
        self._avisar(entidad.label, entidad.id)

        hechos = self.store.facts_about(entidad.label, limit=8)
        resumen = "; ".join(h.fact for h in hechos)
        salida = f"Es {entidad.label} (confianza {similitud:.2f})."
        if resumen:
            salida += f" Lo que sabes: {resumen}"

        return SkillResult.success(
            salida, faces=1, known=True, person=entidad.label, confidence=round(similitud, 3)
        )


class RememberFaceParams(BaseModel):
    name: str = Field(description="Cómo se llama, o cómo prefiere que le llamen")


class RememberFace(Skill, _AvisaInterlocutor):
    name = "remember_face"
    description = (
        "[remember_face] Asocia la cara que ve la cámara con un nombre, para "
        "reconocer a esa persona la próxima vez. Úsala SÓLO después de que alguien "
        "te haya dicho su nombre — nunca guardes la cara de nadie por tu cuenta."
    )
    Params = RememberFaceParams

    def __init__(
        self,
        store: MemoryStore,
        camera: CameraBridge,
        recognizer: FaceRecognizer,
        on_identified: Callable[[str, int], None] | None = None,
    ) -> None:
        self.store = store
        self.camera = camera
        self.recognizer = recognizer
        self.on_identified = on_identified

    def precondition(self, params: RememberFaceParams) -> str | None:
        if len(params.name.strip()) < 2:
            return "el nombre está vacío o es demasiado corto"
        ok, mensaje = self.camera.health()
        return None if ok else mensaje

    def run(self, params: RememberFaceParams) -> SkillResult:
        cara = self.recognizer.primary(self.camera.grab().jpeg)
        if cara is None:
            return SkillResult.fail(
                "no veo ninguna cara. Pídele que se ponga delante de la cámara."
            )

        nombre = params.name.strip()
        # Si ya existe por nombre —porque se identificó hablando— se le añade la
        # cara a esa misma entidad en vez de crear una duplicada. Es el punto
        # donde las dos vías de identidad convergen.
        entidad_id = self.store.upsert_entity(EntityKind.PERSONA, nombre, embedding=cara.embedding)
        self._avisar(nombre, entidad_id)

        return SkillResult.success(
            f"Guardada la cara de {nombre}. La próxima vez que la vea la reconoceré.",
            person=nombre,
            entity_id=entidad_id,
        )


class SeeParams(BaseModel):
    pass


class LookAround(Skill):
    name = "look"
    description = (
        "[look] Mira por la cámara y dice cuántas personas hay y dónde están en el "
        "encuadre. No describe objetos ni la escena — para eso hará falta un modelo "
        "de visión, que aún no está instalado. Úsala para saber si hay alguien."
    )
    Params = SeeParams

    def __init__(self, camera: CameraBridge, recognizer: FaceRecognizer) -> None:
        self.camera = camera
        self.recognizer = recognizer

    def precondition(self, params: SeeParams) -> str | None:
        ok, mensaje = self.camera.health()
        return None if ok else mensaje

    def run(self, params: SeeParams) -> SkillResult:
        fotograma = self.camera.grab()
        caras = self.recognizer.detect(fotograma.jpeg)
        if not caras:
            return SkillResult.success("No hay nadie delante de la cámara.", faces=0)

        # La posición se describe en palabras, no en píxeles: el modelo no tiene
        # que interpretar coordenadas, que es justo donde se equivoca.
        descripciones = []
        for cara in caras:
            x, _ = cara.center
            tercio = (
                "a la izquierda" if x < 426 else ("en el centro" if x < 853 else "a la derecha")
            )
            cerca = "cerca" if cara.area > 40_000 else "de lejos"
            descripciones.append(f"{tercio}, {cerca}")

        cuantas = "una persona" if len(caras) == 1 else f"{len(caras)} personas"
        return SkillResult.success(
            f"Veo {cuantas}: " + "; ".join(descripciones) + ".", faces=len(caras)
        )


class SeeParams2(BaseModel):
    question: str = Field(
        description=(
            "Qué quieres saber de la imagen, EN INGLÉS y en una frase corta. "
            "Ej.: 'What is the person holding?', 'What color is the shirt?'"
        )
    )


class See(Skill):
    """Pregunta abierta sobre lo que hay delante de la cámara."""

    name = "see"
    description = (
        "[see] Mira por la cámara y responde una pregunta sobre lo que hay: objetos, "
        "colores, qué sostiene alguien, cómo es el sitio. La pregunta va EN INGLÉS "
        "porque el modelo de visión sólo entiende ese idioma — tú traduces la "
        "respuesta al español. Para contar dedos usa `count_fingers` y para saber si "
        "alguien sonríe usa `read_expression`: son exactos, esto es una descripción "
        "y puede equivocarse."
    )
    Params = SeeParams2

    def __init__(self, camera: CameraBridge, vlm: VisionLanguageModel) -> None:
        self.camera = camera
        self.vlm = vlm

    def precondition(self, params: SeeParams2) -> str | None:
        ok, mensaje = self.camera.health()
        if not ok:
            return mensaje
        ok, mensaje = self.vlm.health()
        return None if ok else mensaje

    # Por debajo de esto la respuesta es un fragmento, no una respuesta.
    _MINIMO_PALABRAS = 4

    def run(self, params: SeeParams2) -> SkillResult:
        jpeg = self.camera.grab().jpeg
        respuesta = self.vlm.ask(jpeg, params.question)
        texto = respuesta.text

        # Moondream contesta con fragmentos a preguntas específicas ("urns") pero
        # describe bien la escena entera. Cuando la respuesta se queda en dos
        # palabras se le pide además la descripción general, para que el agente
        # tenga contexto con el que responder en vez de repetir el fragmento.
        completado = False
        if len(texto.split()) < self._MINIMO_PALABRAS:
            general = self.vlm.describe(jpeg)
            if not general.is_empty:
                texto = f"{texto}. En la escena: {general.text}" if texto else general.text
                completado = True

        if not texto.strip():
            return SkillResult.success(
                "El modelo de visión no supo responder a eso. Prueba con una pregunta "
                "más simple, o pídele que describa la escena entera.",
                answered=False,
            )

        # Sin prefijos explicativos: cualquier envoltorio acaba copiado literalmente
        # en la respuesta al usuario. El agente ya sabe que esto viene de la cámara.
        return SkillResult.success(
            texto, answered=True, seconds=respuesta.seconds, completed=completado
        )


class CountFingers(Skill):
    """Conteo exacto por geometría. Sin modelo de lenguaje de por medio."""

    name = "count_fingers"
    description = (
        "[count_fingers] Cuenta cuántos dedos está mostrando quien está delante de la "
        "cámara, y reconoce gestos como el puño o la señal de victoria. El conteo es "
        "EXACTO: se calcula con los puntos de la mano, no lo estima ningún modelo. "
        "Úsala siempre para contar dedos — nunca lo deduzcas mirando la escena."
    )
    Params = SinParams

    def __init__(self, camera: CameraBridge, counter: HandCounter) -> None:
        self.camera = camera
        self.counter = counter

    def precondition(self, params: SinParams) -> str | None:
        ok, mensaje = self.camera.health()
        if not ok:
            return mensaje
        ok, mensaje = self.counter.health()
        return None if ok else mensaje

    def run(self, params: SinParams) -> SkillResult:
        manos = self.counter.count(self.camera.grab().jpeg)
        if not manos:
            return SkillResult.success(
                "No veo ninguna mana delante de la cámara.".replace("mana", "mano"), hands=0
            )

        partes = []
        for mano in manos:
            texto = f"{mano.fingers} dedo{'s' if mano.fingers != 1 else ''}"
            if mano.gesture:
                texto += f" ({mano.gesture})"
            partes.append(texto)

        total = sum(m.fingers for m in manos)
        salida = " y ".join(partes)
        if len(manos) > 1:
            salida += f". En total, {total}"
        return SkillResult.success(
            salida + ".",
            hands=len(manos),
            total=total,
            gestures=[m.gesture for m in manos if m.gesture],
        )


class ReadExpression(Skill):
    """Sonrisa, ojos y boca por blendshapes faciales. También determinista."""

    name = "read_expression"
    description = (
        "[read_expression] Dice si quien está delante sonríe, tiene los ojos cerrados "
        "o la boca abierta. Se calcula con la geometría de la cara, así que es fiable. "
        "Úsala cuando pregunten por la expresión — no la deduzcas de una descripción."
    )
    Params = SinParams

    def __init__(self, camera: CameraBridge, reader: ExpressionReader) -> None:
        self.camera = camera
        self.reader = reader

    def precondition(self, params: SinParams) -> str | None:
        ok, mensaje = self.camera.health()
        if not ok:
            return mensaje
        ok, mensaje = self.reader.health()
        return None if ok else mensaje

    def run(self, params: SinParams) -> SkillResult:
        expresion = self.reader.read(self.camera.grab().jpeg)
        if expresion is None:
            return SkillResult.success("No veo ninguna cara delante de la cámara.", face=False)

        rasgos = ["sonriendo" if expresion.smiling else "sin sonreír"]
        if expresion.eyes_closed:
            rasgos.append("con los ojos cerrados")
        if expresion.mouth_open:
            rasgos.append("con la boca abierta")

        return SkillResult.success(
            "Está " + ", ".join(rasgos) + ".",
            face=True,
            smiling=expresion.smiling,
            smile_score=expresion.smile_score,
        )

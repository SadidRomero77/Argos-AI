"""Router y proveedor Claude, probados sin tocar la red.

Las pruebas del proveedor usan un cliente falso con la misma forma que el SDK real,
para verificar lo que de verdad rompe en producción: que la petición se construya
según las capacidades del modelo, y que un `refusal` no reviente el parseo.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from argos.config import Settings
from argos.models.base import (
    Effort,
    LLMProvider,
    LLMResponse,
    TaskKind,
    ToolCall,
    Usage,
    caps_for,
)
from argos.models.claude import ClaudeProvider
from argos.models.router import BudgetExceeded, LocalProviderUnavailable, ModelRouter
from argos.trace import Tracer

# ───────────────────────────── dobles de prueba ───────────────────────────────


class FakeProvider(LLMProvider):
    def __init__(self, model: str = "fake-1", usage: Usage | None = None) -> None:
        self._model = model
        self.usage = usage or Usage(input_tokens=1000, output_tokens=500)
        self.llamadas: list[dict] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages, *, system=None, tools=None, max_tokens=8192, effort=None):
        self.llamadas.append({"messages": messages, "effort": effort, "max_tokens": max_tokens})
        return LLMResponse(text="ok", model=self._model, usage=self.usage)


class FakeAnthropicClient:
    """Imita `anthropic.Anthropic` lo justo para inspeccionar la petición."""

    def __init__(self, respuesta) -> None:
        self.respuesta = respuesta
        self.ultima_peticion: dict = {}
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.ultima_peticion = kwargs
        return self.respuesta


def bloque_texto(texto: str):
    return SimpleNamespace(type="text", text=texto)


def bloque_tool(id_: str, nombre: str, entrada: dict):
    return SimpleNamespace(type="tool_use", id=id_, name=nombre, input=entrada)


def mensaje(content, stop_reason="end_turn", model="claude-haiku-4-5", **uso):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(
            input_tokens=uso.get("input_tokens", 0),
            output_tokens=uso.get("output_tokens", 0),
            cache_read_input_tokens=uso.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=uso.get("cache_creation_input_tokens", 0),
        ),
    )


# ──────────────────────── capacidades y coste ─────────────────────────────────


def test_haiku_no_soporta_effort_ni_thinking_adaptativo():
    """Enviarle effort a Haiku 4.5 es un 400. La tabla evita ese error."""
    caps = caps_for("claude-haiku-4-5")
    assert caps.supports_effort is False
    assert caps.adaptive_thinking is False


def test_sonnet_y_opus_5_si_los_soportan():
    for modelo in ("claude-sonnet-5", "claude-opus-5"):
        caps = caps_for(modelo)
        assert caps.supports_effort and caps.adaptive_thinking


def test_un_modelo_desconocido_se_trata_como_local_conservadoramente():
    caps = caps_for("qwen3-cualquiera")
    assert caps.supports_effort is False
    assert caps.usd_in_per_mtok == 0.0


def test_calculo_de_coste():
    uso = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert uso.cost_usd("claude-haiku-4-5") == pytest.approx(6.0)  # 1 + 5
    assert uso.cost_usd("claude-opus-5") == pytest.approx(30.0)  # 5 + 25


def test_la_lectura_de_cache_cuesta_una_decima_parte():
    caro = Usage(input_tokens=1_000_000).cost_usd("claude-opus-5")
    barato = Usage(cache_read_tokens=1_000_000).cost_usd("claude-opus-5")
    assert barato == pytest.approx(caro * 0.1)


def test_los_usos_se_suman():
    total = Usage(input_tokens=10, output_tokens=5) + Usage(input_tokens=3, output_tokens=2)
    assert (total.input_tokens, total.output_tokens) == (13, 7)


# ──────────────────────────── ClaudeProvider ──────────────────────────────────


def test_la_peticion_a_haiku_no_lleva_effort_ni_thinking():
    cliente = FakeAnthropicClient(mensaje([bloque_texto("hola")]))
    provider = ClaudeProvider("claude-haiku-4-5", api_key="x", client=cliente)

    provider.complete([{"role": "user", "content": "hola"}], effort=Effort.HIGH)

    assert "output_config" not in cliente.ultima_peticion
    assert "thinking" not in cliente.ultima_peticion


def test_la_peticion_a_opus_5_si_los_lleva():
    cliente = FakeAnthropicClient(mensaje([bloque_texto("hola")], model="claude-opus-5"))
    provider = ClaudeProvider("claude-opus-5", api_key="x", client=cliente)

    provider.complete([{"role": "user", "content": "hola"}], effort=Effort.XHIGH)

    assert cliente.ultima_peticion["thinking"] == {"type": "adaptive"}
    assert cliente.ultima_peticion["output_config"] == {"effort": "xhigh"}


def test_el_system_prompt_va_marcado_para_cache():
    """Es la parte estable del prefijo; cachearla ahorra ~90% en ella."""
    cliente = FakeAnthropicClient(mensaje([bloque_texto("hola")]))
    provider = ClaudeProvider("claude-haiku-4-5", api_key="x", client=cliente)

    provider.complete([{"role": "user", "content": "hola"}], system="eres ARGOS")

    bloque = cliente.ultima_peticion["system"][0]
    assert bloque["cache_control"] == {"type": "ephemeral"}
    assert bloque["text"] == "eres ARGOS"


def test_extrae_llamadas_a_herramientas():
    cliente = FakeAnthropicClient(
        mensaje(
            [bloque_texto("voy a mirar"), bloque_tool("tu_1", "read_file", {"path": "a.md"})],
            stop_reason="tool_use",
        )
    )
    provider = ClaudeProvider("claude-haiku-4-5", api_key="x", client=cliente)

    respuesta = provider.complete([{"role": "user", "content": "lee a.md"}])

    assert respuesta.wants_tools
    assert respuesta.tool_calls == [ToolCall(id="tu_1", name="read_file", params={"path": "a.md"})]
    assert respuesta.text == "voy a mirar"


def test_un_refusal_con_content_vacio_no_revienta():
    """Los clasificadores declinan con HTTP 200 y content vacío."""
    cliente = FakeAnthropicClient(mensaje([], stop_reason="refusal"))
    provider = ClaudeProvider("claude-opus-5", api_key="x", client=cliente)

    respuesta = provider.complete([{"role": "user", "content": "..."}])

    assert respuesta.refused
    assert respuesta.text == ""
    assert not respuesta.wants_tools


def test_conserva_los_bloques_de_thinking_verbatim():
    """La API rechaza bloques de thinking alterados al continuar la conversación."""
    thinking = SimpleNamespace(type="thinking", thinking="mmm", signature="firma-abc")
    cliente = FakeAnthropicClient(mensaje([thinking, bloque_texto("resultado")]))
    provider = ClaudeProvider("claude-opus-5", api_key="x", client=cliente)

    respuesta = provider.complete([{"role": "user", "content": "x"}])

    assert respuesta.raw_content[0] == {
        "type": "thinking",
        "thinking": "mmm",
        "signature": "firma-abc",
    }


# ─────────────────────────────── ModelRouter ──────────────────────────────────


def test_enruta_cada_tipo_de_tarea_a_su_proveedor():
    rutina, razonamiento = FakeProvider("rapido"), FakeProvider("listo")
    router = ModelRouter(
        providers={TaskKind.ROUTINE: rutina, TaskKind.REASONING: razonamiento},
        settings=Settings(),
    )
    assert router.provider_for(TaskKind.ROUTINE).model == "rapido"
    assert router.provider_for(TaskKind.REASONING).model == "listo"


def test_el_effort_por_defecto_depende_del_tipo_de_tarea():
    provider = FakeProvider()
    router = ModelRouter(providers={TaskKind.PLANNING: provider}, settings=Settings())

    router.complete(TaskKind.PLANNING, [{"role": "user", "content": "planifica"}])

    assert provider.llamadas[0]["effort"] is Effort.XHIGH


def test_local_only_no_llama_a_la_nube_ni_siquiera_como_respaldo(monkeypatch):
    """El control clave: en modo local, la ausencia de proveedor falla ruidosamente."""
    monkeypatch.setenv("ARGOS_MODELS__LOCAL_ONLY", "true")
    router = ModelRouter(providers={TaskKind.ROUTINE: FakeProvider("nube")}, settings=Settings())

    with pytest.raises(LocalProviderUnavailable, match="API externa"):
        router.provider_for(TaskKind.ROUTINE)


def test_local_only_usa_el_proveedor_local_para_todo(monkeypatch):
    monkeypatch.setenv("ARGOS_MODELS__LOCAL_ONLY", "true")
    local = FakeProvider("qwen-local")
    router = ModelRouter(
        providers={TaskKind.ROUTINE: FakeProvider("nube")},
        local_provider=local,
        settings=Settings(),
    )
    for kind in TaskKind:
        assert router.provider_for(kind).model == "qwen-local"


def test_acumula_uso_y_coste_por_modelo():
    provider = FakeProvider("claude-haiku-4-5", usage=Usage(input_tokens=1_000_000))
    router = ModelRouter(providers={TaskKind.ROUTINE: provider}, settings=Settings())

    router.complete(TaskKind.ROUTINE, [{"role": "user", "content": "a"}])
    router.complete(TaskKind.ROUTINE, [{"role": "user", "content": "b"}])

    assert router.total_usd == pytest.approx(2.0)  # 2 M tokens de entrada a $1/M
    assert router.usage_by_model()["claude-haiku-4-5"].input_tokens == 2_000_000


def test_el_presupuesto_es_un_tope_duro(monkeypatch):
    monkeypatch.setenv("ARGOS_BUDGET__MAX_USD_PER_DAY", "1.5")
    provider = FakeProvider("claude-haiku-4-5", usage=Usage(input_tokens=1_000_000))
    router = ModelRouter(providers={TaskKind.ROUTINE: provider}, settings=Settings())

    router.complete(TaskKind.ROUTINE, [{"role": "user", "content": "a"}])  # $1.00, pasa
    router.complete(TaskKind.ROUTINE, [{"role": "user", "content": "b"}])  # $2.00 acumulado

    with pytest.raises(BudgetExceeded, match="presupuesto agotado"):
        router.complete(TaskKind.ROUTINE, [{"role": "user", "content": "c"}])

    assert len(provider.llamadas) == 2, "no debe haber gastado en la tercera llamada"


def test_traza_la_decision_del_router_y_el_coste(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGOS_PATHS__TRACES", str(tmp_path / "traces"))
    tracer = Tracer(settings=Settings())
    provider = FakeProvider("claude-haiku-4-5", usage=Usage(input_tokens=1000, output_tokens=100))
    router = ModelRouter(
        providers={TaskKind.REASONING: provider}, settings=Settings(), tracer=tracer
    )

    router.complete(TaskKind.REASONING, [{"role": "user", "content": "analiza"}])

    lineas = {linea["event"]: linea for linea in tracer.read_all()}
    assert lineas["router_decision"]["kind"] == "reasoning"
    assert lineas["model_call"]["model"] == "claude-haiku-4-5"
    assert lineas["model_call"]["usd"] > 0


def test_sin_api_key_y_sin_modo_local_el_error_es_accionable():
    """El caso del primer arranque: aún no hay .env. El mensaje debe decir qué hacer."""
    settings = Settings()  # el conftest garantiza que no hay .env ni variables
    assert settings.anthropic_api_key is None

    with pytest.raises(RuntimeError, match=r"\.env"):
        ModelRouter.from_settings(settings=settings)

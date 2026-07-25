"""LocalProvider: la traducción Anthropic <-> OpenAI, probada sin servidor.

Aquí está el grueso del riesgo de este módulo. Un error de traducción no se
manifiesta como una excepción sino como un agente que "se olvida" de lo que hizo
—porque un tool_result se perdió— o que entra en bucle repitiendo la misma
llamada. Estas pruebas fijan la forma exacta del cable.
"""

from __future__ import annotations

import json

import httpx

from argos.models.local import (
    LocalProvider,
    parse_tool_calls,
    to_openai_messages,
    to_openai_tools,
)

# ─────────────────────────── traducción de tools ──────────────────────────────


def test_las_tools_pasan_al_formato_function():
    anthropic = [
        {
            "name": "read_file",
            "description": "Lee un archivo",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]
    assert to_openai_tools(anthropic) == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Lee un archivo",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]


def test_sin_tools_no_se_manda_el_campo():
    assert to_openai_tools(None) is None
    assert to_openai_tools([]) is None


# ────────────────────────── traducción de mensajes ────────────────────────────


def test_el_system_va_como_primer_mensaje():
    salida = to_openai_messages([{"role": "user", "content": "hola"}], system="eres ARGOS")
    assert salida[0] == {"role": "system", "content": "eres ARGOS"}
    assert salida[1] == {"role": "user", "content": "hola"}


def test_un_assistant_con_tool_use_produce_tool_calls():
    entrada = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "voy a leerlo"},
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.md"}},
            ],
        }
    ]
    (mensaje,) = to_openai_messages(entrada)

    assert mensaje["role"] == "assistant"
    assert mensaje["content"] == "voy a leerlo"
    llamada = mensaje["tool_calls"][0]
    assert llamada["id"] == "t1"
    assert llamada["function"]["name"] == "read_file"
    # OpenAI espera los argumentos como CADENA JSON, no como objeto.
    assert json.loads(llamada["function"]["arguments"]) == {"path": "a.md"}


def test_cada_tool_result_se_convierte_en_su_propio_mensaje_role_tool():
    """En Anthropic van juntos en un turno de usuario; en OpenAI son N mensajes."""
    entrada = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "resultado A"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "resultado B"},
            ],
        }
    ]
    salida = to_openai_messages(entrada)

    assert len(salida) == 2
    assert [m["role"] for m in salida] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in salida] == ["t1", "t2"]
    assert salida[0]["content"] == "resultado A"


def test_no_se_pierde_ningun_tool_result_en_un_turno_mixto():
    """Perder uno hace que el modelo repita la llamada indefinidamente."""
    entrada = [
        {"role": "user", "content": "lee dos archivos"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}},
                {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "b"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "A"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "B"},
            ],
        },
    ]
    salida = to_openai_messages(entrada)

    ids_pedidos = {c["id"] for c in salida[1]["tool_calls"]}
    ids_respondidos = {m["tool_call_id"] for m in salida if m["role"] == "tool"}
    assert ids_pedidos == ids_respondidos == {"t1", "t2"}


def test_los_bloques_de_thinking_se_descartan():
    """No tienen equivalente en OpenAI; mandarlos rompe algunos servidores."""
    entrada = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "mmm", "signature": "x"},
                {"type": "text", "text": "respuesta"},
            ],
        }
    ]
    (mensaje,) = to_openai_messages(entrada)
    assert mensaje["content"] == "respuesta"
    assert "thinking" not in json.dumps(mensaje)


def test_el_system_en_bloques_con_cache_control_se_aplana():
    from argos.models.local import _flatten_system

    bloques = [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert _flatten_system(bloques) == "SYS"


# ─────────────────── robustez ante modelos pequeños ───────────────────────────


def test_json_malformado_en_arguments_no_lanza_excepcion():
    """Fallo habitual en modelos de 4B. Reventar aquí mataría el bucle entero."""
    calls = parse_tool_calls(
        [{"id": "t1", "function": {"name": "read_file", "arguments": "{path: a.md"}}]
    )
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].params == {}  # la skill devolverá un error legible al modelo


def test_arguments_ya_parseado_como_dict_tambien_funciona():
    """Algunos servidores no lo serializan a cadena."""
    calls = parse_tool_calls(
        [{"id": "t1", "function": {"name": "eco", "arguments": {"texto": "hola"}}}]
    )
    assert calls[0].params == {"texto": "hola"}


def test_arguments_que_es_json_valido_pero_no_un_objeto():
    calls = parse_tool_calls([{"id": "t1", "function": {"name": "eco", "arguments": "[1,2,3]"}}])
    assert calls[0].params == {}


def test_una_llamada_sin_id_recibe_uno_sintetico():
    """Sin id no hay forma de emparejar el tool_result. Se genera uno estable."""
    calls = parse_tool_calls([{"function": {"name": "eco", "arguments": "{}"}}])
    assert calls[0].id == "call_0"


# ───────────────────────── respuesta del servidor ─────────────────────────────


def responder(payload: dict, captura: list | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if captura is not None:
            captura.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_parsea_una_respuesta_de_texto():
    cliente = responder(
        {
            "model": "qwen3:4b",
            "choices": [{"message": {"content": "hola"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }
    )
    respuesta = LocalProvider("qwen3:4b", client=cliente).complete(
        [{"role": "user", "content": "hola"}]
    )

    assert respuesta.text == "hola"
    assert respuesta.stop_reason == "end_turn"
    assert respuesta.usage.input_tokens == 12
    assert respuesta.usage.cost_usd("qwen3:4b") == 0.0  # local = gratis


def test_parsea_una_respuesta_con_tool_calls():
    cliente = responder(
        {
            "model": "qwen3:4b",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "docs/PLAN.md"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )
    respuesta = LocalProvider("qwen3:4b", client=cliente).complete(
        [{"role": "user", "content": "lee el plan"}]
    )

    assert respuesta.wants_tools
    assert respuesta.stop_reason == "tool_use"
    assert respuesta.tool_calls[0].params == {"path": "docs/PLAN.md"}
    # raw_content debe permitir reconstruir el turno en la iteración siguiente
    assert respuesta.raw_content[0]["type"] == "tool_use"


def test_finish_reason_stop_con_tool_calls_se_corrige():
    """Varios servidores locales mandan 'stop' aunque haya llamadas pendientes."""
    cliente = responder(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"id": "t1", "function": {"name": "eco", "arguments": "{}"}}
                        ],
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )
    respuesta = LocalProvider("m", client=cliente).complete([{"role": "user", "content": "x"}])
    assert respuesta.stop_reason == "tool_use"


def test_la_peticion_lleva_stream_false_y_las_tools_traducidas():
    capturadas: list = []
    cliente = responder(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}, capturadas
    )
    LocalProvider("qwen3:4b", client=cliente).complete(
        [{"role": "user", "content": "hola"}],
        system="SYS",
        tools=[{"name": "eco", "description": "d", "input_schema": {"type": "object"}}],
    )

    peticion = capturadas[0]
    assert peticion["stream"] is False
    assert peticion["model"] == "qwen3:4b"
    assert peticion["tools"][0]["function"]["name"] == "eco"
    assert peticion["messages"][0] == {"role": "system", "content": "SYS"}


def test_health_detecta_que_no_hay_servidor():
    def sin_servidor(request):
        raise httpx.ConnectError("connection refused")

    cliente = httpx.Client(transport=httpx.MockTransport(sin_servidor))
    ok, mensaje = LocalProvider("qwen3:4b", client=cliente).health()

    assert not ok
    assert "ollama serve" in mensaje


def test_health_detecta_que_el_modelo_no_esta_cargado():
    cliente = responder({"data": [{"id": "otro-modelo"}]})
    ok, mensaje = LocalProvider("qwen3:4b", client=cliente).health()

    assert not ok
    assert "qwen3:4b" in mensaje and "otro-modelo" in mensaje


def test_health_ok():
    cliente = responder({"data": [{"id": "qwen3:4b"}]})
    ok, mensaje = LocalProvider("qwen3:4b", client=cliente).health()
    assert ok and "OK" in mensaje


# ──────────────────────── integración con el router ───────────────────────────


def test_el_router_construye_el_proveedor_local_solo(monkeypatch):
    from argos.config import Settings
    from argos.models.base import TaskKind
    from argos.models.router import ModelRouter

    monkeypatch.setenv("ARGOS_MODELS__LOCAL_ONLY", "true")
    monkeypatch.setenv("ARGOS_MODELS__LOCAL_MODEL", "qwen3:4b")

    router = ModelRouter.from_settings(settings=Settings())

    for kind in TaskKind:
        assert router.provider_for(kind).model == "qwen3:4b"


def test_en_modo_local_no_hace_falta_api_key(monkeypatch):
    """El objetivo de trabajar en local: arrancar sin credenciales."""
    from argos.config import Settings
    from argos.models.router import ModelRouter

    monkeypatch.setenv("ARGOS_MODELS__LOCAL_ONLY", "true")
    settings = Settings()
    assert settings.anthropic_api_key is None

    ModelRouter.from_settings(settings=settings)  # no debe lanzar


def test_la_peticion_lleva_num_ctx_para_ollama():
    """El default de Ollama (4096) lo agota un agente de inmediato.

    Al desbordar, el servidor trunca por el principio y el modelo pierde su
    prompt de sistema sin emitir ningún error: el fallo se manifiesta como un
    agente que "se vuelve tonto" a mitad de la tarea.
    """
    capturadas: list = []
    cliente = responder(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}, capturadas
    )
    LocalProvider("qwen3:4b", num_ctx=16384, client=cliente).complete(
        [{"role": "user", "content": "hola"}]
    )
    assert capturadas[0]["options"]["num_ctx"] == 16384


def test_num_ctx_none_no_manda_options():
    """llama.cpp la ignora, pero no ensuciamos la petición si se desactiva."""
    capturadas: list = []
    cliente = responder(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}, capturadas
    )
    LocalProvider("m", num_ctx=None, client=cliente).complete([{"role": "user", "content": "x"}])
    assert "options" not in capturadas[0]


def test_captura_el_campo_reasoning_de_ollama():
    """Ignorarlo hace que un turno que sólo razonó parezca terminado con éxito."""
    cliente = responder(
        {
            "choices": [
                {
                    "message": {"content": "", "reasoning": "Debería usar glob..."},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    r = LocalProvider("qwen3:8b", client=cliente).complete([{"role": "user", "content": "x"}])

    assert r.reasoning.startswith("Debería usar glob")
    assert r.is_empty, "razonar sin producir nada es un turno degenerado, no un éxito"

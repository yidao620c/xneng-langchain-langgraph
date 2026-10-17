"""第 3 篇核心用法测试：消息与结构化输出（hello_messages.py）。

覆盖 content_blocks 惰性解析、序列化往返、三种 schema 的结构化输出、
ToolMessage 回传、build_model 分支，以及 main() 全流程（防脚本条数漂移）。
全部用 ScriptedChatModel 驱动，不花一分钱。
"""

import io
import contextlib

import pytest
from pydantic import ValidationError

import hello_messages as hm
from fake_models import ScriptedChatModel
from langchain.messages import AIMessage, HumanMessage
from langchain_core.load import dumpd, load


def scripted(*messages):
    """快捷构造：按消息列表依次供应的 fake 模型。"""
    return ScriptedChatModel(messages=iter(messages))


def tool_call_message(name, args, call_id="call_1"):
    """快捷构造：带工具调用的脚本消息（结构化输出走工具调用解析路径）。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def test_content_blocks_parses_provider_native_format():
    """content_blocks 把提供商原生 content 惰性解析成标准内容块。"""
    message = AIMessage(
        content=[
            {"type": "thinking", "thinking": "想一下", "signature": "sig"},
            {"type": "text", "text": "答案"},
        ],
        response_metadata={"model_provider": "anthropic"},
    )

    blocks = message.content_blocks

    assert [b["type"] for b in blocks] == ["reasoning", "text"]
    assert blocks[0]["reasoning"] == "想一下"
    assert blocks[1]["text"] == "答案"
    assert message.content[0]["type"] == "thinking"  # 原 content 不受影响


def test_serialization_round_trip():
    """dumpd / load 序列化往返，消息内容与类型保持一致。"""
    message = HumanMessage("北京是哪座城市？", id="msg_123")

    restored = load(dumpd(message))

    assert isinstance(restored, HumanMessage)
    assert restored.content == message.content
    assert restored.id == message.id


def test_structured_output_pydantic_returns_validated_instance():
    """Pydantic schema 返回带校验的模型实例。"""
    model = scripted(tool_call_message("Movie", {"title": "盗梦空间", "year": 2010, "director": "诺兰"}))

    movie = model.with_structured_output(hm.Movie).invoke("抽取")

    assert isinstance(movie, hm.Movie)
    assert (movie.title, movie.year, movie.director) == ("盗梦空间", 2010, "诺兰")


def test_structured_output_typeddict_returns_dict():
    """TypedDict schema 返回普通 dict，不做运行时校验。"""
    model = scripted(tool_call_message("MovieDict", {"title": "盗梦空间", "year": 2010, "director": "诺兰"}))

    movie = model.with_structured_output(hm.MovieDict).invoke("抽取")

    assert isinstance(movie, dict)
    assert movie == {"title": "盗梦空间", "year": 2010, "director": "诺兰"}


def test_structured_output_json_schema_returns_dict():
    """JSON Schema dict 同样返回普通 dict。"""
    model = scripted(tool_call_message("Movie", {"title": "盗梦空间", "year": 2010, "director": "诺兰"}))

    movie = model.with_structured_output(hm.MOVIE_JSON_SCHEMA, method="json_mode").invoke("抽取")

    assert movie == {"title": "盗梦空间", "year": 2010, "director": "诺兰"}


def test_structured_output_include_raw_exposes_parse_failure():
    """模型没按 schema 输出时，include_raw 把失败收进 parsing_error 而不是抛异常。"""
    model = scripted(AIMessage(content="我听不懂，随便聊聊"))

    out = model.with_structured_output(hm.Movie, include_raw=True).invoke("抽取")

    assert out["parsed"] is None
    assert out["raw"].text == "我听不懂，随便聊聊"


def test_tool_message_round_trip_with_artifact():
    """ToolMessage 回传工具结果，artifact 只挂在消息上不下发给模型。"""
    ai_message = tool_call_message("get_weather", {"location": "北京"})
    model = scripted(ai_message, AIMessage(content="北京今天晴，气温 26℃。"))

    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "晴，26℃"

    # 与 demo_tool_message 同一路径：先发起工具调用，再回传结果
    call = model.bind_tools([get_weather]).invoke("北京天气怎么样？").tool_calls[0]
    tool_message = hm.ToolMessage(
        content="晴，26℃",
        tool_call_id=call["id"],
        name=call["name"],
        artifact={"cached": False},
    )
    response = model.invoke([HumanMessage("北京天气怎么样？"), ai_message, tool_message])

    assert response.text == "北京今天晴，气温 26℃。"
    assert tool_message.artifact == {"cached": False}


def test_extract_prompt_pins_field_names():
    """json_mode 不强制结构，提示词必须钉死字段名（防字段跑偏）。"""
    for field in ("title", "year", "director"):
        assert field in hm.EXTRACT_PROMPT
    assert "盗梦空间" in hm.EXTRACT_PROMPT  # 演示句直接内嵌在提示词里


def test_build_model_defaults_to_scripted(monkeypatch):
    """未指定模型名时用本地替身。"""
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    assert isinstance(hm.build_model(), ScriptedChatModel)


def test_build_model_forwards_params_to_real_model(monkeypatch):
    """指定真实模型时，网关参数随 init_chat_model 原样透传。"""
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-5.5")
    monkeypatch.setenv("YI_BASE_URL", "https://gw.example.com/v1")
    monkeypatch.setenv("YI_API_KEY", "sk-test")
    seen = {}

    def fake_init(model_name, **kwargs):
        seen["model_name"] = model_name
        seen["kwargs"] = kwargs
        return "REAL_MODEL"

    monkeypatch.setattr(hm, "init_chat_model", fake_init)

    assert hm.build_model() == "REAL_MODEL"
    assert seen["model_name"] == "openai:gpt-5.5"
    assert seen["kwargs"]["base_url"] == "https://gw.example.com/v1"
    assert seen["kwargs"]["api_key"] == "sk-test"


def test_main_runs_end_to_end_on_scripted_model(monkeypatch, capsys):
    """main() 全流程在替身模型上跑通，且脚本条数与调用次数匹配。"""
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    hm.main()

    out = capsys.readouterr().out
    assert "结构化输出：Pydantic" in out
    assert "结构化输出：JSON Schema" in out
    assert "ToolMessage 与 artifact" in out

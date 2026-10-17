"""第 2 篇示例测试：模型抽象的五种调用方式。

全部用 ScriptedChatModel 驱动，不依赖任何 API key。
fake 模型按脚本顺序供应消息，每个用例自备足量脚本。
"""

from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

import hello_models
from fake_models import ScriptedChatModel


def scripted(*contents):
    """快捷构造：按内容列表依次供应消息的 fake 模型。"""
    return ScriptedChatModel(
        messages=iter([AIMessage(content=c) for c in contents])
    )


def test_invoke_with_string():
    """invoke 传字符串，返回 AIMessage，内容取脚本下一条。"""
    model = scripted("北京今天晴天，气温 26℃。")

    response = model.invoke("北京天气怎么样？")

    assert isinstance(response, AIMessage)
    assert response.text == "北京今天晴天，气温 26℃。"


def test_invoke_with_message_history():
    """invoke 传消息列表：多轮历史进入调用，返回脚本下一条。"""
    model = scripted("北京位于华北平原北部。")
    conversation = [
        SystemMessage("你是一个简洁的地理助手，回答不超过两句话。"),
        HumanMessage("北京是哪座城市？"),
        AIMessage("北京是中华人民共和国的首都。"),
        HumanMessage("它在哪个平原？"),
    ]

    response = model.invoke(conversation)

    assert isinstance(response, AIMessage)
    assert response.text == "北京位于华北平原北部。"


def test_stream_accumulates_chunks():
    """stream 逐块产出 AIMessageChunk，加法聚合成完整消息。"""
    model = scripted("北京 今天 晴天")  # fake 按空白切片，一条消息多个 chunk

    full = None  # None | AIMessageChunk
    n_chunks = 0
    for chunk in model.stream("演示流式输出"):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
        n_chunks += 1

    assert n_chunks > 1  # 多 chunk 才验证得了累积逻辑
    assert full.text == "北京 今天 晴天"


def test_batch_returns_in_order():
    """batch 批量调用，返回顺序与输入顺序一致。"""
    model = scripted("回答一", "回答二")

    responses = model.batch(["问题一", "问题二"])

    assert [r.text for r in responses] == ["回答一", "回答二"]


def test_build_model_uses_scripted_without_env(monkeypatch):
    """未设置 AGENT_MODEL 时用本地替身，无需 API key。"""
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    assert isinstance(hello_models.build_model(), ScriptedChatModel)


def test_build_model_prefers_env(monkeypatch):
    """设置了 AGENT_MODEL 时走 init_chat_model，模型名原样透传。"""
    monkeypatch.setenv("AGENT_MODEL", "deepseek-v4.1-flash")
    monkeypatch.delenv("YI_BASE_URL", raising=False)
    monkeypatch.delenv("YI_API_KEY", raising=False)
    seen = {}

    def fake_init(model_name, **kwargs):
        seen["model_name"] = model_name
        seen["kwargs"] = kwargs
        return "REAL_MODEL"

    monkeypatch.setattr(hello_models, "init_chat_model", fake_init)

    assert hello_models.build_model() == "REAL_MODEL"
    assert seen["model_name"] == "deepseek-v4.1-flash"
    assert seen["kwargs"] == {}


def test_build_model_passes_gateway_config(monkeypatch):
    """约定了 YI_* 网关变量时，base_url 与 api_key 随模型透传。"""
    monkeypatch.setenv("AGENT_MODEL", "openai:deepseek-v4.1-flash")
    monkeypatch.setenv("YI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("YI_API_KEY", "test-key")
    seen = {}

    def fake_init(model_name, **kwargs):
        seen["model_name"] = model_name
        seen["kwargs"] = kwargs
        return "REAL_MODEL"

    monkeypatch.setattr(hello_models, "init_chat_model", fake_init)

    assert hello_models.build_model() == "REAL_MODEL"
    assert seen["kwargs"] == {
        "base_url": "https://api.example.com/v1",
        "api_key": "test-key",
    }

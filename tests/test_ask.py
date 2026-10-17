"""第 2 篇完整案例测试：模型试用台 ask()。

覆盖 ask() 的返回值结构、on_token 回调和 build_model() 的分支。
"""

import ask as ask_mod
from ask import ask, build_model
from fake_models import ScriptedChatModel
from langchain.messages import AIMessage


def scripted(*contents):
    """快捷构造：按内容列表依次供应消息的 fake 模型。"""
    return ScriptedChatModel(
        messages=iter([AIMessage(content=c) for c in contents])
    )


def test_ask_returns_text_and_stats():
    """ask() 聚合流式 chunk，返回文本、chunk 数、耗时与用量。"""
    model = scripted("北京 今天 晴天，气温 26℃。")

    result = ask(model, "北京天气怎么样？")

    assert result["text"] == "北京 今天 晴天，气温 26℃。"
    assert result["chunks"] > 1  # fake 按空白切片，多 chunk 验证累积
    assert result["elapsed_s"] >= 0
    assert result["usage"] is None  # fake 不产用量，真实模型才有


def test_ask_on_token_receives_all_text():
    """on_token 回调逐块收到文本，拼接后等于完整回答。"""
    model = scripted("你好 ，世界")

    seen = []
    result = ask(model, "打招呼", on_token=seen.append)

    assert "".join(seen) == result["text"]
    assert seen  # 至少收到一个 chunk


def test_build_model_defaults_to_scripted(monkeypatch):
    """未指定模型名时用本地替身，temperature 等参数被安全吸收。"""
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    model = build_model(temperature=0.7, max_tokens=256)

    assert isinstance(model, ScriptedChatModel)


def test_build_model_forwards_params_to_real_model(monkeypatch):
    """指定真实模型时，参数随 init_chat_model 原样透传。"""
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-5.5")
    monkeypatch.delenv("YI_BASE_URL", raising=False)
    monkeypatch.delenv("YI_API_KEY", raising=False)
    seen = {}

    def fake_init(model_name, **kwargs):
        seen["model_name"] = model_name
        seen["kwargs"] = kwargs
        return "REAL_MODEL"

    monkeypatch.setattr(ask_mod, "init_chat_model", fake_init)

    assert build_model(temperature=0.2, max_tokens=100) == "REAL_MODEL"
    assert seen["model_name"] == "openai:gpt-5.5"
    assert seen["kwargs"] == {"temperature": 0.2, "max_tokens": 100}

"""第 4 篇完整案例测试：工具箱（toolbox.py）。

全部用 ScriptedChatModel 驱动，不依赖任何 API key。
"""

import pytest
from langchain.messages import AIMessage
from langchain.tools import ToolException

import toolbox
from fake_models import ScriptedChatModel
from toolbox import ToolBox, build_agent


def scripted(*messages):
    """快捷构造：按消息列表依次供应的 fake 模型。"""
    return ScriptedChatModel(messages=iter(list(messages)))


def make_box() -> ToolBox:
    """标准工具箱：天气、计算、换算、时间。"""
    box = ToolBox()
    box.register(toolbox.get_weather, toolbox.calculate_tool, toolbox.convert_tool, toolbox.now_time)
    return box


def test_toolbox_register_and_table():
    """注册去重、一览表含全部工具、下线生效。"""
    box = make_box()
    names = {t.name for t in box.tools}

    assert names == {"get_weather", "calculate", "convert_unit", "now_time"}
    table = box.table()
    for name in names:
        assert name in table

    with pytest.raises(ValueError):
        box.register(toolbox.get_weather)  # 重名直接报错

    box.unregister("now_time")
    assert box.get("now_time") is None


def test_toolbox_call_direct():
    """直调工具返回内容文本；查无此工具给明确报错。"""
    box = make_box()

    result = box.call("calculate", {"expression": "(3 + 4) * 2"})
    assert result == "(3 + 4) * 2 = 14"

    with pytest.raises(ToolException) as exc_info:
        box.call("no_such_tool", {})
    assert "no_such_tool" in str(exc_info.value)
    assert "get_weather" in str(exc_info.value)  # 报错里列出可用工具


def test_calculate_safe_eval():
    """计算器只放行四则运算，别的表达式走兜底文案而不是炸循环。"""
    assert toolbox.calculate("2 ** 10") == "2 ** 10 = 1024"

    bounded = toolbox.calculate_tool.invoke({"expression": "import os"})
    assert "算不了" in bounded
    bounded2 = toolbox.calculate_tool.invoke({"expression": "3 +"})
    assert "算不了" in bounded2


def test_convert_unit():
    """长度与温度换算正确，不认识的单位走兜底文案。"""
    assert toolbox.convert_unit(42.195, "km", "m") == "42.195 km = 42195.0 m"
    assert toolbox.convert_unit(100, "c", "f") == "100 c = 212.0 f"

    bounded = toolbox.convert_tool.invoke({"value": 1, "from_unit": "kg", "to_unit": "m"})
    assert "单位不认识" in bounded


def test_agent_end_to_end():
    """工具箱交给 create_agent，模型自己挑工具并给出最终回答。"""
    box = ToolBox()
    box.register(toolbox.get_weather, toolbox.calculate_tool)
    model = scripted(
        AIMessage(content="", tool_calls=[
            {"name": "get_weather", "args": {"city": "北京"}, "id": "call_1", "type": "tool_call"}
        ]),
        AIMessage(content="北京今天晴，气温 26℃，适合出行。"),
    )

    agent = build_agent(model, box)
    result = agent.invoke({"messages": [{"role": "user", "content": "北京天气怎么样？"}]})

    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].name == "get_weather"
    assert "北京今天晴" in result["messages"][-1].content


def test_build_model_uses_scripted_without_env(monkeypatch):
    """未设置 AGENT_MODEL 时用本地替身。"""
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    assert isinstance(toolbox.build_model(), ScriptedChatModel)


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

    monkeypatch.setattr(toolbox, "init_chat_model", fake_init)

    assert toolbox.build_model() == "REAL_MODEL"
    assert seen["kwargs"] == {
        "base_url": "https://api.example.com/v1",
        "api_key": "test-key",
    }

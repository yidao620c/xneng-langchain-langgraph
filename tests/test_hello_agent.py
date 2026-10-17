"""第 1 篇示例测试：Agent 的工具调用循环。"""

from hello_agent import build_agent, get_weather


def test_get_weather():
    """工具函数返回包含城市名与天气描述的字符串。"""
    assert get_weather("北京") == "北京今天晴天，气温 26℃，适合出行。"


def test_agent_runs_full_tool_call_loop(monkeypatch):
    """无 AGENT_MODEL 时，Agent 用脚本化模型走完整循环：

    模型请求调用工具 -> 工具执行 -> 模型生成最终回答。
    """
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": "北京天气怎么样？"}]})

    # 工具执行结果进入消息历史
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    assert "晴天" in tool_messages[0].content

    # 最后一条是模型读完工具结果后的回答
    assert result["messages"][-1].type == "ai"
    assert "适合出行" in result["messages"][-1].content

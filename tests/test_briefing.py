"""第 5 篇完整案例测试：值守简报员（briefing.py）。

全部用 ScriptedChatModel 驱动，不依赖任何 API key。
"""

from langchain.messages import AIMessage, HumanMessage

import briefing
from briefing import Brief, build_agent, build_model
from fake_models import ScriptedChatModel


def scripted(*messages):
    return ScriptedChatModel(messages=iter(list(messages)))


def brief_call(cid="call_2"):
    """模型提交结构化简报的消息。"""
    return AIMessage(
        content="简报写好了",
        tool_calls=[{
            "name": "Brief",
            "args": {
                "title": "北京天气简报",
                "points": ["北京今天晴，气温 26℃"],
                "confidence": "high",
            },
            "id": cid, "type": "tool_call",
        }],
    )


def test_brief_model_fields():
    """Brief 结构三字段齐备，confidence 有描述。"""
    b = Brief(title="t", points=["a", "b"], confidence="high")
    assert b.title == "t"
    assert b.points == ["a", "b"]
    assert b.confidence == "high"


def test_build_model_fake_and_named():
    """无 AGENT_MODEL 时返回替身；指定模型名时走 init_chat_model。"""
    assert isinstance(build_model(), ScriptedChatModel)

    import os

    monkey_env = dict(os.environ)
    try:
        # 用一个不存在的提供商名，只为确认走了 init_chat_model 分支
        os.environ["AGENT_MODEL"] = "openai:whatever"
        model = build_model()
        assert not isinstance(model, ScriptedChatModel)
    finally:
        os.environ.clear()
        os.environ.update(monkey_env)


def test_agent_full_loop_produces_brief():
    """完整链路：查天气 → 审计日志 → 结构化简报。"""
    agent = build_agent(scripted(
        AIMessage(
            content="好的",
            tool_calls=[{
                "name": "get_weather", "args": {"city": "北京"},
                "id": "call_1", "type": "tool_call",
            }],
        ),
        brief_call(),
    ))
    result = agent.invoke({"messages": [HumanMessage("北京适合跑步吗？")]})

    brief = result["structured_response"]
    assert isinstance(brief, Brief)
    assert brief.title == "北京天气简报"
    assert any("26℃" in p for p in brief.points)

    # 审计日志由 NoteKeeper 中间件写入，reducer 合并
    assert len(result["audit_log"]) == 1
    assert "get_weather" in result["audit_log"][0]


def test_agent_limit_stops_runaway_loop():
    """模型陷入循环时，ModelCallLimitMiddleware 限 4 轮强制收工。"""
    script = [
        AIMessage(
            content="继续查",
            tool_calls=[{
                "name": "get_weather", "args": {"city": f"城市{i}"},
                "id": f"c{i}", "type": "tool_call",
            }],
        )
        for i in range(8)
    ]
    agent = build_agent(scripted(*script))
    result = agent.invoke({"messages": [HumanMessage("到处查天气")]})

    tool_msgs = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_msgs) == 4
    assert result.get("structured_response") is None  # 被截断，没来得及交简报


def test_agent_without_brief_falls_back_to_text():
    """brief=False 降级时没有 structured_response，产出是文本回答。"""
    agent = build_agent(scripted(
        AIMessage(
            content="好的",
            tool_calls=[{
                "name": "get_weather", "args": {"city": "北京"},
                "id": "call_1", "type": "tool_call",
            }],
        ),
        AIMessage(content="北京今天晴，适合跑步。"),
    ), brief=False)
    result = agent.invoke({"messages": [HumanMessage("北京适合跑步吗？")]})

    assert "structured_response" not in result or result.get("structured_response") is None
    assert result["messages"][-1].content == "北京今天晴，适合跑步。"


def test_parallel_tool_calls_merge_audit_log():
    """一轮并行调两个工具，audit_log 两条记录都保留（operator.add）。"""
    agent = build_agent(scripted(
        AIMessage(
            content="同时查",
            tool_calls=[
                {"name": "get_weather", "args": {"city": "北京"},
                 "id": "c1", "type": "tool_call"},
                {"name": "calculate", "args": {"expression": "128 * 4"},
                 "id": "c2", "type": "tool_call"},
            ],
        ),
        brief_call("call_3"),
    ))
    result = agent.invoke({"messages": [HumanMessage("北京天气和 128 乘 4")]})

    assert len(result["audit_log"]) == 2
    assert any("get_weather" in n for n in result["audit_log"])
    assert any("calculate" in n for n in result["audit_log"])
    assert isinstance(result["structured_response"], Brief)


def test_stream_mode_collects_brief_and_audit():
    """--stream 路径：updates 流里能收齐结构化简报与审计日志。"""
    agent = build_agent(scripted(
        AIMessage(
            content="好的",
            tool_calls=[{
                "name": "get_weather", "args": {"city": "北京"},
                "id": "call_1", "type": "tool_call",
            }],
        ),
        brief_call(),
    ))
    brief, audit = None, []
    for chunk in agent.stream(
        {"messages": [HumanMessage("北京适合跑步吗？")]}, stream_mode="updates",
    ):
        for node, update in chunk.items():
            update = update or {}
            if update.get("structured_response") is not None:
                brief = update["structured_response"]
            audit.extend(update.get("audit_log", []))

    assert isinstance(brief, Brief)
    assert len(audit) == 1

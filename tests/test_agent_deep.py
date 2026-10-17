"""第 5 篇示例测试：Agent 深入（agent_deep.py）。

全部用 ScriptedChatModel 驱动，不依赖任何 API key。
"""

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    after_model,
    before_model,
    wrap_tool_call,
)
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool

import agent_deep
from agent_deep import AgentNotes, NoteTaking
from agent_deep import (
    audit,
    fake_model,
    get_weather,
    round_aware_prompt,
    sensitive_guard,
    tool_call,
)
from fake_models import ScriptedChatModel


def scripted(*messages):
    """快捷构造：按消息列表依次供应的 fake 模型。"""
    return ScriptedChatModel(messages=iter(list(messages)))


def run(agent, text="北京天气"):
    return agent.invoke({"messages": [HumanMessage(text)]})


# ---------- 参数与结构 ----------

def test_create_agent_signature_params():
    """create_agent 有全参数签名，middleware 默认空元组。"""
    import inspect

    params = inspect.signature(create_agent).parameters
    for name in ("model", "tools", "system_prompt", "middleware", "response_format",
                 "state_schema", "context_schema", "checkpointer", "name"):
        assert name in params
    assert params["middleware"].default == ()


def test_name_param_keeps_behavior():
    """name 只作标识，不改变执行行为。"""
    agent = create_agent(
        model=fake_model(AIMessage(content="北京今天晴。")),
        tools=[get_weather],
        name="weather_assistant",
    )
    result = run(agent)
    assert result["messages"][-1].content == "北京今天晴。"


# ---------- 系统提示 ----------

def test_dynamic_prompt_changes_with_rounds():
    """多轮对话时动态提示按消息数变化，脚本消息按轮供应。"""
    from langgraph.checkpoint.memory import InMemorySaver

    model = scripted(
        tool_call("get_weather", {"city": "北京"}, "c1"),
        AIMessage(content="北京今天晴。"),
        AIMessage(content="上海多云。"),
    )
    agent = create_agent(
        model=model, tools=[get_weather], middleware=[round_aware_prompt],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-dynamic"}}
    agent.invoke({"messages": [HumanMessage("北京天气")]}, config=config)
    result = agent.invoke({"messages": [HumanMessage("上海呢")]}, config=config)
    assert result["messages"][-1].content == "上海多云。"


# ---------- 自定义 state ----------

def test_state_schema_audit_log():
    """自定义 state 字段在钩子中写入，reducer 合并多条记录。"""
    agent = create_agent(
        model=fake_model(
            tool_call("get_weather", {"city": "北京"}, "c2"),
            AIMessage(content="北京今天晴。"),
        ),
        tools=[get_weather],
        middleware=[NoteTaking()],
        state_schema=AgentNotes,
    )
    result = run(agent)
    assert "开工" in result["audit_log"]
    assert any("get_weather" in note for note in result["audit_log"])


# ---------- 钩子时序 ----------

def test_hook_order():
    """一次完整运行中钩子按文档顺序触发。"""
    order = []

    class Recorder(AgentMiddleware):
        def before_agent(self, state, runtime):
            order.append("before_agent")

        def before_model(self, state, runtime):
            order.append("before_model")

        def wrap_tool_call(self, request, handler):
            order.append("wrap_tool_call")
            return handler(request)

        def after_model(self, state, runtime):
            order.append("after_model")

        def after_agent(self, state, runtime):
            order.append("after_agent")

    agent = create_agent(
        model=fake_model(
            tool_call("get_weather", {"city": "上海"}, "c3"),
            AIMessage(content="上海晴。"),
        ),
        tools=[get_weather],
        middleware=[Recorder()],
    )
    run(agent, "上海天气")
    # 两轮模型调用；tools 节点夹在两轮之间，wrap_tool_call 随 tools 节点触发
    assert order == [
        "before_agent",
        "before_model", "after_model",   # 第一轮：发起工具调用
        "wrap_tool_call",                # tools 节点
        "before_model", "after_model",   # 第二轮：最终回答
        "after_agent",
    ]


def test_wrap_tool_call_sees_request_and_result():
    """wrap_tool_call 能读工具名与入参，能拿到执行结果。"""
    seen = {}

    @wrap_tool_call
    def recorder(request, handler):
        seen["name"] = request.tool_call["name"]
        seen["args"] = request.tool_call["args"]
        result = handler(request)
        seen["content"] = result.content
        return result

    agent = create_agent(
        model=fake_model(
            tool_call("get_weather", {"city": "深圳"}, "c4"),
            AIMessage(content="深圳晴。"),
        ),
        tools=[get_weather],
        middleware=[recorder],
    )
    run(agent, "深圳天气")
    assert seen["name"] == "get_weather"
    assert seen["args"] == {"city": "深圳"}
    assert "深圳" in seen["content"]


# ---------- jump_to 拦截 ----------

def test_before_model_jump_to_end_blocks_model():
    """before_model 设 jump_to=end 后模型一轮都不跑。"""
    agent = create_agent(
        model=fake_model(AIMessage(content="不该出现。")),
        tools=[get_weather],
        middleware=[sensitive_guard],
    )
    result = run(agent, "我的密码是多少？")
    assert [type(m).__name__ for m in result["messages"]] == ["HumanMessage"]


def test_guard_passes_normal_question():
    """正常问题不受拦截，正常走完循环。"""
    agent = create_agent(
        model=fake_model(
            tool_call("get_weather", {"city": "北京"}, "c5"),
            AIMessage(content="北京晴。"),
        ),
        tools=[get_weather],
        middleware=[sensitive_guard],
    )
    result = run(agent, "北京天气")
    assert result["messages"][-1].content == "北京晴。"


# ---------- 轮数限制 ----------

def test_model_call_limit_exit_end():
    """thread_limit=3 + exit_behavior=end 时，模型第 4 次调用被掐断。"""
    script = [tool_call("get_weather", {"city": f"城市{i}"}, f"c{i}") for i in range(5)]
    agent = create_agent(
        model=scripted(*script),
        tools=[get_weather],
        middleware=[ModelCallLimitMiddleware(thread_limit=3, exit_behavior="end")],
    )
    result = run(agent, "把每个城市的天气都查一遍")
    tool_msgs = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert len(tool_msgs) == 3


# ---------- 结构化输出 ----------

def test_response_format_tool_strategy():
    """模型调用 schema 工具后，structured_response 是校验过的对象。"""
    from agent_deep import Brief

    agent = create_agent(
        model=fake_model(
            tool_call("get_weather", {"city": "北京"}, "c6"),
            tool_call("Brief", {"summary": "北京晴", "need_umbrella": False}, "c6r"),
        ),
        tools=[get_weather],
        response_format=Brief,
    )
    result = run(agent, "北京需要带伞吗？")
    assert sorted(result.keys()) == ["messages", "structured_response"]
    assert result["structured_response"].summary == "北京晴"
    assert result["structured_response"].need_umbrella is False


def test_response_format_self_correction_on_missing_field():
    """schema 必填字段缺失时校验失败，模型补全后重交成功。

    校验失败会给模型一条 error ToolMessage，模型有机会自我修正，
    这与第 4 篇工具参数校验的机制一脉相承。
    """
    from agent_deep import Brief

    agent = create_agent(
        model=scripted(
            tool_call("Brief", {"summary": "缺字段"}, "c7"),          # 少 need_umbrella
            tool_call("Brief", {"summary": "补全了", "need_umbrella": False}, "c7r"),
        ),
        tools=[],
        response_format=Brief,
    )
    result = run(agent, "给我个简报")
    assert result["structured_response"].summary == "补全了"
    assert result["structured_response"].need_umbrella is False


# ---------- 流式 ----------

def test_stream_updates_emits_each_step():
    """updates 模式按节点流出状态更新：model → tools → model。"""
    agent = create_agent(
        model=fake_model(
            tool_call("get_weather", {"city": "北京"}, "c8"),
            AIMessage(content="北京晴。"),
        ),
        tools=[get_weather],
    )
    nodes = []
    for chunk in agent.stream({"messages": [HumanMessage("北京天气")]}, stream_mode="updates"):
        nodes.extend(node for node in chunk if chunk[node])
    assert nodes[0] == "model"
    assert "tools" in nodes


def test_stream_custom_surfaces_writer_output():
    """custom 模式流出工具内部 get_stream_writer 写的进度。"""
    agent = create_agent(
        model=fake_model(
            tool_call("lookup_city", {"city": "北京"}, "c9"),
            AIMessage(content="北京人口 2100 万。"),
        ),
        tools=[agent_deep.lookup_city],
    )
    customs = list(agent.stream(
        {"messages": [HumanMessage("北京有多少人？")]}, stream_mode="custom"))
    assert customs == ["正在检索 北京 的资料"]


# ---------- after_model 的坑 ----------

def test_after_model_jump_to_is_ignored_in_1_4_3():
    """坑：after_model 里返回 jump_to 在本仓 1.4.3 不生效。

    钩子会触发、值会写进 state，但该节点的条件边路由没有据此跳转，
    模型下一轮照常执行。拦截请用 before_model（见 guard 用例）。
    """

    @after_model(can_jump_to=["end"])
    def try_limit(state, runtime):
        tool_msgs = sum(1 for m in state["messages"] if type(m).__name__ == "ToolMessage")
        if tool_msgs >= 1:
            return {"jump_to": "end"}
        return None

    agent = create_agent(
        model=scripted(
            tool_call("get_weather", {"city": "广州"}, "c10"),
            AIMessage(content="不该出现的第二轮回答。"),
        ),
        tools=[get_weather],
        middleware=[try_limit],
    )
    result = run(agent, "广州天气")
    # 模型照常跑完两轮
    assert result["messages"][-1].content == "不该出现的第二轮回答。"

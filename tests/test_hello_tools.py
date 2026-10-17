"""第 4 篇示例测试：工具定义与调用。

全部用 ScriptedChatModel 驱动，不依赖任何 API key。
fake 模型按脚本顺序供应消息，每个用例自备足量脚本。
"""

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import ToolException, tool
from langchain_core.tools import StructuredTool

import hello_tools
from fake_models import ScriptedChatModel
from hello_tools import add, fetch_page, flaky, get_weather_detail, graph_for_calls


def scripted(*messages):
    """快捷构造：按消息列表依次供应的 fake 模型。"""
    return ScriptedChatModel(messages=iter(list(messages)))


def call_msg(name, args, call_id):
    """快捷构造：一条带工具调用的 AIMessage。"""
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id, "type": "tool_call"}
    ])


def test_tool_schema_inferred():
    """@tool 从类型注解与 docstring 推断名称、描述与入参 schema。"""
    assert add.name == "add"
    assert "两数相加" in add.description
    assert set(add.args) == {"a", "b"}
    assert add.args["a"]["type"] == "integer"


def test_tool_call_invoke_returns_tool_message():
    """传 tool_call 字典调用返回 ToolMessage；传普通参数字典返回原始内容。"""
    msg = add.invoke({"name": "add", "args": {"a": 2, "b": 3}, "id": "call_x", "type": "tool_call"})

    assert isinstance(msg, ToolMessage)
    assert msg.content == "5"
    assert msg.tool_call_id == "call_x"
    assert add.invoke({"a": 2, "b": 3}) == 5


def test_custom_name_and_description():
    """@tool 第一个位置参数改工具名，description 参数覆盖 docstring。"""

    @tool("web_search", description="搜索互联网获取公开信息")
    def search(query: str) -> str:
        """这段会被覆盖。"""
        return f"结果: {query}"

    assert search.name == "web_search"
    assert search.description == "搜索互联网获取公开信息"


def test_args_schema_fields_and_defaults():
    """args_schema 的枚举、默认值、字段描述都进 schema。"""
    args = get_weather_detail.args

    assert set(args) == {"location", "units", "include_forecast"}
    assert args["units"]["enum"] == ["celsius", "fahrenheit"]
    assert args["include_forecast"]["default"] is False


def test_args_schema_rejects_bad_input():
    """入参不符合 schema 时抛校验错误。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        get_weather_detail.invoke({"location": 123})


def test_content_and_artifact():
    """response_format=content_and_artifact：二元组拆成 content 与 artifact。"""
    plain = fetch_page.invoke({"url": "https://example.com"})
    msg = fetch_page.invoke(
        {"name": "fetch_page", "args": {"url": "https://example.com"}, "id": "c1", "type": "tool_call"}
    )

    assert isinstance(plain, str)  # 普通调用只拿 content
    assert isinstance(msg, ToolMessage)
    assert "网页正文" in msg.content
    assert msg.artifact == {"status": 200, "source": "本地替身"}


def test_bind_and_manual_loop():
    """bind_tools 后模型发起调用，ToolMessage 回传，第二轮模型给最终回答。"""
    model = scripted(
        call_msg("get_weather", {"city": "北京"}, "call_1"),
        AIMessage(content="北京今天晴，气温 26℃。"),
    )
    bound = model.bind_tools([hello_tools.get_weather])

    messages = [HumanMessage("北京天气怎么样？")]
    response = bound.invoke(messages)
    messages.append(response)
    call = response.tool_calls[0]
    assert call["name"] == "get_weather"

    tool_msg = hello_tools.get_weather.invoke(
        {"name": call["name"], "args": call["args"], "id": call["id"], "type": "tool_call"}
    )
    messages.append(tool_msg)
    final = bound.invoke(messages)

    assert "北京今天晴" in final.text


def test_validation_error_self_correction():
    """模型传错参数时，create_agent 把校验错误转成 error ToolMessage，模型修正重发。"""
    model = scripted(
        call_msg("add", {"a": 3}, "call_2"),          # 缺 b，校验失败
        call_msg("add", {"a": 3, "b": 4}, "call_3"),  # 看到错误后修正
        AIMessage(content="3 加 4 等于 7。"),
    )
    from langchain.agents import create_agent

    agent = create_agent(model=model, tools=[add])
    result = agent.invoke({"messages": [{"role": "user", "content": "算一下 3 加 4"}]})

    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0].status == "error"
    assert "add" in tool_msgs[0].content
    assert tool_msgs[1].status == "success"
    assert tool_msgs[1].content == "7"
    assert result["messages"][-1].content == "3 加 4 等于 7。"


def test_tool_exception_bare_vs_bounded():
    """裸调用业务异常直接抛；from_function 挂 handle_tool_error 后转 error ToolMessage。"""
    bare = StructuredTool.from_function(hello_tools.flaky_query)

    with pytest.raises(ToolException):
        bare.invoke({"city": "北京"})

    msg = flaky.invoke(
        {"name": "flaky_query", "args": {"city": "北京"}, "id": "c1", "type": "tool_call"}
    )
    assert msg.status == "error"
    assert msg.content == "接口暂不可用，请稍后再试"


def test_tool_runtime_injection():
    """ToolRuntime 读到历史消息数与 tool_call_id，InjectedToolCallId 注解同样生效。"""
    calls = [
        {"name": "ask", "args": {"question": "在吗"}, "id": "c1", "type": "tool_call"},
        {"name": "echo", "args": {"x": 7}, "id": "c2", "type": "tool_call"},
    ]

    def ask(question: str, runtime) -> str:
        """Ask with context."""
        history = len(runtime.state.get("messages", []))
        return f"问题「{question}」，当前历史 {history} 条，tool_call_id={runtime.tool_call_id}"

    from typing_extensions import Annotated
    from langchain.tools import InjectedToolCallId, ToolRuntime

    ask_tool = tool(ask)

    @tool
    def echo(x: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
        """Echo id."""
        return f"x={x}, id={tool_call_id}"

    out = graph_for_calls([ask_tool, echo], calls).invoke(
        {"messages": [HumanMessage("在吗")]}
    )
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]

    assert len(tool_msgs) == 2
    assert "历史 2 条" in tool_msgs[0].content
    assert tool_msgs[0].content.endswith("tool_call_id=c1")
    assert tool_msgs[1].content == "x=7, id=c2"


def test_create_agent_full_loop():
    """create_agent 把绑定、执行、回传、路由包成一个 invoke。"""
    model = scripted(
        call_msg("get_weather", {"city": "北京"}, "call_1"),
        AIMessage(content="北京今天晴，气温 26℃，适合出行。"),
    )
    from langchain.agents import create_agent

    agent = create_agent(
        model=model,
        tools=[hello_tools.get_weather],
        system_prompt="你是一个简洁的天气助手。",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "北京天气怎么样？"}]})

    assert [m.type for m in result["messages"]] == ["human", "ai", "tool", "ai"]
    assert "北京今天晴" in result["messages"][-1].content


def test_build_model_uses_scripted_without_env(monkeypatch):
    """未设置 AGENT_MODEL 时用本地替身。"""
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    assert isinstance(hello_tools.build_model(), ScriptedChatModel)


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

    monkeypatch.setattr(hello_tools, "init_chat_model", fake_init)

    assert hello_tools.build_model() == "REAL_MODEL"
    assert seen["kwargs"] == {
        "base_url": "https://api.example.com/v1",
        "api_key": "test-key",
    }

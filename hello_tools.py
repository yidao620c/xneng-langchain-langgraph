"""第 4 篇示例：工具定义与调用——让模型从聊天变成干活。

围绕 langchain.tools 演示八个核心用法：

1. @tool 基础定义：docstring 变描述，类型注解变入参 schema；
2. 自定义工具名与描述；
3. args_schema：用 Pydantic 定义复杂入参；
4. 三种回报：字符串、content_and_artifact 二元组、return_direct；
5. bind_tools 与手工工具循环：tool_calls 发起，ToolMessage 回传；
6. 错误处理：参数校验默认转 ToolMessage，业务异常要显式兜底；
7. ToolRuntime 与 InjectedToolCallId：工具里读运行时上下文；
8. create_agent：把整个循环交给框架。

模型来源与前三篇相同：设置环境变量 AGENT_MODEL 时使用真实模型；
未设置时用 ScriptedChatModel 在本地演示，不需要任何 API key。

⚠️ fake 模型按脚本顺序供应消息，每次调用消耗一条；脚本条数必须覆盖
main() 里的全部调用次数（当前为 7 条）。
"""

import os

from pydantic import BaseModel, Field
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import InjectedToolCallId, ToolException, ToolRuntime, tool
from langchain_core.tools import StructuredTool
from typing_extensions import Annotated

from fake_models import ScriptedChatModel


def build_model():
    """组装模型：优先使用环境变量配置的真实模型（约定同前三篇）。"""
    model_name = os.environ.get("AGENT_MODEL")
    if model_name:
        kwargs = {}
        if os.environ.get("YI_BASE_URL"):
            kwargs["base_url"] = os.environ["YI_BASE_URL"]
        if os.environ.get("YI_API_KEY"):
            kwargs["api_key"] = os.environ["YI_API_KEY"]
        return init_chat_model(model_name, **kwargs)
    # 本地演示脚本：按 main() 的调用顺序供应 7 条消息
    return ScriptedChatModel(
        messages=iter(
            [
                AIMessage(  # 1 bind_tools·发起天气调用
                    content="",
                    tool_calls=[{
                        "name": "get_weather", "args": {"city": "北京"},
                        "id": "call_1", "type": "tool_call",
                    }],
                ),
                AIMessage(content="北京今天晴，气温 26℃。"),              # 2 bind_tools·回传后回答
                AIMessage(  # 3 错误处理·校验失败调用（缺 b）
                    content="",
                    tool_calls=[{
                        "name": "add", "args": {"a": 3},
                        "id": "call_2", "type": "tool_call",
                    }],
                ),
                AIMessage(  # 4 错误处理·模型修正后重发
                    content="",
                    tool_calls=[{
                        "name": "add", "args": {"a": 3, "b": 4},
                        "id": "call_3", "type": "tool_call",
                    }],
                ),
                AIMessage(content="3 加 4 等于 7。"),                     # 5 错误处理·拿到结果回答
                AIMessage(  # 6 create_agent·发起天气调用
                    content="",
                    tool_calls=[{
                        "name": "get_weather", "args": {"city": "北京"},
                        "id": "call_4", "type": "tool_call",
                    }],
                ),
                AIMessage(content="北京今天晴，气温 26℃，适合出行。"),      # 7 create_agent·最终回答
            ]
        )
    )


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。

    Args:
        city: 城市名
    """
    return f"{city}今天晴天，气温 26℃，适合出行。"


@tool
def add(a: int, b: int) -> int:
    """两数相加。

    Args:
        a: 第一个数
        b: 第二个数
    """
    return a + b


class WeatherInput(BaseModel):
    """天气查询的入参。"""

    location: str = Field(description="城市名或坐标")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius", description="温度单位"
    )
    include_forecast: bool = Field(default=False, description="是否带五天预报")


@tool(args_schema=WeatherInput)
def get_weather_detail(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """查询天气详情，可指定温度单位与预报。"""
    temp = 22 if units == "celsius" else 72
    result = f"{location}当前 {temp} {units}"
    if include_forecast:
        result += "，未来五天晴"
    return result


@tool(response_format="content_and_artifact")
def fetch_page(url: str) -> tuple:
    """抓取网页，返回正文与原始响应头。

    content 给模型读，artifact 给程序用，不下发给模型。
    """
    return (f"网页正文，长度 1024 字节，来自 {url}", {"status": 200, "source": "本地替身"})


@tool(return_direct=True)
def direct_answer(question: str) -> str:
    """直接回答模式，结果原样返回给用户，不再过模型。"""
    return f"这是 {question} 的直接答案"


def flaky_query(city: str) -> str:
    """模拟一个不稳定的外部接口，总是失败。"""
    raise ToolException("连接超时")


# @tool 装饰器不透传 handle_tool_error，工具级错误兜底走 from_function
flaky = StructuredTool.from_function(
    flaky_query, handle_tool_error="接口暂不可用，请稍后再试"
)


def demo_basic_tool():
    """用法 1：@tool 把普通函数变成工具，docstring 与类型注解各就各位。"""
    print("== @tool 基础 ==")
    print(f"name: {add.name}")
    print(f"description: {add.description!r}")
    print(f"args schema: {add.args}")

    # 传 tool_call 字典调用，返回 ToolMessage；传普通参数字典，返回原始内容
    msg = add.invoke({"name": "add", "args": {"a": 2, "b": 3}, "id": "call_x", "type": "tool_call"})
    print(f"tool_call 入参 -> {type(msg).__name__} {msg.content!r} id={msg.tool_call_id}")
    print(f"普通入参 -> {add.invoke({'a': 2, 'b': 3})!r}")
    print()


def demo_custom_name_description():
    """用法 2：函数名不合适时，自定义工具名与描述。"""
    print("== 自定义名称与描述 ==")

    @tool("web_search", description="搜索互联网获取公开信息，结果含标题与链接")
    def search(query: str) -> str:
        """这段 docstring 会被 description 覆盖。"""
        return f"结果: {query}"

    print(f"name: {search.name}")
    print(f"description: {search.description!r}")
    print()


def demo_args_schema():
    """用法 3：args_schema 用 Pydantic 管复杂入参，枚举与默认值都进 schema。"""
    print("== args_schema 复杂入参 ==")
    print(f"schema: {get_weather_detail.args}")
    result = get_weather_detail.invoke({"location": "北京", "include_forecast": True})
    print(f"调用 -> {result!r}")
    print()


def demo_return_values():
    """用法 4：三种回报，字符串、content_and_artifact、return_direct。"""
    print("== 三种回报 ==")
    msg = fetch_page.invoke(
        {"name": "fetch_page", "args": {"url": "https://example.com"}, "id": "call_y", "type": "tool_call"}
    )
    print(f"content_and_artifact -> content={msg.content!r}")
    print(f"artifact（不下发给模型）-> {msg.artifact}")
    print(f"return_direct: {direct_answer.return_direct}")
    print()


def demo_bind_and_loop(model):
    """用法 5：bind_tools 绑定工具，模型发起调用，ToolMessage 回传，手工循环。"""
    print("== bind_tools 与工具循环 ==")
    bound = model.bind_tools([get_weather])
    messages = [HumanMessage("北京天气怎么样？")]
    response = bound.invoke(messages)
    messages.append(response)
    call = response.tool_calls[0]
    print(f"模型发起调用: {call['name']}({call['args']})")

    tool_msg = get_weather.invoke(
        {"name": call["name"], "args": call["args"], "id": call["id"], "type": "tool_call"}
    )
    messages.append(tool_msg)
    print(f"工具结果回传: {tool_msg.content!r}")

    final = bound.invoke(messages)
    print(f"最终回答: {final.text}")
    print()


def demo_error_handling(model):
    """用法 6：错误处理。参数校验默认转 ToolMessage，模型能自我修正；业务异常要显式兜底。"""
    print("== 错误处理 ==")
    from langchain.agents import create_agent

    # 业务异常：裸调用直接炸
    try:
        StructuredTool.from_function(flaky_query).invoke({"city": "北京"})
    except ToolException as e:
        print(f"裸调用业务异常 -> {type(e).__name__}: {e}")

    # 工具级兜底：from_function 挂 handle_tool_error，异常就地转 error ToolMessage
    flaky_msg = flaky.invoke(
        {"name": "flaky_query", "args": {"city": "北京"}, "id": "call_5", "type": "tool_call"}
    )
    print(f"工具级兜底 -> status={flaky_msg.status}, content={flaky_msg.content!r}")

    # 参数校验：create_agent 内部的 ToolNode 默认把校验错误转成 error ToolMessage，
    # 模型读到错误后修正参数重发。脚本第 3、4、5 条就是这条自我修正链
    agent = create_agent(model=model, tools=[add])
    result = agent.invoke({"messages": [{"role": "user", "content": "算一下 3 加 4"}]})
    for m in result["messages"]:
        if m.type == "tool":
            print(f"工具消息 -> status={m.status}, content={m.content[:40]!r}")
    print(f"最终回答: {result['messages'][-1].content}")
    print()


def graph_for_calls(tools, calls):
    """搭一个两节点小图（假模型 + ToolNode），让指定的一批 tool_calls 跑起来。

    ToolNode 是图节点，脱离图直接 invoke 缺运行时配置，这里兜一层。
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import MessagesState
    from langgraph.prebuilt import ToolNode

    def fake_model(state: MessagesState):
        last = state["messages"][-1]
        if getattr(last, "type", None) == "tool":
            return {"messages": [AIMessage(content="收到，上下文都拿到了")]}
        return {"messages": [AIMessage(content="", tool_calls=calls)]}

    def route(state: MessagesState):
        return "tools" if state["messages"][-1].tool_calls else END

    g = StateGraph(MessagesState)
    g.add_node("model", fake_model)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "model")
    g.add_conditional_edges("model", route)
    g.add_edge("tools", "model")
    return g.compile()


def demo_tool_runtime():
    """用法 7：ToolRuntime 与 InjectedToolCallId，工具里读运行时上下文。"""
    print("== ToolRuntime 与 InjectedToolCallId ==")

    @tool
    def ask(question: str, runtime: ToolRuntime) -> str:
        """回答问题时带上上下文信息。"""
        history = len(runtime.state.get("messages", []))
        return f"问题「{question}」，当前历史 {history} 条，tool_call_id={runtime.tool_call_id}"

    @tool
    def echo(x: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
        """回显工具调用 id。"""
        return f"x={x}, id={tool_call_id}"

    calls = [
        {"name": "ask", "args": {"question": "在吗"}, "id": "c1", "type": "tool_call"},
        {"name": "echo", "args": {"x": 7}, "id": "c2", "type": "tool_call"},
    ]
    out = graph_for_calls([ask, echo], calls).invoke({"messages": [HumanMessage("在吗")]})
    for m in out["messages"]:
        if m.type == "tool":
            print(f"  {m.content}")
    print()


def demo_create_agent(model):
    """用法 8：create_agent 把 bind、执行、回传、路由整个循环包起来。"""
    print("== create_agent 完整循环 ==")
    from langchain.agents import create_agent

    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="你是一个简洁的天气助手，回答不超过两句话。",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "北京天气怎么样？"}]})
    print(f"消息序列: {[m.type for m in result['messages']]}")
    print(f"最终回答: {result['messages'][-1].content}")
    print()


def main() -> None:
    model = build_model()
    print(f"当前模型: {type(model).__name__}")
    print(f"模型名: {getattr(model, 'model_name', None) or 'ScriptedChatModel（本地替身）'}")
    print()
    demo_basic_tool()
    demo_custom_name_description()
    demo_args_schema()
    demo_return_values()
    demo_bind_and_loop(model)
    demo_error_handling(model)
    demo_tool_runtime()
    demo_create_agent(model)


if __name__ == "__main__":
    main()

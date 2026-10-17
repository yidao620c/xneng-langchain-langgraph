"""第 5 篇示例：Agent 深入（agent_deep.py）。

create_agent 是一个高度可配置的「线束」（harness）。本篇把它的参数、
中间件钩子、流式执行逐个过一遍。九个 demo，每个独立成函数，互不共享
模型脚本，跑哪个都行。

模型来源与前三篇一致：环境变量 AGENT_MODEL 指定时用真实模型；未指定时
各 demo 用 ScriptedChatModel 本地演示。
"""

import inspect
import operator
import os
from typing import Annotated

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
    AgentMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langgraph.config import get_stream_writer
from pydantic import BaseModel

from fake_models import ScriptedChatModel


# ---------- 公共工具与脚本模型 ----------

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。

    Args:
        city: 城市名
    """
    return f"{city}今天晴，气温 26℃。"


@tool
def lookup_city(city: str) -> str:
    """查询城市资料，过程中流出进度。"""
    writer = get_stream_writer()
    writer(f"正在检索 {city} 的资料")
    return f"{city}：人口 2100 万，今天的天气晴。"


def tool_call(name, args, cid, content="好的"):
    """构造一条带工具调用的模型消息。"""
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": args, "id": cid, "type": "tool_call"}],
    )


def fake_model(*steps):
    """按给定步骤顺序供消息的替身模型。"""
    return ScriptedChatModel(messages=iter(list(steps)))


def build_model(model_name=None):
    """有 AGENT_MODEL 就用真实模型，否则返回 None 交由各 demo 用替身。"""
    name = model_name or os.environ.get("AGENT_MODEL")
    if not name:
        return None
    params = {}
    if os.environ.get("YI_BASE_URL"):
        params["base_url"] = os.environ["YI_BASE_URL"]
    if os.environ.get("YI_API_KEY"):
        params["api_key"] = os.environ["YI_API_KEY"]
    return init_chat_model(name, **params)


# ---------- demo 1：全参数一览 ----------

def demo_params() -> None:
    """打印 create_agent 的全部参数，再建一个带 name 的最小 agent。"""
    print("== create_agent 参数 ==")
    for name, param in inspect.signature(create_agent).parameters.items():
        default = "" if param.default is inspect.Parameter.empty else f" = {param.default!r}"
        print(f"  {name}{default}")

    agent = create_agent(
        model=fake_model(AIMessage(content="北京今天晴。")),
        tools=[get_weather],
        system_prompt="你是一个简洁的天气助手。",
        name="weather_assistant",
    )
    result = agent.invoke({"messages": [HumanMessage("北京天气")]})
    print("name 参数不影响行为，最终回答:", result["messages"][-1].content)


# ---------- demo 2：系统提示，静态与动态 ----------

@dynamic_prompt
def round_aware_prompt(request) -> str:
    """按对话轮数生成不同的系统提示。"""
    rounds = len(request.state["messages"])
    if rounds > 2:
        return "对话已经很长，回答务必精简，不超过一句话。"
    return "你是简洁的天气助手，回答不超过两句话。"


def demo_system_prompt() -> None:
    """system_prompt 静态写法，以及 dynamic_prompt 动态生成。"""
    # 动态提示：同一个 agent，第一轮与第二轮收到的系统提示不同。
    # 多轮对话靠 checkpointer 记住历史，thread_id 标识同一条线程。
    from langgraph.checkpoint.memory import InMemorySaver

    model = fake_model(
        tool_call("get_weather", {"city": "北京"}, "c1"),
        AIMessage(content="北京今天晴。"),
        AIMessage(content="上海多云，不用带伞。"),
    )
    agent = create_agent(
        model=model, tools=[get_weather], middleware=[round_aware_prompt],
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "t1"}}
    agent.invoke({"messages": [HumanMessage("北京天气")]}, config=config)
    result = agent.invoke(
        {"messages": [HumanMessage("那上海呢？该带伞吗？")]}, config=config,
    )
    print("== 动态系统提示 ==")
    print("两轮对话后最终回答:", result["messages"][-1].content)


# ---------- demo 3：自定义 state ----------

class AgentNotes(AgentState):
    """在默认 messages 之外加一个自定义字段。

    Annotated + operator.add 是合并策略，一轮并行调多个工具时多条
    记录相加，而不是互相覆盖。
    """
    audit_log: Annotated[list[str], operator.add]
    user_name: str


class NoteTaking(AgentMiddleware):
    """before_agent 记一条开工日志，wrap_tool_call 把每次工具调用记进 audit_log。"""

    def before_agent(self, state, runtime):
        return {"audit_log": ["开工"]}

    def wrap_tool_call(self, request, handler):
        from langgraph.types import Command

        result = handler(request)
        name = request.tool_call["name"]
        return Command(update={"messages": [result], "audit_log": [f"调用 {name}"]})


def demo_state_schema() -> None:
    """state_schema 扩展：自定义字段随消息一起流转。"""
    model = fake_model(
        tool_call("get_weather", {"city": "北京"}, "c2"),
        AIMessage(content="北京今天晴。"),
    )
    agent = create_agent(
        model=model, tools=[get_weather],
        middleware=[NoteTaking()], state_schema=AgentNotes,
    )
    result = agent.invoke({"messages": [HumanMessage("北京天气")]})
    print("== 自定义 state ==")
    print("audit_log:", result.get("audit_log"))


# ---------- demo 4：六钩子时序 ----------

class Trace(AgentMiddleware):
    """把每个钩子的触发顺序打印出来。"""

    def before_agent(self, state, runtime):
        print("  1 before_agent   整个 run 开始前，一次")
        return None

    def before_model(self, state, runtime):
        print("  2 before_model   每轮调模型前")
        return None

    def wrap_tool_call(self, request, handler):
        print("  3 wrap_tool_call 工具执行外包一层")
        return handler(request)

    def after_model(self, state, runtime):
        print("  4 after_model    每轮调模型后")
        return None

    def after_agent(self, state, runtime):
        print("  5 after_agent    整个 run 结束前，一次")
        return None


def demo_hooks() -> None:
    """一次完整运行，观察五个钩子的触发时序。"""
    model = fake_model(
        tool_call("get_weather", {"city": "上海"}, "c3"),
        AIMessage(content="上海今天晴。"),
    )
    agent = create_agent(model=model, tools=[get_weather], middleware=[Trace()])
    print("== 钩子时序 ==")
    agent.invoke({"messages": [HumanMessage("上海天气")]})


# ---------- demo 5：wrap_tool_call 审计与改写 ----------

@wrap_tool_call
def audit(request, handler):
    """打印每次工具调用的入参与出参。"""
    print(f"  [审计] {request.tool_call['name']} 入参 {request.tool_call['args']}")
    result = handler(request)
    print(f"  [审计] 出参 {result.content[:24]}")
    return result


def demo_wrap_tool_call() -> None:
    """wrap_tool_call 拿到 ToolCallRequest，可改入参、改结果、可拒绝。"""
    model = fake_model(
        tool_call("get_weather", {"city": "深圳"}, "c4"),
        AIMessage(content="深圳今天晴。"),
    )
    agent = create_agent(model=model, tools=[get_weather], middleware=[audit])
    print("== 工具审计 ==")
    agent.invoke({"messages": [HumanMessage("深圳天气")]})


# ---------- demo 6：before_model 拦截 ----------

@before_model(can_jump_to=["end"])
def sensitive_guard(state, runtime):
    """问题里带敏感词就不让模型跑，直接结束。"""
    question = state["messages"][0].content
    if "密码" in str(question):
        return {"jump_to": "end"}
    return None


def demo_jump_to() -> None:
    """jump_to 控制流：before_model 里设 end，模型一轮都不跑。"""
    model = fake_model(AIMessage(content="这句话不该出现。"))
    agent = create_agent(model=model, tools=[get_weather], middleware=[sensitive_guard])
    result = agent.invoke({"messages": [HumanMessage("我的密码是多少？")]})
    print("== before_model 拦截 ==")
    print("拦截后消息序列:", [type(m).__name__ for m in result["messages"]])


# ---------- demo 7：模型调用轮数限制 ----------

def demo_model_call_limit() -> None:
    """ModelCallLimitMiddleware 防止模型无限循环调工具。"""
    script = [tool_call("get_weather", {"city": f"城市{i}"}, f"c{i}") for i in range(5)]
    agent = create_agent(
        model=fake_model(*script),
        tools=[get_weather],
        middleware=[ModelCallLimitMiddleware(thread_limit=3, exit_behavior="end")],
    )
    result = agent.invoke({"messages": [HumanMessage("把每个城市的天气都查一遍")]})
    rounds = sum(1 for m in result["messages"] if type(m).__name__ == "ToolMessage")
    print("== 轮数限制 ==")
    print(f"模型要求调 5 次工具，实际执行 {rounds} 次后被掐断")


# ---------- demo 8：结构化输出 ----------

class Brief(BaseModel):
    """简报结构。"""
    summary: str
    need_umbrella: bool


def demo_response_format() -> None:
    """response_format 让 agent 的最终产出是校验过的对象，进 structured_response。"""
    model = fake_model(
        tool_call("get_weather", {"city": "北京"}, "c5"),
        tool_call("Brief", {"summary": "北京晴，不用带伞", "need_umbrella": False}, "c5r"),
    )
    agent = create_agent(model=model, tools=[get_weather], response_format=Brief)
    result = agent.invoke({"messages": [HumanMessage("北京需要带伞吗？")]})
    print("== 结构化输出 ==")
    print("keys:", sorted(result.keys()))
    print("structured_response:", result["structured_response"])


# ---------- demo 9：流式执行 ----------

def demo_stream() -> None:
    """updates 看每步状态，custom 收工具内部的自定义进度。"""
    model = fake_model(
        tool_call("lookup_city", {"city": "北京"}, "c6"),
        AIMessage(content="北京人口 2100 万，今天晴。"),
    )
    agent = create_agent(model=model, tools=[lookup_city])

    print("== stream_mode=updates（每步状态更新） ==")
    for chunk in agent.stream({"messages": [HumanMessage("北京有多少人？天气如何？")]},
                              stream_mode="updates"):
        for node, update in chunk.items():
            kinds = [type(m).__name__ for m in update.get("messages", [])]
            print(f"  {node}: {kinds}")

    model2 = fake_model(
        tool_call("lookup_city", {"city": "北京"}, "c7"),
        AIMessage(content="北京人口 2100 万，今天晴。"),
    )
    agent2 = create_agent(model=model2, tools=[lookup_city])
    print("== stream_mode=custom（工具内部自定义进度） ==")
    for data in agent2.stream({"messages": [HumanMessage("北京有多少人？天气如何？")]},
                              stream_mode="custom"):
        print("  [进度]", data)


# ---------- 入口 ----------

DEMOS = {
    "params": demo_params,
    "prompt": demo_system_prompt,
    "state": demo_state_schema,
    "hooks": demo_hooks,
    "audit": demo_wrap_tool_call,
    "guard": demo_jump_to,
    "limit": demo_model_call_limit,
    "format": demo_response_format,
    "stream": demo_stream,
}


def main() -> None:
    import sys

    names = sys.argv[1:] or list(DEMOS)
    for name in names:
        DEMOS[name]()


if __name__ == "__main__":
    main()

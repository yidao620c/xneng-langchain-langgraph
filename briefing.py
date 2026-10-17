"""第 5 篇完整案例：值守简报员（briefing.py）。

把本篇的中间件件件用上。接到一个问题，它先查资料再写简报：
- NoteKeeper 中间件把每次工具调用记进自定义 state 字段 audit_log；
- 动态提示把审计条数带进系统提示，模型每轮都知道自己干到哪了；
- ModelCallLimitMiddleware 限 4 轮，模型陷入循环时强制收工；
- response_format 让最终产出是校验过的 Brief 对象，不是一段自由文本；
- --stream 模式用 updates 流式看它每一步在干什么。

用法：
    uv run python briefing.py --list
    uv run python briefing.py "北京适合跑步吗？顺便算下 6 乘 8"
    uv run python briefing.py "..." --stream
    uv run python briefing.py "..." --model openai:gpt-5.5 --temperature 0

模型来源与前三篇一致：--model 或环境变量 AGENT_MODEL 指定时用真实模型；
都没指定时用 ScriptedChatModel 本地演示。
"""

import argparse
import json
import operator
import os
from datetime import datetime
from typing import Annotated

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRequest,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langchain.tools import tool
from pydantic import BaseModel, Field

from fake_models import ScriptedChatModel


# ---------- 简报结构与工具 ----------

class Brief(BaseModel):
    """值守简报的固定结构。"""
    title: str = Field(description="简报标题，一句话")
    points: list[str] = Field(description="要点列表，每条一句话")
    confidence: str = Field(description="信息可信度，high / medium / low")


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。

    Args:
        city: 城市名
    """
    return f"{city}今天晴，气温 26℃，湿度 40%，微风。"


_ARITH_HINT = "只支持数字与加减乘除"


@tool
def calculate(expression: str) -> str:
    """计算四则运算表达式。

    Args:
        expression: 算式，例如 128 * 4
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
    except Exception:
        return f"算不了，{_ARITH_HINT}"
    return f"{expression} = {result}"


# ---------- 中间件：审计 + 动态提示 + 轮数限制 ----------

class DutyState(AgentState):
    """在 messages 之外记录审计日志。

    模型可能一轮并行调多个工具，每个 wrap_tool_call 都写 audit_log，
    用 operator.add 让多条记录合并而不是互相覆盖。
    """
    audit_log: Annotated[list[str], operator.add]


class NoteKeeper(AgentMiddleware):
    """把每次工具调用记进 audit_log，这是简报「依据」的来源。"""

    def wrap_tool_call(self, request, handler):
        from langgraph.types import Command

        result = handler(request)
        call = request.tool_call
        note = f"{datetime.now().strftime('%H:%M:%S')} 调用 {call['name']}({call['args']})"
        return Command(update={"messages": [result], "audit_log": [note]})


@dynamic_prompt
def duty_prompt(request: ModelRequest) -> str:
    """把审计进度写进系统提示，模型每轮都知道自己已经干了什么。"""
    logs = request.state.get("audit_log") or []
    audit = "；".join(logs) if logs else "还没有调用过任何工具"
    return (
        "你是值守简报员，用工具核实信息后输出简报，回答不超过三句话。"
        f"已完成的操作：{audit}。"
    )


# ---------- 模型与 CLI ----------

def build_model(model_name=None, temperature=None):
    """按参数组装模型（约定同前三篇）。"""
    name = model_name or os.environ.get("AGENT_MODEL")
    params = {}
    if temperature is not None:
        params["temperature"] = temperature
    if name:
        if os.environ.get("YI_BASE_URL"):
            params["base_url"] = os.environ["YI_BASE_URL"]
        if os.environ.get("YI_API_KEY"):
            params["api_key"] = os.environ["YI_API_KEY"]
        return init_chat_model(name, **params)
    return ScriptedChatModel(
        messages=iter(
            [
                AIMessage(  # 1 查天气
                    content="好的",
                    tool_calls=[{
                        "name": "get_weather", "args": {"city": "北京"},
                        "id": "call_1", "type": "tool_call",
                    }],
                ),
                AIMessage(  # 2 查完了写结构化简报
                    content="简报写好了",
                    tool_calls=[{
                        "name": "Brief",
                        "args": {
                            "title": "北京天气简报",
                            "points": ["北京今天晴，气温 26℃", "湿度 40%，微风，适合户外活动"],
                            "confidence": "high",
                        },
                        "id": "call_2", "type": "tool_call",
                    }],
                ),
            ]
        ),
        **params,
    )


def build_agent(model, brief: bool = True):
    """中间件件件上岗的值守简报员。

    brief=True 时 response_format 用 ToolStrategy，schema 变成一个工具，
    模型调用它就算交简报。注意 ToolStrategy 会强制 tool_choice，部分网关
    的思考模式不支持，报 400 时用 brief=False 降级成纯文本回答。
    """
    return create_agent(
        model=model,
        tools=[get_weather, calculate],
        middleware=[
            NoteKeeper(),
            duty_prompt,
            ModelCallLimitMiddleware(thread_limit=4, exit_behavior="end"),
        ],
        response_format=Brief if brief else None,
        state_schema=DutyState,
        name="duty_briefing",
    )


def print_brief(brief: Brief) -> None:
    print(f"简报: {brief.title}")
    for p in brief.points:
        print(f"  · {p}")
    print(f"可信度: {brief.confidence}")


def main() -> None:
    parser = argparse.ArgumentParser(description="值守简报员：查资料、记审计、出结构化简报")
    parser.add_argument("question", nargs="?", help="交给简报员的问题")
    parser.add_argument("--list", action="store_true", help="打印能力清单")
    parser.add_argument("--stream", action="store_true", help="流式观察每一步")
    parser.add_argument("--plain", action="store_true", help="不用结构化输出，纯文本回答（网关不支持 tool_choice 时降级用）")
    parser.add_argument("--model", help="提供商:模型名，例如 openai:gpt-5.5")
    parser.add_argument("--temperature", type=float, help="采样温度")
    cli = parser.parse_args()

    if cli.list:
        print("工具: get_weather, calculate")
        print("中间件: NoteKeeper（审计）, duty_prompt（动态提示）, ModelCallLimitMiddleware（限 4 轮）")
        print("输出: Brief(title, points, confidence)")
        return
    if not cli.question:
        parser.error("给一个问题，或者用 --list")

    model = build_model(cli.model, cli.temperature)
    name = cli.model or os.environ.get("AGENT_MODEL") or "ScriptedChatModel（本地替身）"
    print(f"模型: {name}")
    print(f"问题: {cli.question}")
    print()

    agent = build_agent(model, brief=not cli.plain)
    inputs = {"messages": [{"role": "user", "content": cli.question}]}

    if cli.stream:
        brief = None
        audit = []
        plain_answer = ""
        for chunk in agent.stream(inputs, stream_mode="updates"):
            for node, update in chunk.items():
                update = update or {}
                for m in update.get("messages", []):
                    kind = type(m).__name__
                    extra = ""
                    if getattr(m, "tool_calls", None):
                        extra = " -> " + "; ".join(
                            f"{c['name']}({c['args']})" for c in m.tool_calls)
                    print(f"  [{node}] {kind}{extra}")
                    if kind == "AIMessage" and m.content and not m.tool_calls:
                        plain_answer = m.content
                if update.get("structured_response") is not None:
                    brief = update["structured_response"]
                audit.extend(update.get("audit_log", []))
    else:
        result = agent.invoke(inputs)
        brief = result.get("structured_response")
        audit = result.get("audit_log") or []
        plain_answer = result["messages"][-1].content

    print()
    print("审计日志:")
    for note in audit:
        print(f"  {note}")
    print()
    if isinstance(brief, Brief):
        print_brief(brief)
    else:
        # plain 模式或被限流截断时，打印模型的文本回答
        print(f"回答: {plain_answer}")


if __name__ == "__main__":
    main()

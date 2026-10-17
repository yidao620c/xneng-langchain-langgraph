"""第 9 篇完整案例：带人工审批的发布流水线（release.py）。

把本篇的进阶件件用上：
- plan 节点生成变更摘要，approve 节点用 interrupt() 把摘要递给
  人工，等批准或驳回；
- 恢复值带 response_schema（Decision）校验，乱按直接被拦；
- 批准走 deploy，驳回走 reject，路由在节点内用 Command 完成；
- checkpointer 默认 InMemorySaver 跑通单进程交互闭环，
  --db 指定 SQLite 文件就能演示「进程重启后接着审批」。

用法：
    uv run python release.py --list
    uv run python release.py "把搜索阈值从 0.8 调到 0.75"
    uv run python release.py "..." --db release_state.db
    uv run python release.py "..." --auto-approve   # 无人值守演示
"""

import argparse
import sqlite3
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field



class Decision(BaseModel):
    """人工审批的恢复值结构，不合规的输入在这里被拦下。"""
    approved: bool = Field(description="是否批准")
    note: str = Field(default="", description="审批备注，一句话")


class State(TypedDict):
    change: str
    summary: str
    decision: Decision | None
    result: str


# ---------- 节点 ----------

def plan(state: State):
    """把变更描述整理成摘要。真实模型没接就本地拼装。"""
    change = state["change"]
    model = build_model()
    if model is not None:
        reply = model.invoke(f"把这条变更整理成一句发布摘要：{change}")
        summary = str(reply.content)
    else:
        summary = f"变更「{change}」已通过静态检查，涉及 1 个配置文件，风险等级低。"
    return {"summary": summary}


def approve(state: State) -> Command[Literal["deploy", "reject"]]:
    """把摘要递给人工，恢复值经 Decision 校验后路由。"""
    decision = interrupt(
        {"question": "是否批准发布？", "summary": state["summary"]},
        response_schema=Decision,
    )
    if decision.approved:
        return Command(update={"decision": decision}, goto="deploy")
    return Command(update={"decision": decision}, goto="reject")


def deploy(state: State):
    note = state["decision"].note or "无备注"
    return {"result": f"已发布。审批备注：{note}"}


def reject(state: State):
    note = state["decision"].note or "无备注"
    return {"result": f"已驳回。审批备注：{note}"}


# ---------- 组图与模型 ----------

def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("approve", approve)
    g.add_node("deploy", deploy)
    g.add_node("reject", reject)
    g.add_edge(START, "plan")
    g.add_edge("plan", "approve")
    return g.compile(checkpointer=checkpointer)


def build_model():
    """有 AGENT_MODEL 用真实模型，否则返回 None 走本地拼装。"""
    import os

    name = os.environ.get("AGENT_MODEL")
    if not name:
        return None
    from langchain.chat_models import init_chat_model

    params = {}
    if os.environ.get("YI_BASE_URL"):
        params["base_url"] = os.environ["YI_BASE_URL"]
    if os.environ.get("YI_API_KEY"):
        params["api_key"] = os.environ["YI_API_KEY"]
    return init_chat_model(name, **params)


def make_checkpointer(db_path):
    if db_path:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    return InMemorySaver()


# ---------- CLI ----------

def ask_human(payload) -> dict:
    """终端里问人，返回符合 Decision 的字典。"""
    print()
    print(f"变更摘要: {payload['summary']}")
    answer = input("批准发布？(y/n) ").strip().lower()
    note = input("审批备注（可空）: ").strip()
    return {"approved": answer == "y", "note": note}


def main() -> None:
    parser = argparse.ArgumentParser(description="带人工审批的发布流水线")
    parser.add_argument("change", nargs="?", help="变更描述")
    parser.add_argument("--list", action="store_true", help="打印图的节点清单")
    parser.add_argument("--db", help="SQLite 文件路径，指定后重启进程也能接着审批")
    parser.add_argument("--auto-approve", action="store_true", help="无人值守演示，直接批准")
    cli = parser.parse_args()

    if cli.list:
        print("节点: plan, approve(interrupt), deploy, reject")
        print("恢复值: Decision(approved, note)，response_schema 校验")
        print("checkpointer: InMemorySaver 默认；--db 换 SqliteSaver")
        return
    if not cli.change:
        parser.error("给一条变更描述，或者用 --list")

    app = build_graph(make_checkpointer(cli.db))
    cfg = {"configurable": {"thread_id": "release-1"}}

    result = app.invoke({"change": cli.change, "summary": "", "decision": None, "result": ""}, cfg)
    if "__interrupt__" not in result:
        print(result["result"])
        return

    payload = result["__interrupt__"][0].value
    if cli.auto_approve:
        resume = {"approved": True, "note": "无人值守演示，自动批准"}
    else:
        resume = ask_human(payload)

    final = app.invoke(Command(resume=resume), cfg)
    print()
    print(f"结论: {final['result']}")


if __name__ == "__main__":
    main()

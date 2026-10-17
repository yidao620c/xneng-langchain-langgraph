"""第 10 篇完整案例：编辑部多智能体流水线（newsroom.py）。

把本篇的编排件件用上：
- supervisor 按阶段调度三个 agent，research 与 write 完工把控制权
  交回 supervisor，路由全部走 Command；
- reviewer 是质检关口，草稿带禁词就地打回，批注经 Command 更新
  写回状态，writer 带着批注重写；
- steps 用 reducer 合并每个 agent 的动作，trace 能看见完整协作链。

用法：
    uv run python newsroom.py --list
    uv run python newsroom.py "LangGraph 是什么"
    uv run python newsroom.py "LangGraph 是什么" --trace
    uv run python newsroom.py "写一篇绝对要用'赋能'这个词的稿子"   # 被打回一次
"""

import argparse
from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

BANNED = ("赋能", "抓手", "闭环")
REWRITE = {"赋能": "助力", "抓手": "切入点", "闭环": "循环"}
MAX_REVISIONS = 3


class NewsroomState(TypedDict):
    topic: str
    notes: Annotated[list[str], add]
    draft: str
    review_note: str
    revisions: int
    steps: Annotated[list[str], add]
    final: str


# ---------- 三个 agent ----------

def researcher(state: NewsroomState):
    """查资料。真实实现是一个带检索工具的 create_agent 子图。"""
    return {
        "notes": [f"关于「{state['topic']}」的查证要点 3 条"],
        "steps": ["researcher: 收集资料"],
    }


def writer(state: NewsroomState):
    """写稿。有审校批注就按批注真改，改完清掉批注等再审。"""
    note = state.get("review_note") or ""
    if note:
        draft = state["draft"]
        for word in BANNED:                 # 改稿就是把禁词换掉
            draft = draft.replace(word, REWRITE[word])
        return {"draft": draft, "review_note": "",
                "steps": [f"writer: 按批注出第 {state['revisions'] + 1} 版"]}
    draft = f"初稿：{state['topic']} 值得了解，理由来自资料。"
    return {"draft": draft, "steps": ["writer: 出初稿"]}


def reviewer(state: NewsroomState) -> Command[Literal["supervisor"]]:
    """审校。带禁词就地打回，Command 把批注带给 writer。"""
    draft = state["draft"]
    hit = next((w for w in BANNED if w in draft), None)
    if hit:
        return Command(update={
            "review_note": f"禁词「{hit}」请替换",
            "revisions": state["revisions"] + 1,
            "steps": [f"reviewer: 打回（发现禁词 {hit}）"],
        }, goto="supervisor")
    return Command(update={
        "final": f"定稿：{draft}",
        "steps": ["reviewer: 审校通过"],
    }, goto="supervisor")


# ---------- supervisor ----------

def supervisor(state: NewsroomState) -> Command[Literal[
        "research", "writer", "reviewer", "__end__"]]:
    """按当前状态决定下一步，真实实现换成模型结构化输出即可。"""
    if not state["notes"]:
        return Command(goto="research")
    if not state["draft"]:
        return Command(goto="writer")
    if state["revisions"] >= MAX_REVISIONS:   # 返工上限，防止无限打回
        return Command(update={
            "final": f"审校打回 {state['revisions']} 次仍未通过，转人工处理",
            "steps": ["supervisor: 达到返工上限，终止流水线"],
        }, goto="__end__")
    if state["review_note"]:
        return Command(goto="writer")       # 有批注说明被打回过，先改稿
    if not state["final"]:
        return Command(goto="reviewer")
    return Command(goto="__end__")


def build_graph():
    g = StateGraph(NewsroomState)
    g.add_node("supervisor", supervisor)
    g.add_node("research", researcher)
    g.add_node("writer", writer)
    g.add_node("reviewer", reviewer)
    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", END)
    for agent in ("research", "writer", "reviewer"):
        g.add_edge(agent, "supervisor")
    return g.compile()


def print_trace(chunks) -> dict:
    """按 updates 流打印每步轨迹并累积最终状态。"""
    state: dict = {}
    for chunk in chunks:
        for node, update in chunk.items():
            if node == "__end__":
                continue
            update = update or {}
            steps = update.get("steps") or []
            marker = f"  [{node}]"
            fields = ", ".join(
                f"{k}={str(v)[:52]}" for k, v in update.items() if k not in ("steps",))
            print(f"{marker} {fields}".rstrip())
            for s in steps:
                print(f"      · {s}")
            for k, v in update.items():
                if k in ("steps", "notes") and k in state:
                    state[k] = state[k] + v
                else:
                    state[k] = v
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="编辑部多智能体流水线")
    parser.add_argument("topic", nargs="?", help="选题")
    parser.add_argument("--list", action="store_true", help="打印团队清单")
    parser.add_argument("--trace", action="store_true", help="打印协作轨迹")
    cli = parser.parse_args()

    if cli.list:
        print("团队: supervisor, researcher, writer, reviewer")
        print("路由: supervisor 队列调度；reviewer 发现禁词直接打回 writer")
        return
    if not cli.topic:
        parser.error("给一个选题，或者用 --list")

    app = build_graph()
    inputs = {"topic": cli.topic, "notes": [], "draft": "",
              "review_note": "", "revisions": 0, "steps": [], "final": ""}

    if cli.trace:
        print("协作轨迹:")
        result = print_trace(app.stream(inputs, stream_mode="updates"))
    else:
        result = app.invoke(inputs)
    print()
    print(f"协作步骤: {' -> '.join(result['steps'])}")
    print(f"成稿: {result['final']}")


if __name__ == "__main__":
    main()

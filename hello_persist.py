"""第 9 篇示例：LangGraph 进阶（hello_persist.py）。

把「断点、恢复、回放」拆成九个 demo：中断闭环、结构化 payload、
节点重跑、吞异常坑、静态断点、状态历史、时间旅行、SqliteSaver
落盘、response_schema 校验。全部纯函数，不依赖任何模型。

九个 demo 用的都是同一个审批图骨架，节点里放一个 interrupt()，
恢复值走不同分支，便于对照。
"""

import sqlite3
import tempfile
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel


# ---------- 共享的审批图骨架 ----------

class State(TypedDict):
    task: str
    approved: bool
    result: str


def do_work(state: State):
    return {"result": f"已处理 {state['task']}"}


def approval(state: State):
    answer = interrupt({"question": "批准吗？", "task": state["task"]})
    return {"approved": answer}


def finish(state: State):
    verdict = "通过" if state["approved"] else "驳回"
    return {"result": f"{state.get('result', '')} -> 人工{verdict}"}


def build_approval_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("work", do_work)
    g.add_node("approval", approval)
    g.add_node("finish", finish)
    g.add_edge(START, "work")
    g.add_edge("work", "approval")
    g.add_conditional_edges("approval", lambda s: "finish" if s["approved"] else END)
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


# ---------- demo 1：中断闭环 ----------

def demo_interrupt() -> None:
    """interrupt 暂停等输入，Command(resume) 的值成为它的返回值。"""
    app = build_approval_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    result = app.invoke({"task": "删库跑路", "approved": False, "result": ""}, cfg)
    print("== 中断闭环 ==")
    print(f"  中断 payload: {result['__interrupt__'][0].value}")
    paused = app.invoke(Command(resume=False), cfg)
    print(f"  恢复 False（拒绝）: 图在 {paused.get('result')} 后结束")
    app2 = build_approval_graph(InMemorySaver())
    cfg2 = {"configurable": {"thread_id": "t2"}}
    app2.invoke({"task": "发布文档", "approved": False, "result": ""}, cfg2)
    ok = app2.invoke(Command(resume=True), cfg2)
    print(f"  恢复 True（批准）: {ok['result']}")


# ---------- demo 2：结构化 payload ----------

def demo_payload() -> None:
    """payload 可以是任意 JSON 值，恢复值原样成为 interrupt 的返回值。"""
    class Ask(TypedDict):
        task: str
        answer: str

    def ask(state: Ask):
        picked = interrupt({"options": ["加急", "普通"], "default": "普通"})
        return {"answer": picked}

    g = StateGraph(Ask)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    result = app.invoke({"task": "发货", "answer": ""}, cfg)
    payload = result["__interrupt__"][0].value
    resumed = app.invoke(Command(resume="加急"), cfg)
    print("== 结构化 payload ==")
    print(f"  发出去的问题: {payload}")
    print(f"  收回来的答案: {resumed['answer']}")


# ---------- demo 3：节点从头重跑 ----------

def demo_rerun() -> None:
    """interrupt 抛异常暂停，恢复时节点从第一行重跑，副作用要幂等。"""
    counter = {"n": 0}

    class Ask(TypedDict):
        task: str
        answer: str

    def flaky(state: Ask):
        counter["n"] += 1  # interrupt 之前的代码，恢复时会再执行一遍
        answer = interrupt("确认？")
        return {"answer": answer}

    g = StateGraph(Ask)
    g.add_node("flaky", flaky)
    g.add_edge(START, "flaky")
    g.add_edge("flaky", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    app.invoke({"task": "x", "answer": ""}, cfg)
    app.invoke(Command(resume="好的"), cfg)
    print("== 节点重跑 ==")
    print(f"  节点函数执行了 {counter['n']} 次（中断一次 + 恢复重跑一次）")


# ---------- demo 4：吞异常坑 ----------

def demo_swallow() -> None:
    """try/except 包住 interrupt 会吞掉暂停信号，节点直接跑完。"""
    class Ask(TypedDict):
        task: str
        answer: str

    def bad(state: Ask):
        try:
            answer = interrupt("确认？")
        except Exception:
            answer = "fallback"  # 中断就是靠异常实现的，这里把它吞了
        return {"answer": answer}

    g = StateGraph(Ask)
    g.add_node("bad", bad)
    g.add_edge(START, "bad")
    g.add_edge("bad", END)
    app = g.compile(checkpointer=InMemorySaver())
    result = app.invoke({"task": "y", "answer": ""}, {"configurable": {"thread_id": "t"}})
    print("== 吞异常坑 ==")
    print(f"  图没有暂停，直接得到: {result['answer']}")


# ---------- demo 5：静态断点 ----------

def demo_static_break() -> None:
    """interrupt_before 编译期钉死断点，invoke(None) 从断点续跑。"""
    class State(TypedDict):
        n: int

    g = StateGraph(State)
    g.add_node("a", lambda s: {"n": s["n"] + 1})
    g.add_node("b", lambda s: {"n": s["n"] * 10})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    app = g.compile(checkpointer=InMemorySaver(), interrupt_before=["b"])
    cfg = {"configurable": {"thread_id": "t1"}}
    app.invoke({"n": 2}, cfg)
    snapshot = app.get_state(cfg)
    print("== 静态断点 ==")
    print(f"  停在 {snapshot.next} 前，状态 {snapshot.values}")
    resumed = app.invoke(None, cfg)
    print(f"  invoke(None) 续跑: {resumed}")


# ---------- demo 6：状态历史 ----------

def demo_history() -> None:
    """每个 super-step 存一个 checkpoint，get_state_history 全部翻出来。"""
    class State(TypedDict):
        n: int

    g = StateGraph(State)
    g.add_node("a", lambda s: {"n": s["n"] + 1})
    g.add_node("b", lambda s: {"n": s["n"] * 10})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    app = g.compile(checkpointer=InMemorySaver(), interrupt_before=["b"])
    cfg = {"configurable": {"thread_id": "t1"}}
    app.invoke({"n": 2}, cfg)
    history = sorted(app.get_state_history(cfg), key=lambda s: s.metadata.get("step", 0))
    print("== 状态历史 ==")
    for snap in history:
        step = snap.metadata.get("step")
        print(f"  step={step} next={snap.next} values={snap.values}")


# ---------- demo 7：时间旅行 ----------

def demo_timetravel() -> None:
    """拿历史 checkpoint 的 config 重新 invoke，之前的步骤跳过，之后重放。"""
    class State(TypedDict):
        n: int

    def b(state: State):
        return {"n": state["n"] * 10}

    g = StateGraph(State)
    g.add_node("a", lambda s: {"n": s["n"] + 1})
    g.add_node("b", b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    app = g.compile(checkpointer=InMemorySaver(), interrupt_before=["b"])
    cfg = {"configurable": {"thread_id": "t1"}}
    app.invoke({"n": 2}, cfg)
    history = list(app.get_state_history(cfg))
    step1 = next(s for s in history if s.metadata.get("step") == 1)
    replayed = app.invoke(None, step1.config)
    print("== 时间旅行 ==")
    print(f"  从 step=1 的 checkpoint 回放: {replayed}")


# ---------- demo 8：SqliteSaver 落盘 ----------

def demo_sqlite() -> None:
    """checkpoint 落进 SQLite 文件，换个连接照样接着跑。"""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "ck.db"
        conn = sqlite3.connect(db, check_same_thread=False)
        app = build_approval_graph(SqliteSaver(conn))
        cfg = {"configurable": {"thread_id": "t1"}}
        paused = app.invoke({"task": "发布上线", "approved": False, "result": ""}, cfg)
        interrupted = "__interrupt__" in paused
        conn.close()

        conn2 = sqlite3.connect(db, check_same_thread=False)
        app2 = build_approval_graph(SqliteSaver(conn2))
        resumed = app2.invoke(Command(resume=True), cfg)
        conn2.close()
        print("== SqliteSaver 落盘 ==")
        print(f"  第一段连接中断: {interrupted}")
        print(f"  新连接恢复: {resumed['result']}")


# ---------- demo 9：response_schema 校验 ----------

def demo_schema() -> None:
    """给 interrupt 配 response_schema，恢复值不符合 schema 直接抛错可重试。"""
    class Decision(BaseModel):
        approved: bool
        reason: str

    class Ask(TypedDict):
        task: str
        answer: str

    def decide(state: Ask):
        answer = interrupt("审批", response_schema=Decision)
        return {"answer": f"{answer.approved}（{answer.reason}）"}

    g = StateGraph(Ask)
    g.add_node("d", decide)
    g.add_edge(START, "d")
    g.add_edge("d", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    app.invoke({"task": "x", "answer": ""}, cfg)
    try:
        app.invoke(Command(resume={"approved": "yes"}), cfg)
    except Exception as e:
        print("== response_schema ==")
        print(f"  非法恢复值被拦下: {type(e).__name__}")
    ok = app.invoke(Command(resume={"approved": True, "reason": "没问题"}), cfg)
    print(f"  合法恢复值: {ok['answer']}")


# ---------- 入口 ----------

DEMOS = {
    "interrupt": demo_interrupt,
    "payload": demo_payload,
    "rerun": demo_rerun,
    "swallow": demo_swallow,
    "static": demo_static_break,
    "history": demo_history,
    "timetravel": demo_timetravel,
    "sqlite": demo_sqlite,
    "schema": demo_schema,
}


def main() -> None:
    import sys

    names = [n for n in sys.argv[1:] if n in DEMOS] or list(DEMOS)
    for name in names:
        DEMOS[name]()


if __name__ == "__main__":
    main()

"""第 9 篇示例测试：LangGraph 进阶（hello_persist.py）。

全部纯函数图，不依赖任何模型与 API key。
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError
from typing import TypedDict

import hello_persist
from hello_persist import build_approval_graph


CFG = lambda tid: {"configurable": {"thread_id": tid}}  # noqa: E731


def run_paused(task="删库跑路", tid="t1"):
    """发起一次审批并停在中断点，返回图实例与 config。"""
    app = build_approval_graph(InMemorySaver())
    cfg = CFG(tid)
    result = app.invoke(
        {"task": task, "approved": False, "result": ""}, cfg)
    return app, cfg, result


# ---------- 中断与恢复 ----------

def test_interrupt_pauses_with_payload():
    """invoke 返回值带 __interrupt__，payload 原样递出。"""
    _, _, result = run_paused()
    assert "__interrupt__" in result
    value = result["__interrupt__"][0].value
    assert value["question"] == "批准吗？"
    assert value["task"] == "删库跑路"


def test_resume_true_routes_to_deploy():
    """恢复 True，图走 finish 分支，人工通过写进结果。"""
    app, cfg, _ = run_paused()
    final = app.invoke(Command(resume=True), cfg)
    assert "人工通过" in final["result"]


def test_resume_false_ends_without_finish():
    """恢复 False，条件边直接到 END，结果停在 work 的产出。"""
    app, cfg, _ = run_paused()
    final = app.invoke(Command(resume=False), cfg)
    assert final["result"] == "已处理 删库跑路"


def test_interrupt_value_becomes_return_value():
    """Command(resume=X) 的 X 就是 interrupt() 的返回值。"""
    app = hello_persist.build_approval_graph(InMemorySaver())
    cfg = CFG("t9")
    app.invoke({"task": "x", "approved": False, "result": ""}, cfg)
    final = app.invoke(Command(resume=True), cfg)
    assert final["approved"] is True


# ---------- 坑位行为 ----------

def test_node_reruns_after_resume():
    """恢复时节点从头重跑，interrupt 之前的代码执行两次。"""
    counter = {"n": 0}

    class Ask(TypedDict):
        task: str
        answer: str

    def flaky(state: Ask):
        counter["n"] += 1
        answer = hello_persist.interrupt("确认？")
        return {"answer": answer}

    g = hello_persist.StateGraph(Ask)
    g.add_node("flaky", flaky)
    g.add_edge(hello_persist.START, "flaky")
    g.add_edge("flaky", hello_persist.END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = CFG("t1")
    app.invoke({"task": "x", "answer": ""}, cfg)
    app.invoke(Command(resume="好的"), cfg)
    assert counter["n"] == 2


def test_try_except_swallows_interrupt():
    """try/except 包住 interrupt，中断信号被吞，图一口气跑完。"""
    class Ask(TypedDict):
        task: str
        answer: str

    def bad(state: Ask):
        try:
            answer = hello_persist.interrupt("确认？")
        except Exception:
            answer = "fallback"
        return {"answer": answer}

    g = hello_persist.StateGraph(Ask)
    g.add_node("bad", bad)
    g.add_edge(hello_persist.START, "bad")
    g.add_edge("bad", hello_persist.END)
    app = g.compile(checkpointer=InMemorySaver())
    result = app.invoke({"task": "y", "answer": ""}, CFG("t1"))
    assert "__interrupt__" not in result
    assert result["answer"] == "fallback"


def test_plain_dict_invoke_hits_same_interrupt():
    """中断后用普通 dict invoke，只会原地再撞同一个断点。"""
    app, cfg, _ = run_paused()
    again = app.invoke({"task": "删库跑路", "approved": False, "result": ""}, cfg)
    assert "__interrupt__" in again


# ---------- 断点、历史与回放 ----------

def build_counter_graph():
    class State(TypedDict):
        n: int

    g = hello_persist.StateGraph(State)
    g.add_node("a", lambda s: {"n": s["n"] + 1})
    g.add_node("b", lambda s: {"n": s["n"] * 10})
    g.add_edge(hello_persist.START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", hello_persist.END)
    return g.compile(checkpointer=InMemorySaver(), interrupt_before=["b"])


def test_static_break_pauses_before_node():
    """interrupt_before 停在目标节点前，invoke(None) 续跑。"""
    app = build_counter_graph()
    cfg = CFG("t1")
    app.invoke({"n": 2}, cfg)
    snapshot = app.get_state(cfg)
    assert snapshot.next == ("b",)
    assert snapshot.values["n"] == 3
    assert app.invoke(None, cfg)["n"] == 30


def test_state_history_has_all_steps():
    """get_state_history 按 super-step 翻出全部 checkpoint。

    interrupt_before 停在 b 之前，b 执行后的快照不存在，
    所以只有 input(-1)、a(0)、b 前(1) 三个。
    """
    app = build_counter_graph()
    cfg = CFG("t1")
    app.invoke({"n": 2}, cfg)
    steps = sorted(
        s.metadata.get("step", 0) for s in app.get_state_history(cfg))
    assert steps == [-1, 0, 1]


def test_replay_from_history_checkpoint():
    """拿历史 checkpoint 的 config 回放，之后的节点重新执行。"""
    app = build_counter_graph()
    cfg = CFG("t1")
    app.invoke({"n": 2}, cfg)
    history = list(app.get_state_history(cfg))
    step1 = next(s for s in history if s.metadata.get("step") == 1)
    assert app.invoke(None, step1.config)["n"] == 30


# ---------- response_schema ----------

def test_response_schema_rejects_bad_resume():
    """恢复值不符合 schema 抛 ValidationError，之后还能合法重试。"""
    class Decision(hello_persist.BaseModel):
        approved: bool
        reason: str

    class Ask(TypedDict):
        task: str
        answer: str

    def decide(state: Ask):
        answer = hello_persist.interrupt("审批", response_schema=Decision)
        return {"answer": answer.reason}

    g = hello_persist.StateGraph(Ask)
    g.add_node("d", decide)
    g.add_edge(hello_persist.START, "d")
    g.add_edge("d", hello_persist.END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = CFG("t1")
    app.invoke({"task": "x", "answer": ""}, cfg)
    with pytest.raises(ValidationError):
        app.invoke(Command(resume={"approved": "yes"}), cfg)
    final = app.invoke(
        Command(resume={"approved": True, "reason": "没问题"}), cfg)
    assert final["answer"] == "没问题"


# ---------- demo 冒烟 ----------

def test_all_demos_run(capsys):
    """九个 demo 全部可执行不抛错。"""
    hello_persist.main()
    out = capsys.readouterr().out
    for marker in ("中断闭环", "结构化 payload", "节点重跑", "吞异常坑",
                   "静态断点", "状态历史", "时间旅行", "SqliteSaver 落盘",
                   "response_schema"):
        assert marker in out

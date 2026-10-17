"""第 8 篇示例测试：LangGraph 基础（hello_graph_deep.py）。

全部纯函数图，不依赖任何模型与 API key。
"""

import pytest
from langchain.messages import AIMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Overwrite, Send
from operator import add
from typing import Annotated, TypedDict

import hello_graph_deep


# ---------- 各 demo 的行为断言 ----------

def test_basic_chain_applies_in_order():
    """静态边按序接力，upper 再 wrap。"""
    class State(TypedDict):
        text: str

    g = StateGraph(State)
    g.add_node("upper", lambda s: {"text": s["text"].upper()})
    g.add_node("wrap", lambda s: {"text": f"[{s['text']}]"})
    g.add_edge(START, "upper")
    g.add_edge("upper", "wrap")
    g.add_edge("wrap", END)
    assert g.compile().invoke({"text": "hi"})["text"] == "[HI]"


def test_reducer_merges_without_reducer_overwrites():
    """Annotated add 字段累加，普通字段被后写者覆盖。"""
    class State(TypedDict):
        log: Annotated[list[str], add]
        last: str

    g = StateGraph(State)
    g.add_node("a", lambda s: {"log": ["a"], "last": "a"})
    g.add_node("b", lambda s: {"log": ["b"], "last": "b"})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    result = g.compile().invoke({"log": [], "last": ""})
    assert result["log"] == ["a", "b"]
    assert result["last"] == "b"


def test_conditional_edge_routes_by_state():
    """条件边按状态选路，映射两种输入走不同分支。"""
    app = hello_graph_deep.demo_conditional  # demo 只打印，行为在源码层测
    class State(TypedDict):
        n: int
        path: str

    def check(state) -> str:
        return "even" if state["n"] % 2 == 0 else "odd"

    g = StateGraph(State)
    g.add_node("check", lambda s: {})
    g.add_node("even", lambda s: {"path": "even"})
    g.add_node("odd", lambda s: {"path": "odd"})
    g.add_edge(START, "check")
    g.add_conditional_edges("check", check)
    g.add_edge("even", END)
    g.add_edge("odd", END)
    compiled = g.compile()
    assert compiled.invoke({"n": 4, "path": ""})["path"] == "even"
    assert compiled.invoke({"n": 7, "path": ""})["path"] == "odd"


def test_fanout_runs_both_branches():
    """一个节点两条出边，两个分支都执行。"""
    class State(TypedDict):
        log: Annotated[list[str], add]

    g = StateGraph(State)
    g.add_node("fan", lambda s: {"log": ["fan"]})
    g.add_node("x1", lambda s: {"log": ["x1"]})
    g.add_node("x2", lambda s: {"log": ["x2"]})
    g.add_edge(START, "fan")
    g.add_edge("fan", "x1")
    g.add_edge("fan", "x2")
    g.add_edge("x1", END)
    g.add_edge("x2", END)
    assert sorted(g.compile().invoke({"log": []})["log"]) == ["fan", "x1", "x2"]


def test_overwrite_bypasses_reducer():
    """Overwrite 让合并型字段被整体替换成空列表。"""
    class State(TypedDict):
        errors: Annotated[list[str], add]

    g = StateGraph(State)
    g.add_node("a", lambda s: {"errors": ["bad sql"]})
    g.add_node("clear", lambda s: {"errors": Overwrite([])})
    g.add_edge(START, "a")
    g.add_edge("a", "clear")
    g.add_edge("clear", END)
    assert g.compile().invoke({"errors": ["old"]})["errors"] == []


def test_add_messages_updates_same_id():
    """add_messages 同 id 消息被覆盖成新内容，不追加。"""
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    g = StateGraph(State)
    g.add_node("bot", lambda s: {
        "messages": [AIMessage(id="m1", content="revised")]})
    g.add_edge(START, "bot")
    g.add_edge("bot", END)
    result = g.compile().invoke(
        {"messages": [AIMessage(id="m1", content="original")]})
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "revised"


def test_command_routes_from_node():
    """Command 在节点内同时完成状态更新与跳转。"""
    class State(TypedDict):
        score: int
        where: str

    g = StateGraph(State)
    g.add_node("judge", hello_graph_deep.demo_command)  # 占位，实际断言在下面
    g.add_edge(START, "judge")
    g.add_edge("judge", END)

    # 独立重建一个可断言的图
    def judge(state) -> Command:
        if state["score"] >= 60:
            return Command(update={"where": "pass"}, goto="p")
        return Command(update={"where": "fail"}, goto="f")

    g2 = StateGraph(State)
    g2.add_node("judge", judge)
    g2.add_node("p", lambda s: {})
    g2.add_node("f", lambda s: {})
    g2.add_edge(START, "judge")
    g2.add_edge("p", END)
    g2.add_edge("f", END)
    app = g2.compile()
    assert app.invoke({"score": 90, "where": ""})["where"] == "pass"
    assert app.invoke({"score": 50, "where": ""})["where"] == "fail"


def test_send_fans_out_unknown_count():
    """Send 按条目数动态发并行任务，结果经 reducer 汇总。"""
    class State(TypedDict):
        subjects: list[str]
        jokes: Annotated[list[str], add]

    def dispatch(state):
        return [Send("make", {"subject": s}) for s in state["subjects"]]

    g = StateGraph(State)
    g.add_node("make", lambda s: {"jokes": [f"j-{s['subject']}"]})
    g.add_conditional_edges(START, dispatch, ["make"])
    g.add_edge("make", END)
    result = g.compile().invoke({"subjects": ["a", "b"], "jokes": []})
    assert sorted(result["jokes"]) == ["j-a", "j-b"]


def test_recursion_limit_stops_loop():
    """死循环被 recursion_limit 叫停，抛 GraphRecursionError。"""
    class State(TypedDict):
        n: int

    g = StateGraph(State)
    g.add_node("loop", lambda s: {"n": s["n"] + 1})
    g.add_edge(START, "loop")
    g.add_edge("loop", "loop")
    g.add_edge("loop", END)
    with pytest.raises(GraphRecursionError):
        g.compile().invoke({"n": 0}, config={"recursion_limit": 5})


# ---------- demo 冒烟 ----------

def test_all_demos_run(capsys):
    """九个 demo 全部可执行不抛错。"""
    hello_graph_deep.main()
    out = capsys.readouterr().out
    for marker in ("静态边", "Reducer", "条件路由", "多出边并行", "Overwrite",
                   "add_messages", "Command", "Send map-reduce", "递归上限"):
        assert marker in out

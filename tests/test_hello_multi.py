"""第 10 篇示例测试：多智能体编排（hello_multi.py）。

全部纯函数图，不依赖任何模型与 API key。
"""

import pytest
from langgraph.errors import ParentCommand
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from operator import add
from typing import Annotated, Literal, TypedDict

import hello_multi


# ---------- 各 demo 对应的行为断言 ----------

def test_subgraph_runs_as_node():
    """compile 过的子图可直接当节点，数据在共享字段间流动。"""
    class Base(TypedDict):
        text: str

    sub = StateGraph(Base)
    sub.add_node("upper", lambda s: {"text": s["text"].upper()})
    sub.add_edge(START, "upper")
    sub.add_edge("upper", END)

    outer = StateGraph(Base)
    outer.add_node("sub", sub.compile())
    outer.add_node("wrap", lambda s: {"text": f"[{s['text']}]"})
    outer.add_edge(START, "sub")
    outer.add_edge("sub", "wrap")
    outer.add_edge("wrap", END)
    assert outer.compile().invoke({"text": "hello"})["text"] == "[HELLO]"


def test_schemas_narrow_input_output():
    """input_schema/output_schema 只暴露窄接口，外层拿不到 raw。"""
    class In(TypedDict):
        raw: str

    class Out(TypedDict):
        clean: str

    class Overall(TypedDict):
        raw: str
        clean: str

    sub = StateGraph(Overall, input_schema=In, output_schema=Out)
    sub.add_node("clean", lambda s: {"clean": s["raw"].strip()})
    sub.add_edge(START, "clean")
    sub.add_edge("clean", END)
    app = sub.compile()

    # 输入走 In，输出只剩 Out 的字段
    result = app.invoke({"raw": "  padded  "})
    assert result["clean"] == "padded"
    assert set(result.keys()) == {"clean"}


def test_handoff_routes_via_command():
    """Command(goto=) 在节点内同时交更新与控制权。"""
    class Team(TypedDict):
        log: Annotated[list[str], add]
        current: str

    def agent_a(state: Team):
        return Command(goto="b", update={"log": ["a 干完移交 b"]})

    g = StateGraph(Team)
    g.add_node("a", agent_a)
    g.add_node("b", lambda s: {"log": ["b 接手完成"], "current": "b"})
    g.add_edge(START, "a")
    g.add_edge("b", END)
    result = g.compile().invoke({"log": [], "current": ""})
    assert result["log"] == ["a 干完移交 b", "b 接手完成"]
    assert result["current"] == "b"


def test_parent_command_at_top_level_raises():
    """顶层图节点返回 Command.PARENT 没有父图可上浮，抛 ParentCommand。"""
    class Team(TypedDict):
        log: Annotated[list[str], add]
        current: str

    def bad(state: Team):
        return Command(goto="b", graph=Command.PARENT)

    g = StateGraph(Team)
    g.add_node("a", bad)
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_edge(START, "a")
    g.add_edge("b", END)
    with pytest.raises(ParentCommand):
        g.compile().invoke({"log": [], "current": ""})


def test_parent_navigate_merges_both_updates():
    """子图节点 Command.PARENT 跳父图节点，两份更新都合并进状态。"""
    class Shared(TypedDict):
        items: Annotated[list[str], add]

    def inner_cmd(state: Shared):
        return Command(goto="outer_end", update={"items": ["child"]},
                       graph=Command.PARENT)

    sub = StateGraph(Shared)
    sub.add_node("cmd", inner_cmd)
    sub.add_edge(START, "cmd")
    sub.add_edge("cmd", END)

    outer = StateGraph(Shared)
    outer.add_node("sub", sub.compile())
    outer.add_node("outer_end", lambda s: {"items": ["outer"]})
    outer.add_edge(START, "sub")
    outer.add_edge("outer_end", END)
    assert outer.compile().invoke({"items": ["init"]})["items"] == \
        ["init", "child", "outer"]


def test_annotation_target_must_be_local():
    """编译期校验 Command[Literal[...]] 注解目标必须在本图节点里。"""
    class Shared(TypedDict):
        items: Annotated[list[str], add]

    def inner_cmd(state: Shared) -> Command[Literal["outer_end"]]:
        return Command(goto="outer_end", graph=Command.PARENT)

    sub = StateGraph(Shared)
    sub.add_node("cmd", inner_cmd)
    sub.add_edge(START, "cmd")
    sub.add_edge("cmd", END)
    # outer_end 是父图节点名，子图编译时就过不了注解校验
    with pytest.raises(ValueError, match="unknown node"):
        sub.compile()


# ---------- demo 冒烟 ----------

def test_all_demos_run(capsys):
    """八个 demo 全部可执行不抛错。"""
    hello_multi.main()
    out = capsys.readouterr().out
    for marker in ("子图当节点", "输入输出收窄", "同图移交", "PARENT 误用坑",
                   "跨图导航", "Supervisor", "Swarm 网络", "可视化"):
        assert marker in out

"""第 10 篇示例：多智能体编排（hello_multi.py）。

把「怎么把多个 agent 排到一张图里」拆成八个 demo：子图当节点、
输入输出收窄、同图移交、PARENT 误用坑、跨图导航、Supervisor、
Swarm、可视化。全部纯函数，不依赖任何模型。

单 agent 是第 5 篇的事，这一篇的每个节点都可能是一个完整的
agent 子图，编排方式决定它们怎么协作。
"""

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


# ---------- demo 1：子图当节点 ----------

def demo_subgraph() -> None:
    """compile 过的图能直接 add_node，共享 schema 的字段全图可见。"""
    class Base(TypedDict):
        text: str

    sub = StateGraph(Base)
    sub.add_node("upper", lambda s: {"text": s["text"].upper()})
    sub.add_edge(START, "upper")
    sub.add_edge("upper", END)
    sub_app = sub.compile()

    outer = StateGraph(Base)
    outer.add_node("sub", sub_app)
    outer.add_node("wrap", lambda s: {"text": f"[{s['text']}]"})
    outer.add_edge(START, "sub")
    outer.add_edge("sub", "wrap")
    outer.add_edge("wrap", END)
    result = outer.compile().invoke({"text": "hello"})
    print("== 子图当节点 ==")
    print(f"  {result['text']}")


# ---------- demo 2：输入输出收窄 ----------

def demo_schemas() -> None:
    """子图配 input_schema 与 output_schema，对外只露两个窄接口。"""
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

    outer = StateGraph(Overall)
    outer.add_node("sub", sub.compile())
    outer.add_edge(START, "sub")
    outer.add_edge("sub", END)
    result = outer.compile().invoke({"raw": "  padded  ", "clean": ""})
    print("== 输入输出收窄 ==")
    print(f"  子图吃 raw 得 clean: {result['clean']!r}")


# ---------- demo 3：同图移交 ----------

def demo_handoff() -> None:
    """节点返回 Command(goto=) 把控制权连同更新一起交出去。"""
    class Team(TypedDict):
        log: Annotated[list[str], add]
        current: str

    def agent_a(state: Team) -> Command[TypedDict("Targets", {"b": str})]:
        return Command(goto="b", update={"log": ["a 干完移交 b"]})

    def agent_b(state: Team):
        return {"log": ["b 接手完成"], "current": "b"}

    g = StateGraph(Team)
    g.add_node("a", agent_a)
    g.add_node("b", agent_b)
    g.add_edge(START, "a")
    g.add_edge("b", END)
    result = g.compile().invoke({"log": [], "current": ""})
    print("== 同图移交 ==")
    print(f"  {result['log']}")


# ---------- demo 4：PARENT 误用坑 ----------

def demo_parent_trap() -> None:
    """顶层图的节点返回 Command(graph=Command.PARENT) 没有父图可上浮。"""
    from langgraph.errors import ParentCommand

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
    print("== PARENT 误用坑 ==")
    try:
        g.compile().invoke({"log": [], "current": ""})
    except ParentCommand as e:
        print(f"  顶层图抛 {type(e).__name__}，没有父图接得住")
    print("  graph=Command.PARENT 只能写在子图节点里")


# ---------- demo 5：跨图导航 ----------

def demo_parent_navigate() -> None:
    """子图节点用 graph=Command.PARENT 直接跳到父图的节点。

    注意这里的节点函数没有写 Command[Literal[...]] 注解——编译期
    会校验注解目标必须在本图节点里，跨图目标只能不写注解。
    """
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
    result = outer.compile().invoke({"items": ["init"]})
    print("== 跨图导航 ==")
    print(f"  子图与父图的更新都算数: {result['items']}")


# ---------- demo 6：Supervisor ----------

def demo_supervisor() -> None:
    """主管节点决定下一个干活的，干完的把控制权交回主管。

    真实场景主管用模型的结构化输出决定 next_agent；demo 里用
    固定序列替代，路由机制完全一致。
    """
    class Flow(TypedDict):
        question: str
        steps: Annotated[list[str], add]
        answer: str

    decisions = ["research", "write", "__end__"]

    def supervisor(state: Flow):
        return Command(goto=decisions.pop(0))

    def research(state: Flow):
        return {"steps": ["查资料"], "answer": "资料显示"}

    def write(state: Flow):
        return {"steps": ["写稿"], "answer": state["answer"] + "，成稿"}

    g = StateGraph(Flow)
    g.add_node("supervisor", supervisor)
    g.add_node("research", research)
    g.add_node("write", write)
    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", END)
    for agent in ("research", "write"):
        g.add_edge(agent, "supervisor")
    result = g.compile().invoke({"question": "写个东西", "steps": [], "answer": ""})
    print("== Supervisor ==")
    print(f"  调度顺序 {result['steps']}，产出「{result['answer']}」")


# ---------- demo 7：Swarm 网络 ----------

def demo_swarm() -> None:
    """没有主管，agent 之间多对多，谁干完谁决定交给谁。"""
    class Net(TypedDict):
        log: Annotated[list[str], add]

    def agent_x(state: Net):
        return {"log": ["x 查完事实"]}

    def agent_y(state: Net):
        return {"log": ["y 写完初稿"]}

    def agent_z(state: Net) -> Command[TypedDict("T", {"y": str})]:
        # z 判断自己不擅长写稿，移交给 y
        return Command(goto="y", update={"log": ["z 改不了稿，移交 y"]})

    g = StateGraph(Net)
    g.add_node("x", agent_x)
    g.add_node("y", agent_y)
    g.add_node("z", agent_z)
    g.add_edge(START, "x")
    g.add_edge("x", "z")
    g.add_edge("z", "y")
    g.add_edge("y", END)
    result = g.compile().invoke({"log": []})
    print("== Swarm 网络 ==")
    for item in result["log"]:
        print(f"  · {item}")


# ---------- demo 8：可视化 ----------

def demo_draw() -> None:
    """get_graph().draw_mermaid() 生成文本图，零依赖零联网。"""
    g = StateGraph(TypedDict("S", {"text": str}))
    g.add_node("upper", lambda s: {"text": s["text"].upper()})
    g.add_edge(START, "upper")
    g.add_edge("upper", END)
    mermaid = g.compile().get_graph().draw_mermaid()
    print("== 可视化 ==")
    print("  mermaid 源码前四行：")
    for line in mermaid.splitlines()[:4]:
        print(f"    {line}")
    print("  draw_png 需要 pygraphviz，draw_mermaid_png 需要联网渲染")


# ---------- 入口 ----------

DEMOS = {
    "subgraph": demo_subgraph,
    "schemas": demo_schemas,
    "handoff": demo_handoff,
    "parent_trap": demo_parent_trap,
    "parent_navigate": demo_parent_navigate,
    "supervisor": demo_supervisor,
    "swarm": demo_swarm,
    "draw": demo_draw,
}


def main() -> None:
    import sys

    names = [n for n in sys.argv[1:] if n in DEMOS] or list(DEMOS)
    for name in names:
        DEMOS[name]()


if __name__ == "__main__":
    main()

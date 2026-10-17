"""第 8 篇示例：LangGraph 基础（hello_graph_deep.py）。

把「图怎么搭、状态怎么流」拆成九个 demo：静态边、Reducers 两种
语义、条件路由、多出边并行、Overwrite、add_messages、Command、
Send、递归上限。全部纯函数，不依赖任何模型与 API key。

第 1 篇的 hello_graph.py 跑通了单节点直线图，这一篇把图语言的
其余词汇补齐。
"""

from operator import add
from typing import Annotated, Literal, TypedDict

from langchain.messages import AIMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Overwrite, Send


# ---------- demo 1：静态边 ----------

def demo_basic() -> None:
    """节点做活，边定走向，一条直线图就是两次接力。"""
    class State(TypedDict):
        text: str

    def upper(state: State):
        return {"text": state["text"].upper()}

    def wrap(state: State):
        return {"text": f"[{state['text']}]"}

    g = StateGraph(State)
    g.add_node("upper", upper)
    g.add_node("wrap", wrap)
    g.add_edge(START, "upper")
    g.add_edge("upper", "wrap")
    g.add_edge("wrap", END)
    result = g.compile().invoke({"text": "hello"})
    print("== 静态边 ==")
    print(f"  {result['text']}")


# ---------- demo 2：Reducer 两种语义 ----------

def demo_reducers() -> None:
    """没标 reducer 的字段被覆盖，标了 Annotated 合并的两个才相加。"""
    class State(TypedDict):
        log: Annotated[list[str], add]
        last: str

    def node_a(state: State):
        return {"log": ["a"], "last": "a"}

    def node_b(state: State):
        return {"log": ["b"], "last": "b"}

    g = StateGraph(State)
    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    result = g.compile().invoke({"log": [], "last": ""})
    print("== Reducer ==")
    print(f"  log 合并成 {result['log']}，last 被覆盖成 {result['last']}")


# ---------- demo 3：条件路由 ----------

def demo_conditional() -> None:
    """add_conditional_edges 按状态决定下一跳，映射字典让返回值更好读。"""
    class State(TypedDict):
        n: int
        path: str

    def check(state: State) -> Literal["even", "odd"]:
        return "even" if state["n"] % 2 == 0 else "odd"

    def even(state: State):
        return {"path": "even"}

    def odd(state: State):
        return {"path": "odd"}

    g = StateGraph(State)
    g.add_node("check", lambda s: {})
    g.add_node("even", even)
    g.add_node("odd", odd)
    g.add_edge(START, "check")
    g.add_conditional_edges("check", check)
    g.add_edge("even", END)
    g.add_edge("odd", END)
    app = g.compile()
    print("== 条件路由 ==")
    print(f"  n=4 走 {app.invoke({'n': 4, 'path': ''})['path']}")
    print(f"  n=7 走 {app.invoke({'n': 7, 'path': ''})['path']}")


# ---------- demo 4：多出边并行 ----------

def demo_fanout() -> None:
    """一个节点多条出边，目标节点在同一个 super-step 并行执行。"""
    class State(TypedDict):
        log: Annotated[list[str], add]

    def fan(state: State):
        return {"log": ["fan"]}

    def x1(state: State):
        return {"log": ["x1"]}

    def x2(state: State):
        return {"log": ["x2"]}

    g = StateGraph(State)
    g.add_node("fan", fan)
    g.add_node("x1", x1)
    g.add_node("x2", x2)
    g.add_edge(START, "fan")
    g.add_edge("fan", "x1")
    g.add_edge("fan", "x2")
    g.add_edge("x1", END)
    g.add_edge("x2", END)
    result = g.compile().invoke({"log": []})
    print("== 多出边并行 ==")
    print(f"  执行了 {sorted(result['log'])}")


# ---------- demo 5：Overwrite 绕过 reducer ----------

def demo_overwrite() -> None:
    """合并型字段想清空，返回空列表没用，得用 Overwrite。"""
    class State(TypedDict):
        errors: Annotated[list[str], add]

    def node_a(state: State):
        return {"errors": ["bad sql"]}

    def node_clear(state: State):
        return {"errors": Overwrite([])}

    g = StateGraph(State)
    g.add_node("a", node_a)
    g.add_node("clear", node_clear)
    g.add_edge(START, "a")
    g.add_edge("a", "clear")
    g.add_edge("clear", END)
    result = g.compile().invoke({"errors": ["old"]})
    print("== Overwrite ==")
    print(f"  清空后 {result['errors']}")


# ---------- demo 6：add_messages 按 id 更新 ----------

def demo_messages_state() -> None:
    """add_messages 按消息 id 对账，同 id 是更新，新 id 是追加。"""
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def bot(state: State):
        return {"messages": [AIMessage(id="m1", content="revised answer")]}

    g = StateGraph(State)
    g.add_node("bot", bot)
    g.add_edge(START, "bot")
    g.add_edge("bot", END)
    result = g.compile().invoke(
        {"messages": [AIMessage(id="m1", content="original")]})
    print("== add_messages ==")
    print(f"  同 id 更新后: {result['messages'][0].content}")


# ---------- demo 7：Command 节点内路由 ----------

def demo_command() -> None:
    """Command 把状态更新和跳转写在一个返回值里，路由逻辑跟着节点走。"""
    class State(TypedDict):
        score: int
        where: str

    def judge(state: State) -> Command[Literal["pass", "fail"]]:
        if state["score"] >= 60:
            return Command(update={"where": "pass"}, goto="pass")
        return Command(update={"where": "fail"}, goto="fail")

    g = StateGraph(State)
    g.add_node("judge", judge)
    g.add_node("pass", lambda s: {})
    g.add_node("fail", lambda s: {})
    g.add_edge(START, "judge")
    g.add_edge("pass", END)
    g.add_edge("fail", END)
    app = g.compile()
    print("== Command ==")
    print(f"  score=90 -> {app.invoke({'score': 90, 'where': ''})['where']}")
    print(f"  score=50 -> {app.invoke({'score': 50, 'where': ''})['where']}")


# ---------- demo 8：Send map-reduce ----------

def demo_send() -> None:
    """条数未知就别画固定边，Send 把每个子任务连各自的 state 发出去。"""
    class State(TypedDict):
        subjects: list[str]
        jokes: Annotated[list[str], add]

    def dispatch(state: State):
        return [Send("make", {"subject": s}) for s in state["subjects"]]

    def make(state: dict):
        return {"jokes": [f"关于{state['subject']}的笑话"]}

    g = StateGraph(State)
    g.add_node("make", make)
    g.add_conditional_edges(START, dispatch, ["make"])
    g.add_edge("make", END)
    result = g.compile().invoke({"subjects": ["猫", "狗", "鸟"], "jokes": []})
    print("== Send map-reduce ==")
    print(f"  {sorted(result['jokes'])}")


# ---------- demo 9：递归上限 ----------

def demo_recursion() -> None:
    """循环图要有出口，recursion_limit 是兜底，GraphRecursionError 是哨声。"""
    class State(TypedDict):
        n: int

    def loop(state: State):
        return {"n": state["n"] + 1}

    g = StateGraph(State)
    g.add_node("loop", loop)
    g.add_edge(START, "loop")
    g.add_conditional_edges("loop", lambda s: "loop" if s["n"] < 3 else END)
    g.add_edge("loop", END)
    app = g.compile()
    print("== 递归上限 ==")
    print(f"  循环到 3 正常结束: n={app.invoke({'n': 0})['n']}")
    try:
        g.add_conditional_edges  # noqa: B018  复用 app 改不了图，重建一个死循环
        g2 = StateGraph(State)
        g2.add_node("loop", loop)
        g2.add_edge(START, "loop")
        g2.add_edge("loop", "loop")
        g2.add_edge("loop", END)
        g2.compile().invoke({"n": 0}, config={"recursion_limit": 5})
    except GraphRecursionError:
        print("  死循环被 recursion_limit=5 叫停")


# ---------- 入口 ----------

DEMOS = {
    "basic": demo_basic,
    "reducers": demo_reducers,
    "conditional": demo_conditional,
    "fanout": demo_fanout,
    "overwrite": demo_overwrite,
    "messages": demo_messages_state,
    "command": demo_command,
    "send": demo_send,
    "recursion": demo_recursion,
}


def main() -> None:
    import sys

    names = [n for n in sys.argv[1:] if n in DEMOS] or list(DEMOS)
    for name in names:
        DEMOS[name]()


if __name__ == "__main__":
    main()

"""第 1 篇示例：第一个 LangGraph 应用。

用 StateGraph 定义一个只有单个节点的图：

    START -> greet -> END

节点函数读取输入消息，返回一条 AI 消息；MessagesState 自带
add_messages 归约器，节点返回的消息会追加进消息历史。
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState


def build_graph():
    """构建并编译图：一个节点，一条直线。"""

    def greet(state: MessagesState):
        """节点函数：根据用户输入生成一句问候。"""
        user_text = state["messages"][-1].content
        reply = f"你好，我是运行在图上的第一个节点，收到了：{user_text}"
        return {"messages": [{"role": "ai", "content": reply}]}

    graph = StateGraph(MessagesState)
    graph.add_node("greet", greet)
    graph.add_edge(START, "greet")
    graph.add_edge("greet", END)
    return graph.compile()


def main() -> None:
    app = build_graph()
    result = app.invoke({"messages": [{"role": "user", "content": "很高兴认识你"}]})

    print("== 消息流 ==")
    for message in result["messages"]:
        print(f"[{message.type}] {message.content}")


if __name__ == "__main__":
    main()

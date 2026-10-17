"""第 7 篇示例：记忆系统（hello_memory.py）。

把「短期记对话、长期记事实」拆成九个 demo：短期记忆、线程隔离、
手动裁剪、自动摘要、Store 读写、工具里用 Store、跨会话记忆、
TTL 坑、持久化边界。

记忆组件全部来自 langgraph：InMemorySaver 管短期（线程内的对话
状态），InMemoryStore 管长期（跨线程的键值数据）。两者都是进程内
实现，重启即清空，正好用来看清各自的边界。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import AIMessage, HumanMessage, RemoveMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.store.memory import InMemoryStore

from fake_models import ScriptedChatModel


def scripted(*texts: str) -> ScriptedChatModel:
    """按顺序吐文本回复的替身模型。"""
    return ScriptedChatModel(messages=iter([AIMessage(content=t) for t in texts]))


# ---------- demo 1：短期记忆 ----------

def demo_short_term() -> None:
    """checkpointer 加 thread_id，同一个线程里的对话自动续上。"""
    agent = create_agent(
        model=scripted("你好小明", "你是小明，喜欢短句"),
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "chat-1"}}
    agent.invoke({"messages": [HumanMessage("我叫小明")]}, cfg)
    result = agent.invoke({"messages": [HumanMessage("我叫什么？")]}, cfg)
    print("== 短期记忆 ==")
    print(f"  线程内消息数: {len(result['messages'])}")
    print(f"  模型回答: {result['messages'][-1].content}")


# ---------- demo 2：线程隔离与状态查看 ----------

def demo_isolation() -> None:
    """不同 thread_id 各记各的账，get_state 能看线程状态快照。"""
    agent = create_agent(
        model=scripted("你好小明", "我不知道你叫什么"),
        checkpointer=InMemorySaver(),
    )
    cfg1 = {"configurable": {"thread_id": "chat-1"}}
    agent.invoke({"messages": [HumanMessage("我叫小明")]}, cfg1)

    cfg2 = {"configurable": {"thread_id": "chat-2"}}
    result = agent.invoke({"messages": [HumanMessage("我叫什么？")]}, cfg2)
    print("== 线程隔离 ==")
    print(f"  chat-2 的回答: {result['messages'][-1].content}")

    state = agent.get_state(cfg1)
    print(f"  chat-1 快照字段: {list(state.values.keys())}")
    print(f"  chat-1 消息数: {len(state.values['messages'])}")


# ---------- demo 3：手动裁剪历史 ----------

def demo_trim() -> None:
    """RemoveMessage 删旧消息，REMOVE_ALL_MESSAGES 一键清空再保留尾部。"""
    agent = create_agent(
        model=scripted("好的", "好的", "你是小明"),
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "chat-1"}}
    agent.invoke({"messages": [HumanMessage("我叫小明")]}, cfg)
    agent.invoke({"messages": [HumanMessage("我在学 Python")]}, cfg)
    before = len(agent.get_state(cfg).values["messages"])

    state = agent.get_state(cfg).values
    keep = [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + state["messages"][-2:]
    agent.update_state(cfg, {"messages": keep})
    after = agent.get_state(cfg).values["messages"]

    print("== 手动裁剪 ==")
    print(f"  裁剪前 {before} 条，裁剪后 {len(after)} 条")
    print(f"  保留的首条: {after[0].content}")


# ---------- demo 4：自动摘要 ----------

def demo_summarize() -> None:
    """SummarizationMiddleware 在消息数越线时请模型写摘要替换历史。"""
    main_model = ScriptedChatModel(messages=iter([
        AIMessage(content="你好小明"),
        AIMessage(content="好的"),
        AIMessage(content="你是小明，来自杭州"),
    ]))
    summarizer = ScriptedChatModel(messages=iter([
        AIMessage(content="用户叫小明，来自杭州，正在学 Python。"),
    ]))
    agent = create_agent(
        model=main_model,
        middleware=[SummarizationMiddleware(
            model=summarizer, trigger=("messages", 4), keep=("messages", 2),
        )],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "chat-1"}}
    agent.invoke({"messages": [HumanMessage("我叫小明，来自杭州")]}, cfg)
    agent.invoke({"messages": [HumanMessage("我在学 Python")]}, cfg)
    result = agent.invoke({"messages": [HumanMessage("我叫什么？")]}, cfg)

    history = result["messages"]
    print("== 自动摘要 ==")
    print(f"  触发摘要后消息数: {len(history)}")
    for m in history:
        print(f"  [{type(m).__name__}] {m.content[:30]}")


# ---------- demo 5：Store 键值读写 ----------

def demo_store_basics() -> None:
    """长期记忆是 namespace 加 key 组织的 JSON 文档，search 支持前缀与过滤。"""
    store = InMemoryStore()
    store.put(("users", "u1"), "profile", {"name": "小明", "city": "杭州"})
    store.put(("users", "u1"), "note", {"text": "喜欢美式咖啡"})
    store.put(("users", "u2"), "profile", {"name": "小红", "city": "上海"})

    item = store.get(("users", "u1"), "profile")
    print("== Store 读写 ==")
    print(f"  get: {item.value}（namespace={item.namespace}）")
    print(f"  精确空间检索: {[i.key for i in store.search(('users', 'u1'))]}")
    print(f"  前缀检索: {[i.key for i in store.search(('users',))]}")
    print(f"  内容过滤: {[i.value['name'] for i in store.search(('users',), filter={'city': '上海'})]}")
    print(f"  命名空间列表: {store.list_namespaces(prefix=('users',))}")


# ---------- demo 6：工具里读写长期记忆 ----------

def demo_tool_store() -> None:
    """工具拿到 ToolRuntime，runtime.store 就是 create_agent 传进来的那个。"""
    store = InMemoryStore()

    @tool
    def remember(fact: str, runtime: ToolRuntime) -> str:
        """记住一条关于用户的事实"""
        import hashlib
        key = hashlib.md5(fact.encode()).hexdigest()[:8]
        runtime.store.put(("memories",), key, {"text": fact})
        return "已记住"

    @tool
    def recall(runtime: ToolRuntime) -> str:
        """回忆用户告诉过你的事"""
        items = runtime.store.search(("memories",))
        return "；".join(i.value["text"] for i in items) or "还没有记忆"

    agent = create_agent(
        model=ScriptedChatModel(messages=iter([
            AIMessage(content="好的", tool_calls=[{
                "name": "remember", "args": {"fact": "用户喜欢短句回复"},
                "id": "call_1", "type": "tool_call",
            }]),
            AIMessage(content="我记住了"),
            AIMessage(content="好的", tool_calls=[{
                "name": "recall", "args": {},
                "id": "call_2", "type": "tool_call",
            }]),
            AIMessage(content="你喜欢短句回复"),
        ])),
        tools=[remember, recall],
        store=store,
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "chat-1"}}
    agent.invoke({"messages": [HumanMessage("记住我喜欢短句回复")]}, cfg)
    result = agent.invoke({"messages": [HumanMessage("我喜欢什么样的回复")]}, cfg)
    print("== 工具读写长期记忆 ==")
    print(f"  工具返回: {result['messages'][-2].content}")
    print(f"  模型回答: {result['messages'][-1].content}")
    print(f"  Store 里实际存了: {[i.value for i in store.search(('memories',))]}")


# ---------- demo 7：跨会话记忆 ----------

def demo_cross_session() -> None:
    """换一个全新 thread_id，短期记忆清零，Store 里的长期记忆还在。"""
    store = InMemoryStore()
    store.put(("memories",), "abc123", {"text": "用户喜欢短句回复"})

    @tool
    def recall(runtime: ToolRuntime) -> str:
        """回忆用户告诉过你的事"""
        items = runtime.store.search(("memories",))
        return "；".join(i.value["text"] for i in items) or "还没有记忆"

    agent = create_agent(
        model=ScriptedChatModel(messages=iter([
            AIMessage(content="好的", tool_calls=[{
                "name": "recall", "args": {},
                "id": "call_1", "type": "tool_call",
            }]),
            AIMessage(content="根据长期记忆，你喜欢短句回复"),
        ])),
        tools=[recall],
        store=store,
        checkpointer=InMemorySaver(),
    )
    fresh_cfg = {"configurable": {"thread_id": "brand-new"}}
    result = agent.invoke(
        {"messages": [HumanMessage("我喜欢什么样的回复")]}, fresh_cfg)
    print("== 跨会话记忆 ==")
    print(f"  全新线程消息数: {len(result['messages'])}")
    print(f"  工具返回: {result['messages'][-2].content}")
    print(f"  模型回答: {result['messages'][-1].content}")


# ---------- demo 8：TTL 坑 ----------

def demo_ttl_trap() -> None:
    """InMemoryStore 声称有 TTL 参数，实际一用就 NotImplementedError。"""
    store = InMemoryStore()
    print("== TTL 坑 ==")
    print(f"  put 签名里有 ttl 参数: True")
    print(f"  supports_ttl: {store.supports_ttl}")
    try:
        store.put(("t",), "k", {"v": 1}, ttl=5)
    except NotImplementedError as e:
        print(f"  put(ttl=5) 报错: {e}")


# ---------- demo 9：持久化边界 ----------

def demo_persist_gap() -> None:
    """InMemorySaver 与 InMemoryStore 都没有 dump/load，重启即失忆。"""
    from langgraph.checkpoint.memory import InMemorySaver

    print("== 持久化边界 ==")
    print(f"  InMemoryStore 有 dump: {hasattr(InMemoryStore(), 'dump')}")
    print(f"  InMemorySaver 有 dump: {hasattr(InMemorySaver(), 'dump')}")
    print("  进程一退全没了，要落盘换 SqliteSaver 或 PostgresStore")


# ---------- 入口 ----------

DEMOS = {
    "short_term": demo_short_term,
    "isolation": demo_isolation,
    "trim": demo_trim,
    "summarize": demo_summarize,
    "store": demo_store_basics,
    "tool_store": demo_tool_store,
    "cross_session": demo_cross_session,
    "ttl": demo_ttl_trap,
    "persist": demo_persist_gap,
}


def main() -> None:
    import sys

    names = sys.argv[1:] or list(DEMOS)
    for name in names:
        DEMOS[name]()


if __name__ == "__main__":
    main()

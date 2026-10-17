"""第 7 篇示例测试：记忆系统（hello_memory.py）。

全部用 ScriptedChatModel 驱动，不依赖任何 API key。
"""

import pytest
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.store.memory import InMemoryStore

import hello_memory
from assistant import MEM_NAMESPACE, recall_facts, remember_fact
from fake_models import ScriptedChatModel


def scripted(*texts):
    return ScriptedChatModel(
        messages=iter([AIMessage(content=t) for t in texts]))


def ai(text, name, args, cid):
    return AIMessage(content="好的", tool_calls=[{
        "name": name, "args": args, "id": cid, "type": "tool_call",
    }])


# ---------- 短期记忆 ----------

def test_same_thread_carries_context():
    """同一个 thread_id 连续两轮，模型靠历史认出用户。"""
    agent = create_agent(
        model=scripted("你好小明", "你是小明"),
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "t1"}}
    agent.invoke({"messages": [HumanMessage("我叫小明")]}, cfg)
    result = agent.invoke({"messages": [HumanMessage("我叫什么")]}, cfg)
    assert len(result["messages"]) == 4
    assert result["messages"][-1].content == "你是小明"


def test_threads_are_isolated():
    """换个 thread_id 历史清零，线程之间互不可见。"""
    agent = create_agent(
        model=scripted("你好小明", "我不认识你"),
        checkpointer=InMemorySaver(),
    )
    agent.invoke({"messages": [HumanMessage("我叫小明")]},
                 {"configurable": {"thread_id": "a"}})
    result = agent.invoke(
        {"messages": [HumanMessage("我叫什么")]},
        {"configurable": {"thread_id": "b"}})
    assert result["messages"][-1].content == "我不认识你"


def test_get_state_returns_snapshot():
    """get_state 拿到线程快照，values 里只有 messages。"""
    agent = create_agent(
        model=scripted("好的"),
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "t1"}}
    agent.invoke({"messages": [HumanMessage("hi")]}, cfg)
    state = agent.get_state(cfg)
    assert list(state.values.keys()) == ["messages"]
    assert len(state.values["messages"]) == 2


def test_update_state_trims_history():
    """RemoveMessage 加 REMOVE_ALL_MESSAGES 一键清空再保留尾部。"""
    agent = create_agent(
        model=scripted("好的", "好的"),
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "t1"}}
    agent.invoke({"messages": [HumanMessage("第一句")]}, cfg)
    agent.invoke({"messages": [HumanMessage("第二句")]}, cfg)
    state = agent.get_state(cfg).values
    keep = [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + state["messages"][-2:]
    agent.update_state(cfg, {"messages": keep})
    messages = agent.get_state(cfg).values["messages"]
    assert len(messages) == 2
    assert messages[0].content == "第二句"


def test_summarization_replaces_old_history():
    """消息越线触发摘要，老历史被替换成摘要消息。"""
    main_model = ScriptedChatModel(messages=iter([
        AIMessage(content="你好小明"),
        AIMessage(content="好的"),
        AIMessage(content="你是小明，来自杭州"),
    ]))
    summarizer = ScriptedChatModel(messages=iter([
        AIMessage(content="用户叫小明，来自杭州。"),
    ]))
    agent = hello_memory.create_agent(
        model=main_model,
        middleware=[hello_memory.SummarizationMiddleware(
            model=summarizer, trigger=("messages", 4), keep=("messages", 2))],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "t1"}}
    agent.invoke({"messages": [HumanMessage("我叫小明，来自杭州")]}, cfg)
    agent.invoke({"messages": [HumanMessage("我在学 Python")]}, cfg)
    result = agent.invoke({"messages": [HumanMessage("我叫什么")]}, cfg)

    history = result["messages"]
    assert len(history) == 4
    joined = "".join(str(m.content) for m in history)
    assert "我叫小明，来自杭州" not in joined
    assert "summary" in joined.lower()


# ---------- Store 长期记忆 ----------

def test_store_put_get_roundtrip():
    """put 存什么 get 拿什么，namespace 和 key 原样带回。"""
    store = InMemoryStore()
    store.put(("users", "u1"), "profile", {"name": "小明"})
    item = store.get(("users", "u1"), "profile")
    assert item.value == {"name": "小明"}
    assert item.key == "profile"
    assert item.namespace == ("users", "u1")


def test_store_search_prefix_matches():
    """search 只给前缀时，把子命名空间全部翻出来。"""
    store = InMemoryStore()
    store.put(("users", "u1"), "a", {"v": 1})
    store.put(("users", "u2"), "b", {"v": 2})
    keys = sorted(i.key for i in store.search(("users",)))
    assert keys == ["a", "b"]
    assert sorted(store.list_namespaces(prefix=("users",))) == [
        ("users", "u1"), ("users", "u2")]


def test_store_filter_on_content():
    """filter 按内容等值过滤，跨命名空间也能筛。"""
    store = InMemoryStore()
    store.put(("users", "u1"), "p", {"city": "杭州"})
    store.put(("users", "u2"), "p", {"city": "上海"})
    hits = store.search(("users",), filter={"city": "上海"})
    assert len(hits) == 1
    assert hits[0].namespace == ("users", "u2")


def test_store_query_without_index_is_silently_ignored():
    """没配 embedding 时 search 的 query 被静默忽略，不报错。"""
    store = InMemoryStore()
    store.put(("m",), "x", {"text": "喜欢咖啡"})
    store.put(("m",), "y", {"text": "在学 Python"})
    hits = store.search(("m",), query="完全无关的查询")
    assert len(hits) == 2


def test_tool_writes_store():
    """工具经 runtime.store 写入的数据，外部直接可查。"""
    store = InMemoryStore()
    agent = create_agent(
        model=ScriptedChatModel(messages=iter([
            ai("好的", "remember_fact", {"fact": "喜欢短句"}, "c1"),
            AIMessage(content="记住了"),
        ])),
        tools=[remember_fact],
        store=store,
    )
    agent.invoke({"messages": [HumanMessage("记住我喜欢短句")]})
    items = store.search(MEM_NAMESPACE)
    assert [i.value["text"] for i in items] == ["喜欢短句"]


def test_tool_reads_store():
    """工具经 runtime.store 读回外部预置的数据。"""
    store = InMemoryStore()
    store.put(MEM_NAMESPACE, "k1", {"text": "喜欢咖啡"})
    agent = create_agent(
        model=ScriptedChatModel(messages=iter([
            ai("好的", "recall_facts", {}, "c1"),
            AIMessage(content="你喜欢咖啡"),
        ])),
        tools=[remember_fact, recall_facts],
        store=store,
    )
    result = agent.invoke(
        {"messages": [HumanMessage("我爱喝什么")]})
    assert result["messages"][-2].content == "喜欢咖啡"
    assert result["messages"][-1].content == "你喜欢咖啡"


def test_new_thread_reads_store_written_elsewhere():
    """全新线程没有历史，但 Store 里的长期记忆照样读得到。"""
    store = InMemoryStore()
    store.put(MEM_NAMESPACE, "k1", {"text": "喜欢短句"})
    agent = create_agent(
        model=ScriptedChatModel(messages=iter([
            ai("好的", "recall_facts", {}, "c1"),
            AIMessage(content="你喜欢短句"),
        ])),
        tools=[remember_fact, recall_facts],
        store=store,
        checkpointer=InMemorySaver(),
    )
    result = agent.invoke(
        {"messages": [HumanMessage("我喜欢什么")]},
        {"configurable": {"thread_id": "brand-new"}})
    assert len(result["messages"]) == 4
    assert result["messages"][-2].content == "喜欢短句"


# ---------- 坑位回归 ----------

def test_ttl_not_supported_in_memory_store():
    """InMemoryStore 的 put 有 ttl 形参，一用就 NotImplementedError。"""
    store = InMemoryStore()
    with pytest.raises(NotImplementedError, match="TTL"):
        store.put(("t",), "k", {"v": 1}, ttl=5)


def test_memory_components_cannot_dump():
    """InMemorySaver 与 InMemoryStore 都没有 dump/load，重启即失忆。"""
    assert not hasattr(InMemoryStore(), "dump")
    assert not hasattr(InMemoryStore(), "load")
    assert not hasattr(InMemorySaver(), "dump")

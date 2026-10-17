"""第 7 篇完整案例测试：带长期记忆的个人助理（assistant.py）。

全部用 ScriptedChatModel 驱动，不依赖任何 API key。
"""

from langchain.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

import assistant
from assistant import (
    DEMO_STEPS,
    MEM_NAMESPACE,
    build_agent,
    fact_key,
    print_mems,
    recall_facts,
    remember_fact,
    run_session,
)
from fake_models import ScriptedChatModel


def ai(name, args, cid):
    return AIMessage(content="好的", tool_calls=[{
        "name": name, "args": args, "id": cid, "type": "tool_call",
    }])


# ---------- 记忆键与工具 ----------

def test_fact_key_is_deterministic():
    """同一条事实永远得到同一个 key，重复记住等于更新。"""
    assert fact_key("用户喜欢美式咖啡") == fact_key("用户喜欢美式咖啡")
    assert fact_key("a") != fact_key("b")


def test_remember_then_recall_roundtrip(capsys):
    """remember_fact 写进的记忆，recall_facts 能原样读回。"""
    store = InMemoryStore()
    agent = build_agent(
        ScriptedChatModel(messages=iter([
            ai("remember_fact", {"fact": "用户喜欢美式咖啡"}, "c1"),
            AIMessage(content="好的", tool_calls=[{
                "name": "recall_facts", "args": {},
                "id": "c2", "type": "tool_call",
            }]),
            AIMessage(content="你喜欢美式咖啡"),
        ])),
        store, None,
    )
    result = agent.invoke({"messages": [HumanMessage("记住我喜欢美式")]})
    assert result["messages"][-2].content == "用户喜欢美式咖啡"
    assert result["messages"][-1].content == "你喜欢美式咖啡"


def test_remember_same_fact_overwrites():
    """重复记同一条事实只留一条，key 相同互相覆盖。"""
    store = InMemoryStore()
    agent = build_agent(
        ScriptedChatModel(messages=iter([
            ai("remember_fact", {"fact": "喜欢美式"}, "c1"),
            AIMessage(content="好的"),
            ai("remember_fact", {"fact": "喜欢美式"}, "c3"),
            AIMessage(content="好的"),
        ])),
        store, None,
    )
    agent.invoke({"messages": [HumanMessage("第一遍")]})
    agent.invoke({"messages": [HumanMessage("第二遍")]})
    assert len(store.search(MEM_NAMESPACE)) == 1


# ---------- 会话与演示 ----------

def test_demo_sessions_share_memory(capsys):
    """会话 A 记两笔，会话 B 换新线程照样回忆出来。"""
    store = InMemoryStore()
    run_session(store, "session-a", DEMO_STEPS["remember"],
                "记住：我喜欢美式咖啡")
    out = capsys.readouterr().out
    assert "已记住：用户喜欢美式咖啡" in out

    run_session(store, "session-b", DEMO_STEPS["recall"],
                "我爱喝什么？")
    out = capsys.readouterr().out
    assert "工具返回: 用户喜欢美式咖啡；用户在学 LangChain" in out
    assert "你喜欢美式咖啡" in out


def test_demo_steps_cover_both_tools():
    """演示脚本覆盖两个工具，字段齐全可复放。"""
    remember_steps = [m for m in DEMO_STEPS["remember"] if m.tool_calls]
    recall_steps = [m for m in DEMO_STEPS["recall"] if m.tool_calls]
    assert {m.tool_calls[0]["name"] for m in remember_steps} == {"remember_fact"}
    assert recall_steps[0].tool_calls[0]["name"] == "recall_facts"


def test_print_mems_lists_all(capsys):
    """print_mems 把库里的每条记忆连 key 一起打出来。"""
    store = InMemoryStore()
    store.put(MEM_NAMESPACE, "k1", {"text": "喜欢美式"})
    store.put(MEM_NAMESPACE, "k2", {"text": "在学 LangChain"})
    print_mems(store)
    out = capsys.readouterr().out
    assert "[k1] 喜欢美式" in out
    assert "[k2] 在学 LangChain" in out


def test_recall_empty_store_message():
    """记忆库为空时 recall_facts 说人话，不炸不糊弄。"""
    store = InMemoryStore()
    agent = build_agent(
        ScriptedChatModel(messages=iter([
            ai("recall_facts", {}, "c1"),
            AIMessage(content="你还没告诉过我任何偏好"),
        ])),
        store, None,
    )
    result = agent.invoke({"messages": [HumanMessage("我爱喝什么")]})
    assert result["messages"][-2].content == "（记忆库是空的）"

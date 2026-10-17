"""第 7 篇完整案例：带长期记忆的个人助理（assistant.py）。

把本篇的记忆件件用上：
- InMemorySaver 做短期记忆，thread_id 圈住一次会话的对话历史；
- InMemoryStore 做长期记忆，remember_fact 与 recall_facts 两个工具
  经 runtime.store 存取用户的偏好事实，namespace 按用户分组；
- --mems 直接翻看记忆库，--new 开一个全新会话演示「对话忘了、
  偏好还在」。

用法：
    uv run python assistant.py --list
    uv run python assistant.py --demo          # 一键演示记忆闭环
    uv run python assistant.py --mems          # 翻看长期记忆库
    uv run python assistant.py "我喜欢喝美式" --model openai:gpt-5.5

模型来源与前几篇一致：--model 或环境变量 AGENT_MODEL 指定时用真实
模型；都没指定时用 ScriptedChatModel 本地演示（交互走 --demo）。
"""

import argparse
import hashlib
import os

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from fake_models import ScriptedChatModel

# ---------- 长期记忆的存放约定 ----------

MEM_NAMESPACE = ("memories", "default-user")


def fact_key(fact: str) -> str:
    """同一条事实永远映射到同一个 key，重复记住等于更新。"""
    return hashlib.md5(fact.encode()).hexdigest()[:8]


@tool
def remember_fact(fact: str, runtime: ToolRuntime) -> str:
    """把用户告诉你的长期偏好或事实写进记忆库。

    Args:
        fact: 一句话事实，例如「用户喜欢美式咖啡」
    """
    runtime.store.put(MEM_NAMESPACE, fact_key(fact), {"text": fact})
    return f"已记住：{fact}"


@tool
def recall_facts(runtime: ToolRuntime) -> str:
    """回忆记忆库里关于用户的所有事实"""
    items = runtime.store.search(MEM_NAMESPACE)
    if not items:
        return "（记忆库是空的）"
    return "；".join(i.value["text"] for i in items)


# ---------- 组装 ----------

def build_agent(model, store, checkpointer):
    """短期靠 checkpointer，长期靠 store，工具在两者之间搭桥。"""
    return create_agent(
        model=model,
        tools=[remember_fact, recall_facts],
        store=store,
        checkpointer=checkpointer,
        system_prompt=(
            "你是有长期记忆的助理。用户说出偏好或事实时，先调 remember_fact 存档；"
            "被问到自己的偏好或说过的话时，先调 recall_facts 再回答。回答不超过两句话。"
        ),
    )


def run_session(store, thread_id, scripted_steps, user_input):
    """跑一个会话并打印完整轨迹，返回最终回复。"""
    model = ScriptedChatModel(messages=iter(scripted_steps))
    agent = build_agent(model, store, InMemorySaver())
    cfg = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(user_input)]}, cfg)
    print(f"[会话 {thread_id}] 用户: {user_input}")
    for m in result["messages"][1:]:
        if getattr(m, "tool_calls", None):
            for c in m.tool_calls:
                print(f"    工具调用: {c['name']}({c['args']})")
        elif m.type == "tool":
            print(f"    工具返回: {m.content}")
        else:
            print(f"    助理: {m.content}")
    return result


# ---------- 一键演示 ----------

DEMO_STEPS = {
    "remember": [
        AIMessage(content="好的", tool_calls=[{
            "name": "remember_fact",
            "args": {"fact": "用户喜欢美式咖啡"},
            "id": "call_1", "type": "tool_call",
        }]),
        AIMessage(content="好的", tool_calls=[{
            "name": "remember_fact",
            "args": {"fact": "用户在学 LangChain"},
            "id": "call_2", "type": "tool_call",
        }]),
        AIMessage(content="都记下了，美式咖啡和在学 LangChain。"),
    ],
    "recall": [
        AIMessage(content="好的", tool_calls=[{
            "name": "recall_facts", "args": {},
            "id": "call_1", "type": "tool_call",
        }]),
        AIMessage(content="你喜欢美式咖啡，最近在学 LangChain。"),
    ],
}


def run_demo() -> None:
    """会话 A 记两笔，换全新线程开 session B，看它还记得多少。"""
    store = InMemoryStore()

    run_session(store, "session-a", DEMO_STEPS["remember"],
                "记住：我喜欢美式咖啡，我在学 LangChain")
    print()
    run_session(store, "session-b", DEMO_STEPS["recall"],
                "我爱喝什么？最近在忙什么？")
    print()
    print_mems(store)


def print_mems(store) -> None:
    print("长期记忆库:")
    for item in store.search(MEM_NAMESPACE):
        print(f"  [{item.key}] {item.value['text']}")


# ---------- 模型与 CLI ----------

def build_model(model_name=None):
    """约定同前几篇：有名字用真实模型，没有就用脚本替身。"""
    name = model_name or os.environ.get("AGENT_MODEL")
    if name:
        from langchain.chat_models import init_chat_model

        params = {}
        if os.environ.get("YI_BASE_URL"):
            params["base_url"] = os.environ["YI_BASE_URL"]
        if os.environ.get("YI_API_KEY"):
            params["api_key"] = os.environ["YI_API_KEY"]
        return init_chat_model(name, **params)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="带长期记忆的个人助理")
    parser.add_argument("question", nargs="?", help="对助理说一句话")
    parser.add_argument("--list", action="store_true", help="打印能力清单")
    parser.add_argument("--demo", action="store_true", help="一键演示记忆闭环")
    parser.add_argument("--mems", action="store_true", help="翻看长期记忆库")
    parser.add_argument("--thread", default="main", help="会话线程 ID")
    parser.add_argument("--model", help="提供商:模型名，例如 openai:gpt-5.5")
    cli = parser.parse_args()

    if cli.list:
        print("工具: remember_fact, recall_facts")
        print("短期记忆: InMemorySaver（thread_id 圈会话）")
        print("长期记忆: InMemoryStore（namespace=memories/default-user）")
        return
    if cli.demo:
        run_demo()
        return
    if cli.mems:
        print_mems(InMemoryStore())
        print("（进程内存储，重启后为空；单轮 --mems 演示的是空库长相）")
        return
    if not cli.question:
        parser.error("给一句话，或者用 --demo / --mems / --list")

    store = InMemoryStore()
    model = build_model(cli.model)
    name = cli.model or os.environ.get("AGENT_MODEL")
    if model is None:
        print("未指定模型，单轮对话需要真实模型；试试 --demo")
        return
    print(f"模型: {name}")
    agent = build_agent(model, store, InMemorySaver())
    result = agent.invoke(
        {"messages": [HumanMessage(cli.question)]},
        {"configurable": {"thread_id": cli.thread}},
    )
    print(f"你: {cli.question}")
    print(f"助理: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()

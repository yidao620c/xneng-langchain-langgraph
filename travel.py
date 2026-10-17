"""第 12 篇完整案例：个人旅游助手（travel.py）。

把系列里学过的件拼成最后一个完整项目：

- 三个信息工具，天气、景点攻略都是本地模拟数据，行为确定，演示
  和测试不花钱；真实项目里把函数体换成外部 API 调用即可；
- 长期记忆走 InMemoryStore，偏好记一次，跨会话都记得（第 7 篇的
  模式）；
- 行程规划是 create_agent，模型自己决定先查什么后查什么，查完把
  行程交给 save_itinerary 工具落盘成 markdown 文件；
- 工具调用全程在 messages 里留痕，哪一步查了什么一清二楚。

用法：
    uv run python travel.py plan "十一去杭州玩两天"      # 无 key 走替身
    uv run python travel.py plan "..." --model openai:deepseek-v4.1-flash
    uv run python travel.py mems                        # 看记了什么偏好
    uv run python travel.py --trace                     # 看完整协作轨迹

行程落盘在 source/travel_plans/，模型与嵌入来源约定同前两篇。
"""

import argparse
import os
from pathlib import Path

from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain.tools import ToolRuntime, tool
from langgraph.store.memory import InMemoryStore

from fake_models import ScriptedChatModel

MEM_NAMESPACE = ("travel", "default-user")
PLAN_DIR = Path(__file__).parent / "travel_plans"


# ---------- 本地模拟数据（真实项目换成外部 API） ----------

WEATHER = {
    "杭州": [("第一天", "多云", "22", "东南风 2 级"),
             ("第二天", "小雨", "19", "东风 3 级")],
    "北京": [("第一天", "晴", "18", "西北风 2 级"),
             ("第二天", "晴", "17", "北风 1 级")],
}

ATTRACTIONS = [
    {"city": "杭州", "name": "西湖", "tag": "山水",
     "note": "环湖步行三小时，断桥到雷峰塔一线最经典，雨天有雾景"},
    {"city": "杭州", "name": "灵隐寺", "tag": "人文",
     "note": "上午人少，飞来峰石刻值得慢看，寺内素面可以当午饭"},
    {"city": "杭州", "name": "中国丝绸博物馆", "tag": "室内",
     "note": "免费，雨天首选，玉皇山脚下逛完可顺路爬一段山"},
    {"city": "北京", "name": "故宫", "tag": "人文",
     "note": "必须提前七天约票，中轴线走完再进珍宝馆"},
    {"city": "北京", "name": "颐和园", "tag": "山水",
     "note": "下午光线最好，昆明湖东岸走到十七孔桥约两小时"},
]


# ---------- 工具（模型可调） ----------

@tool
def get_weather(city: str) -> str:
    """查一个城市未来两天的天气预报，返回日期、天气、气温与风向。"""
    forecast = WEATHER.get(city)
    if not forecast:
        return f"没有 {city} 的天气数据，现有城市：{'、'.join(WEATHER)}"
    return "\n".join(f"{d}：{w}，{t}℃，{wind}" for d, w, t, wind in forecast)


@tool
def search_attractions(city: str, tag: str = "") -> str:
    """按城市查景点攻略，可用 tag 过滤（山水/人文/室内），返回名称与玩法笔记。"""
    hits = [a for a in ATTRACTIONS
            if a["city"] == city and (not tag or a["tag"] == tag)]
    if not hits:
        return f"没有找到 {city} 的{tag or ''}景点记录"
    return "\n".join(f"{a['name']}（{a['tag']}）：{a['note']}" for a in hits)


@tool
def remember_preference(key: str, value: str, runtime: ToolRuntime) -> str:
    """把用户的旅行偏好写进长期记忆，例如 不吃辣、喜欢博物馆、预算两千。"""
    runtime.store.put(MEM_NAMESPACE, key, {"key": key, "value": value})
    return f"已记住偏好 {key}={value}"


@tool
def recall_preferences(runtime: ToolRuntime) -> str:
    """取回全部旅行偏好，规划行程前先看一眼。"""
    items = runtime.store.search(MEM_NAMESPACE)
    if not items:
        return "（还没有任何偏好记录）"
    prefs = [f"{i.value.get('key', '?')}={i.value.get('value', '?')}"
             for i in items if "key" in i.value]
    return "\n".join(prefs) if prefs else "（还没有任何偏好记录）"


@tool
def save_itinerary(city: str, text: str, runtime: ToolRuntime) -> str:
    """把最终行程写成 markdown 文件存进 travel_plans 目录，并在长期记忆留档。"""
    PLAN_DIR.mkdir(exist_ok=True)
    path = PLAN_DIR / f"{city}.md"
    path.write_text(text, encoding="utf-8")
    runtime.store.put(MEM_NAMESPACE, f"last_plan_{city}", {"text": text[:80]})
    return f"行程已保存到 {path.name}，并在长期记忆留档"


TOOLS = [get_weather, search_attractions,
         remember_preference, recall_preferences, save_itinerary]


# ---------- 组装 ----------

def build_model():
    name = os.environ.get("AGENT_MODEL")
    if name:
        from langchain.chat_models import init_chat_model

        params = {}
        if os.environ.get("YI_BASE_URL"):
            params["base_url"] = os.environ["YI_BASE_URL"]
        if os.environ.get("YI_API_KEY"):
            params["api_key"] = os.environ["YI_API_KEY"]
        return init_chat_model(name, **params)
    return None


def fake_model() -> ScriptedChatModel:
    """无 key 替身，四步脚本走完整条工具链（工具都是真执行）。"""
    return ScriptedChatModel(messages=iter([
        AIMessage(content="", tool_calls=[
            {"name": "recall_preferences", "args": {}, "id": "t1", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[
            {"name": "get_weather", "args": {"city": "杭州"}, "id": "t2", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[
            {"name": "search_attractions", "args": {"city": "杭州"}, "id": "t3", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[
            {"name": "save_itinerary", "args": {
                "city": "杭州",
                "text": "# 杭州两日游\n\nDay1 西湖环湖\nDay2 灵隐寺上午、丝绸博物馆下午（雨天备选）"},
             "id": "t4", "type": "tool_call"}]),
        AIMessage(content="杭州两日行程已生成并存档。Day1 走西湖环湖线，"
                          "Day2 上午灵隐寺、下午丝绸博物馆（第二天下雨，正好用室内备选）。"
                          "行程文件在 travel_plans/杭州.md，偏好记得下次直接用。"),
    ]))


def build_agent(store: InMemoryStore, model):
    return create_agent(model, tools=TOOLS, store=store, debug=False)


def print_trace(result) -> int:
    """打印协作轨迹，返回工具调用轮数。"""
    rounds = 0
    for m in result["messages"]:
        name = type(m).__name__
        if name == "AIMessage" and m.tool_calls:
            rounds += 1
            calls = ", ".join(
                f"{c['name']}({str(c['args'])[:40]})" for c in m.tool_calls)
            print(f"  [{name}] 调用 {calls}")
        elif name == "ToolMessage":
            content = m.content if isinstance(m.content, str) else str(m.content)
            print(f"      = {content.replace(chr(10), ' / ')[:72]}")
        else:
            text = m.content if isinstance(m.content, str) else ""
            print(f"  [{name}] {text[:60]}")
    return rounds


# ---------- CLI ----------

def cmd_plan(request: str, model_name=None, trace=False) -> None:
    store = InMemoryStore()
    model = build_model() if not model_name else None
    if model_name:
        os.environ["AGENT_MODEL"] = model_name
        model = build_model()
    if model is None:
        model = fake_model()
        mode = "替身模型（四步脚本，工具真执行）"
    else:
        mode = f"真实模型 {os.environ.get('AGENT_MODEL')}"
    print(f"旅游助手启动，{mode}")

    agent = build_agent(store, model)
    result = agent.invoke({"messages": [request]})

    if trace:
        print("协作轨迹:")
        rounds = print_trace(result)
        print(f"（共 {rounds} 轮工具调用）")

    answer = result["messages"][-1].content
    if not isinstance(answer, str):
        answer = str(answer)
    print()
    print(answer)
    plans = sorted(p.name for p in PLAN_DIR.glob("*.md")) if PLAN_DIR.exists() else []
    if plans:
        print()
        print(f"已存行程: {', '.join(plans)}")


def cmd_mems() -> None:
    store = InMemoryStore()
    items = store.search(MEM_NAMESPACE)
    if not items:
        print("（本进程的长期记忆为空，InMemoryStore 不跨进程，接真实模型跑一轮即有记录）")
        return
    for i in items:
        print(f"  {i.key} -> {i.value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="个人旅游助手：查天气攻略、记偏好、出行程")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="规划行程")
    p_plan.add_argument("request", help="需求描述，例如 十一去杭州玩两天")
    p_plan.add_argument("--model", help="提供商:模型名，例如 openai:deepseek-v4.1-flash")
    p_plan.add_argument("--trace", action="store_true", help="打印完整协作轨迹")

    sub.add_parser("mems", help="查看长期记忆")

    cli = parser.parse_args()
    if cli.cmd == "plan":
        cmd_plan(cli.request, cli.model, cli.trace)
    elif cli.cmd == "mems":
        cmd_mems()


if __name__ == "__main__":
    main()

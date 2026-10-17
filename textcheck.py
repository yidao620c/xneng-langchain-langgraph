"""第 8 篇完整案例：文本体检流水线（textcheck.py）。

把本篇的图语言件件用上：
- clean 先清洗文本，detect 按中英文占比条件路由；
- 中英文各走各的统计节点，产出写进同一个 reducer 合并的 stats；
- summary 汇总成报告，空文本用 Command 直接跳 END 提前收工；
- --trace 用 updates 流式看图每一步把状态改成了什么。

用法：
    uv run python textcheck.py --list
    uv run python textcheck.py "LangGraph 让工作流变成一张能看见的图"
    uv run python textcheck.py "..." --trace
    echo "" | uv run python textcheck.py  # 空输入看 Command 提前收工
"""

import argparse
import re
import sys
from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class State(TypedDict):
    """流水线的共享状态，stats 用 reducer 合并并行结果。"""
    text: str
    cleaned: str
    lang: str
    stats: Annotated[list[str], add]
    report: str


# ---------- 节点 ----------

def clean(state: State):
    """清洗：压平空白、去首尾。"""
    text = re.sub(r"\s+", " ", state["text"]).strip()
    return {"cleaned": text}


def detect(state: State):
    """判断语言：中文字符占比过半算中文。"""
    text = state["cleaned"]
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    lang = "zh" if zh * 2 >= len(text.replace(" ", "")) else "en"
    return {"lang": lang}


def zh_count(state: State):
    """中文统计：字数、字频前五、句子数。"""
    text = state["cleaned"]
    chars = [c for c in text if re.match(r"[\u4e00-\u9fff]", c)]
    freq: dict[str, int] = {}
    for c in chars:
        freq[c] = freq.get(c, 0) + 1
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:5]
    top_text = "、".join(f"{c}x{n}" for c, n in top) or "无"
    sentences = len([s for s in re.split(r"[。！？]", text) if s.strip()])
    return {"stats": [f"中文字数 {len(chars)}", f"字频前五 {top_text}",
                      f"句子数 {sentences}"]}


def en_count(state: State):
    """英文统计：词数、词频前五、句子数。"""
    text = state["cleaned"].lower()
    words = re.findall(r"[a-z']+", text)
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:5]
    top_text = ", ".join(f"{w}x{n}" for w, n in top) or "none"
    sentences = len([s for s in re.split(r"[.!?]", text) if s.strip()])
    return {"stats": [f"english words {len(words)}", f"top words {top_text}",
                      f"sentences {sentences}"]}


def summary(state: State) -> Command[Literal[END]]:
    """汇总报告；空文本没得体检，Command 直接跳 END。"""
    if not state["cleaned"]:
        return Command(update={"report": "空文本，没什么可体检的。"}, goto=END)
    lines = [f"语言判定 {state['lang']}"] + list(state["stats"])
    body = "\n".join(f"  · {line}" for line in lines)
    return {"report": f"文本体检报告\n{body}"}


# ---------- 组图 ----------

def build_graph():
    """clean -> detect 条件路由中英文，结果汇进 summary。"""
    g = StateGraph(State)
    g.add_node("clean", clean)
    g.add_node("detect", detect)
    g.add_node("zh_count", zh_count)
    g.add_node("en_count", en_count)
    g.add_node("summary", summary)

    g.add_edge(START, "clean")
    g.add_edge("clean", "detect")

    def route_lang(state: State) -> Literal["zh_count", "en_count", "summary"]:
        if not state["cleaned"]:
            return "summary"
        return "zh_count" if state["lang"] == "zh" else "en_count"

    g.add_conditional_edges("detect", route_lang)
    g.add_edge("zh_count", "summary")
    g.add_edge("en_count", "summary")
    return g.compile()


def print_trace(chunks) -> dict:
    """按 updates 流打印每步轨迹，同时按 reducer 语义累积出最终状态。"""
    state: dict = {}
    for chunk in chunks:
        for node, update in chunk.items():
            if node == "__end__":
                continue
            update = update or {}
            fields = ", ".join(f"{k}={str(v)[:44]}" for k, v in update.items()
                               if k != "stats")
            print(f"  [{node}] {fields}".rstrip())
            for s in update.get("stats") or []:
                print(f"      + stats: {s}")
            for k, v in update.items():
                if k == "stats" and k in state:
                    state[k] = state[k] + v
                else:
                    state[k] = v
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="文本体检流水线")
    parser.add_argument("text", nargs="?",
                        help="要体检的文本，留空则从 stdin 读")
    parser.add_argument("--list", action="store_true", help="打印图的节点清单")
    parser.add_argument("--trace", action="store_true", help="流式看每步状态")
    cli = parser.parse_args()

    if cli.list:
        print("节点: clean, detect, zh_count, en_count, summary")
        print("路由: detect --条件边--> zh_count / en_count；summary 空文本 Command 跳 END")
        return

    text = cli.text if cli.text is not None else sys.stdin.read()
    app = build_graph()

    if cli.trace:
        print("执行轨迹:")
        result = print_trace(app.stream({"text": text}, stream_mode="updates"))
    else:
        result = app.invoke({"text": text})
    print()
    print(result["report"])


if __name__ == "__main__":
    main()

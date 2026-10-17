"""第 11 篇完整案例：个人知识库（library.py）。

第 6 篇的 kb.py 是雏形，检索几条拼进提示词，模型照单全收。这一篇
把它升级成 Agent 版：

- ingest 文档导入管道，按「文件名#块号」生成确定性 id，重复导入
  幂等，内容更新自动替换旧块；
- ask 走 create_agent，模型自己决定检索几次、用什么查询词，不再是
  固定 top-k 一锤子买卖；
- 回答里的 [来源#块号] 标记会被解析成溯源清单，每个引用都能翻回
  库内的原文块。

用法：
    uv run python library.py ingest notes/        # 导入文档或目录
    uv run python library.py list                 # 库存清单
    uv run python library.py stats                # 统计
    uv run python library.py ask "LangChain 怎么装"
    uv run python library.py ask "..." --show-sources   # 展开引用原文

模型与嵌入来源约定同 kb.py：YI 环境变量可用时走真实模型与嵌入
（AGENT_MODEL / EMBED_MODEL），都没有时用本地替身，全程不依赖 API。
"""

import argparse
import os
import re
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fake_models import HashEmbeddings, ScriptedChatModel
from langchain.messages import AIMessage

STORE_FILE = Path(__file__).parent / "kb_store.json"
SUFFIXES = {".txt", ".md"}
CITE = re.compile(r"\[([^\[\]#]+#[0-9]+)\]")

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=30,
    separators=["\n\n", "\n", "。", "，"],
)


# ---------- 模型与嵌入 ----------

def build_embeddings():
    """有 YI 环境变量走真实嵌入，否则哈希替身。"""
    if os.environ.get("YI_API_KEY") and os.environ.get("YI_BASE_URL"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
            openai_api_base=os.environ["YI_BASE_URL"],
            openai_api_key=os.environ["YI_API_KEY"],
            check_embedding_ctx_length=False,
        )
    return HashEmbeddings(size=64)


def load_store():
    emb = build_embeddings()
    if STORE_FILE.exists():
        return InMemoryVectorStore.load(str(STORE_FILE), embedding=emb)
    return InMemoryVectorStore(embedding=emb)


def build_model():
    """AGENT_MODEL 可用时走真实模型，否则本地替身（由 ask 动态编排脚本）。"""
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


# ---------- 文档导入管道 ----------

def load_docs(path: Path) -> list[Document]:
    """把一个 txt/md 文件或目录变成 Document 列表，来源写进 metadata。"""
    if path.is_file():
        files = [path] if path.suffix.lower() in SUFFIXES else []
    else:
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in SUFFIXES)
    return [Document(page_content=f.read_text(encoding="utf-8"),
                     metadata={"source": f.name}) for f in files]


def chunk_ids_of(store, source: str) -> list[str]:
    """找出库里某文件现存的全部块 id（id 规则就是 文件名#块号）。"""
    return [item["id"] for item in store.store.values()
            if item["metadata"].get("source") == source]


def ingest(paths: list[str]) -> None:
    """加载、分割、替换式导入。同文件重导不重复，内容变了旧块自动清掉。"""
    store = load_store()
    raw_docs = []
    for p in paths:
        raw_docs.extend(load_docs(Path(p)))
    if not raw_docs:
        print("没有找到可导入的 txt/md 文件")
        return

    total_old = total_new = 0
    for doc in raw_docs:
        source = doc.metadata["source"]
        old = chunk_ids_of(store, source)
        if old:
            store.delete(old)               # 先清这个文件的旧块
            total_old += len(old)
        chunks = SPLITTER.split_documents([doc])
        for i, c in enumerate(chunks):
            c.metadata["chunk"] = i
        store.add_documents(chunks, ids=[f"{source}#{i}" for i in range(len(chunks))])
        total_new += len(chunks)
        print(f"  {source}: 旧块 {len(old)} -> 新块 {len(chunks)}")
    store.dump(str(STORE_FILE))
    print(f"导入完成，替换 {total_old} 块，写入 {total_new} 块，"
          f"库内现有 {len(store.store)} 块")


# ---------- 检索工具与 Agent ----------

def format_hits(hits) -> str:
    """检索结果统一格式化，带 [文件名#块号] 来源标记供模型引用。"""
    if not hits:
        return "（没有检索到相关资料）"
    return "\n\n".join(
        f"[{d.metadata['source']}#{d.metadata['chunk']}] {d.page_content}"
        for d, _ in hits)


def build_agent(store, model):
    """create_agent + 检索工具。模型自己决定查什么、查几次。"""
    @tool
    def search_notes(query: str, k: int = 3) -> str:
        """在个人笔记知识库里检索相关段落。

        返回带 [文件名#块号] 来源标记的资料，回答时引用这些标记。
        """
        return format_hits(store.similarity_search_with_score(query, k=k))

    return create_agent(model, tools=[search_notes], debug=False)


# ---------- 提问与溯源 ----------

def parse_citations(text: str) -> list[str]:
    """从回答里按出现顺序抽出引用标记，去重保序。"""
    seen, cites = set(), []
    for m in CITE.finditer(text):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            cites.append(m.group(1))
    return cites


def source_trace(store, cites: list[str]) -> list[tuple[str, str]]:
    """引用标记 -> 库内原文，翻得出原文的引用才算有效溯源。"""
    trace = []
    for cite in cites:
        item = store.store.get(cite)
        trace.append((cite, item["text"] if item else "（库内找不到该块）"))
    return trace


def fake_model_for(question: str, store) -> ScriptedChatModel:
    """无 key 时的替身：先按问题检索一次，再引用真实检索结果作答。"""
    probe = store.similarity_search_with_score(question, k=3)
    cites = [f"{d.metadata['source']}#{d.metadata['chunk']}" for d, _ in probe]
    body = "、".join(f"[{c}]" for c in cites[:2]) or "资料中没有直接相关的内容"
    return ScriptedChatModel(messages=iter([
        AIMessage(content="", tool_calls=[
            {"name": "search_notes", "args": {"query": question, "k": 3},
             "id": "t1", "type": "tool_call"}]),
        AIMessage(content=f"依据笔记，{question} 的要点整理如下，出处 {body}。"
                          "细节见引用的原文块。"),
    ]))


def ask(question: str, show_sources: bool = False) -> None:
    """Agent 问答，回答解析引用标记，给出溯源清单。"""
    store = load_store()
    if not store.store:
        print("知识库还是空的，先跑 ingest")
        return

    model = build_model() or fake_model_for(question, store)
    agent = build_agent(store, model)
    result = agent.invoke({"messages": [question]})
    answer = result["messages"][-1].content
    if not isinstance(answer, str):
        answer = str(answer)

    print(f"问题: {question}")
    print(f"回答: {answer}")

    cites = parse_citations(answer)
    tool_rounds = sum(1 for m in result["messages"]
                      if type(m).__name__ == "AIMessage" and m.tool_calls)
    print(f"（模型检索了 {tool_rounds} 轮，回答带 {len(cites)} 处引用）")

    if cites:
        print("引用溯源:")
        for cite, text in source_trace(store, cites):
            print(f"  {cite} -> {text.replace(chr(10), ' ')[:44]}")
            if show_sources:
                print(f"      原文: {text}")


# ---------- 查看与统计 ----------

def list_store() -> None:
    """列出库内每一块的来源与开头。"""
    store = load_store()
    if not store.store:
        print("知识库还是空的，先跑 ingest")
        return
    print(f"库内共 {len(store.store)} 块")
    for i, item in enumerate(store.store.values(), 1):
        head = item["text"].replace("\n", " ")[:38]
        print(f"  {i}. [{item['metadata'].get('source', '?')}#"
              f"{item['metadata'].get('chunk', '?')}] {head}")


def stats() -> None:
    """按文件统计块数，导入管道的体检表。"""
    store = load_store()
    if not store.store:
        print("知识库还是空的，先跑 ingest")
        return
    by_source: dict[str, int] = {}
    for item in store.store.values():
        src = item["metadata"].get("source", "?")
        by_source[src] = by_source.get(src, 0) + 1
    print(f"文件 {len(by_source)} 个，共 {len(store.store)} 块")
    for src, n in sorted(by_source.items()):
        print(f"  {src}: {n} 块")


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="个人知识库：导入文档、Agent 问答、引用溯源")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="导入文件或目录（txt/md），重复导入幂等")
    p_ingest.add_argument("paths", nargs="+")

    sub.add_parser("list", help="列出库内条目")
    sub.add_parser("stats", help="按文件统计块数")

    p_ask = sub.add_parser("ask", help="Agent 问答，回答带引用溯源")
    p_ask.add_argument("question")
    p_ask.add_argument("--show-sources", action="store_true", help="展开引用块的原文")

    cli = parser.parse_args()
    if cli.cmd == "ingest":
        ingest(cli.paths)
    elif cli.cmd == "list":
        list_store()
    elif cli.cmd == "stats":
        stats()
    elif cli.cmd == "ask":
        ask(cli.question, cli.show_sources)


if __name__ == "__main__":
    main()

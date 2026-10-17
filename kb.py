"""第 6 篇完整案例：个人知识库雏形（kb.py）。

把九个 demo 装成一个能用的命令行小工具：把 markdown 与 txt 笔记吃进
本地向量库，之后随时提问，它检索相关段落再让模型只依据料回答，
回答附带出处。

用法：
    uv run python kb.py ingest notes/            # 吃进整个目录
    uv run python kb.py ingest notes/plan.md     # 吃进单个文件
    uv run python kb.py list                     # 看库里有什么
    uv run python kb.py ask "LangChain 怎么装"   # 检索并回答
    uv run python kb.py ask "..." --k 3          # 多检索几条

模型与嵌入来源约定同 hello_rag.py：YI 环境变量可用时走真实嵌入
（EMBED_MODEL，默认 text-embedding-3-small）与真实模型（AGENT_MODEL）；
都没有时用本地替身，全程不依赖 API。
"""

import argparse
import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fake_models import HashEmbeddings, ScriptedChatModel
from langchain.messages import AIMessage

STORE_FILE = Path(__file__).parent / "kb_store.json"


# ---------- 模型与嵌入 ----------

def build_embeddings():
    """约定同 hello_rag.py：有 YI 环境变量走真实嵌入，否则哈希替身。"""
    if os.environ.get("YI_API_KEY") and os.environ.get("YI_BASE_URL"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
            openai_api_base=os.environ["YI_BASE_URL"],
            openai_api_key=os.environ["YI_API_KEY"],
            check_embedding_ctx_length=False,
        )
    return HashEmbeddings(size=64)


def build_model(model_name=None):
    """聊天模型：AGENT_MODEL 或 --model 指定时走真实模型，否则本地替身。"""
    name = model_name or os.environ.get("AGENT_MODEL")
    if name:
        from langchain.chat_models import init_chat_model

        params = {}
        if os.environ.get("YI_BASE_URL"):
            params["base_url"] = os.environ["YI_BASE_URL"]
        if os.environ.get("YI_API_KEY"):
            params["api_key"] = os.environ["YI_API_KEY"]
        return init_chat_model(name, **params)
    return ScriptedChatModel(
        messages=iter([AIMessage(content="根据笔记，安装 LangChain 用 pip install langchain，需要 Python 3.9 以上版本。")])
    )


# ---------- 加载与分割 ----------

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=30,
    separators=["\n\n", "\n", "。", "，"],
)


def load_docs(path: Path) -> list[Document]:
    """把一个 txt/md 文件变成带来源元数据的 Document。"""
    suffixes = {".txt", ".md"}
    if path.is_file():
        files = [path] if path.suffix.lower() in suffixes else []
    else:
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in suffixes)
    docs = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": f.name}))
    return docs


def ingest(paths: list[str]) -> None:
    """加载、分割、并入向量库，落盘。"""
    emb = build_embeddings()
    if STORE_FILE.exists():
        store = InMemoryVectorStore.load(str(STORE_FILE), embedding=emb)
    else:
        store = InMemoryVectorStore(embedding=emb)

    raw_docs = []
    for p in paths:
        raw_docs.extend(load_docs(Path(p)))
    if not raw_docs:
        print("没有找到可吃的 txt/md 文件")
        return

    chunks = SPLITTER.split_documents(raw_docs)
    ids = store.add_documents(chunks)
    store.dump(str(STORE_FILE))
    print(f"吃进 {len(raw_docs)} 个文件，切成 {len(chunks)} 块，库里现在有 {len(store.store)} 块")


# ---------- 查看 ----------

def list_store() -> None:
    """列出库内每一块的来源与开头。

    InMemoryVectorStore.store 的元素是 {id, vector, text, metadata} 字典。
    """
    if not STORE_FILE.exists():
        print("知识库还是空的，先跑 ingest")
        return
    store = InMemoryVectorStore.load(str(STORE_FILE), embedding=build_embeddings())
    print(f"库内共 {len(store.store)} 块")
    for i, item in enumerate(store.store.values(), 1):
        head = item["text"].replace("\n", " ")[:38]
        print(f"  {i}. [{item['metadata'].get('source', '?')}] {head}")


# ---------- 提问 ----------

def ask(question: str, k: int, model_name=None) -> None:
    """检索 top-k 段落拼进提示词，模型只依据料回答，附出处。"""
    if not STORE_FILE.exists():
        print("知识库还是空的，先跑 ingest")
        return
    store = InMemoryVectorStore.load(str(STORE_FILE), embedding=build_embeddings())

    hits = store.similarity_search_with_score(question, k=k)
    if not hits:
        print("库里没有内容可检索")
        return

    context = "\n\n".join(
        f"[{d.metadata.get('source', '?')}] {d.page_content}" for d, _ in hits
    )
    prompt = (
        "只依据下面的资料回答问题，资料里没有的信息就直说不知道。\n\n"
        f"资料：\n{context}\n\n问题：{question}"
    )
    model = build_model(model_name)
    answer = model.invoke([SystemMessage("你是简洁的问答助手"), HumanMessage(prompt)])
    answer_text = answer.content if isinstance(answer.content, str) else str(answer.content)

    print(f"问题: {question}")
    print(f"检索到 {len(hits)} 块:")
    for d, score in hits:
        print(f"  ({score:.3f}) [{d.metadata.get('source', '?')}] "
              f"{d.page_content.replace(chr(10), ' ')[:32]}")
    print(f"回答: {answer_text}")


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="个人知识库雏形：吃笔记、查库存、检索问答")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="吃进文件或目录（txt/md）")
    p_ingest.add_argument("paths", nargs="+", help="文件或目录路径")

    sub.add_parser("list", help="列出库内条目")

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question", help="问题")
    p_ask.add_argument("--k", type=int, default=2, help="检索条数，默认 2")
    p_ask.add_argument("--model", help="提供商:模型名，例如 openai:gpt-5.5")

    cli = parser.parse_args()
    if cli.cmd == "ingest":
        ingest(cli.paths)
    elif cli.cmd == "list":
        list_store()
    elif cli.cmd == "ask":
        ask(cli.question, cli.k, cli.model)


if __name__ == "__main__":
    main()

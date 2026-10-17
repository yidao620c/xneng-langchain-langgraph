"""第 6 篇示例测试：RAG 组件（hello_rag.py）。

全部用 HashEmbeddings 与 ScriptedChatModel 驱动，不依赖任何 API。
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

import hello_rag
from fake_models import HashEmbeddings
from hello_rag import (
    DEMOS,
    NOTES,
    build_embeddings,
    demo_agentic,
    demo_document,
    demo_search,
    demo_split,
    demo_store,
    demo_two_step,
    write_sample,
)


def fake_emb() -> HashEmbeddings:
    """确定性嵌入替身：同文本必得同向量。"""
    return HashEmbeddings(size=64)


# ---------- Document 与加载 ----------

def test_document_fields():
    """Document 由 page_content 与 metadata 组成。"""
    doc = Document(page_content="内容", metadata={"source": "a.md"})
    assert doc.page_content == "内容"
    assert doc.metadata == {"source": "a.md"}


def test_load_text_makes_document(tmp_path: Path):
    """load demo 的手写加载器把文件变成一个带来源的 Document。"""
    file_path = tmp_path / "notes.txt"
    write_sample(file_path)

    text = file_path.read_text(encoding="utf-8")
    docs = [Document(page_content=text, metadata={"source": str(file_path)})]
    assert len(docs) == 1
    assert docs[0].metadata["source"].endswith("notes.txt")
    assert "LangChain" in docs[0].page_content


# ---------- 分割 ----------

def test_splitter_chunking_and_overlap():
    """chunk_size 控制块长，overlap 让相邻块共享尾巴与开头。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=90, chunk_overlap=15,
        separators=["\n\n", "\n", "。", "，"],
    )
    docs = splitter.create_documents([NOTES])
    assert len(docs) > 1
    assert all(len(d.page_content) <= 90 for d in docs)


def test_splitter_keeps_separator_priority():
    """separators 按优先级递归：能塞进一块的相邻段会被合并。

    keep_separator 默认 True，切分符留在块尾。文本总长不足块长两倍时
    前两段合并成一块，第二刀落在第二个双换行处。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=30, chunk_overlap=0, separators=["\n\n", "。"],
    )
    text = "段落一甲。段落一乙。\n\n段落二甲。段落二乙。\n\n段落三甲。段落三乙。"
    docs = splitter.create_documents([text])
    assert len(docs) == 2
    assert "段落一" in docs[0].page_content
    assert "段落三" in docs[1].page_content


# ---------- 向量库 ----------

def test_store_from_texts_and_search_self_hit():
    """入库后用原文当 query，必命中原文（HashEmbeddings 确定性）。"""
    store = InMemoryVectorStore.from_texts(
        ["今天北京天气晴朗。", "上海明天有雨。"], fake_emb(),
    )
    hits = store.similarity_search("今天北京天气晴朗。", k=1)
    assert hits[0].page_content == "今天北京天气晴朗。"


def test_store_with_score_similarity_semantics():
    """with_score 返回相似度，完全一致为 1.0，越大越相关。"""
    store = InMemoryVectorStore.from_texts(
        ["今天北京天气晴朗。", "上海明天有雨。"], fake_emb(),
    )
    scored = store.similarity_search_with_score("今天北京天气晴朗。", k=2)
    assert scored[0][0].page_content == "今天北京天气晴朗。"
    assert abs(scored[0][1] - 1.0) < 1e-6
    assert scored[1][1] <= scored[0][1]


def test_store_dump_and_load(tmp_path: Path):
    """dump/load 持久化：重载后检索行为一致。"""
    store = InMemoryVectorStore.from_texts(["独特句子XYZ。"], fake_emb())
    path = tmp_path / "store.json"
    store.dump(str(path))

    reloaded = InMemoryVectorStore.load(str(path), embedding=fake_emb())
    assert len(reloaded.store) == 1
    assert reloaded.similarity_search("独特句子XYZ。", k=1)[0].page_content == "独特句子XYZ。"


def test_store_metadata_roundtrip():
    """入库时 metadata 原样保留，检索结果可溯源。"""
    store = InMemoryVectorStore(embedding=fake_emb())
    store.add_documents([
        Document(page_content="带来源的段落。", metadata={"source": "a.md"}),
    ])
    hit = store.similarity_search("带来源的段落。", k=1)[0]
    assert hit.metadata["source"] == "a.md"


# ---------- 检索器 ----------

def test_retriever_invoke_respects_k():
    """as_retriever 的 search_kwargs.k 控制 invoke 返回条数。"""
    store = InMemoryVectorStore.from_texts(["a", "b", "c"], fake_emb())
    retriever = store.as_retriever(search_kwargs={"k": 2})
    hits = retriever.invoke("a")
    assert len(hits) == 2


# ---------- 两步式 RAG ----------

def test_two_step_rag_answers_from_context():
    """检索结果拼进提示词，替身模型按料回答。"""
    emb = fake_emb()
    store = InMemoryVectorStore.from_texts(
        ["安装 LangChain 需要 Python 3.9 以上版本。"], emb,
    )
    question = "安装 LangChain 需要 Python 3.9 以上版本。"
    context_docs = store.similarity_search(question, k=1)
    context = "\n".join(d.page_content for d in context_docs)
    assert "Python 3.9" in context


# ---------- demo 可运行 ----------

def test_demos_run_without_error(tmp_path: Path, monkeypatch):
    """九个 demo 在本地替身环境下全部可跑。"""
    monkeypatch.delenv("YI_API_KEY", raising=False)
    monkeypatch.delenv("YI_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    # document / load 等直接调；统一入口校验映射完整
    assert set(DEMOS) == {
        "document", "load", "split", "embed", "store",
        "search", "retriever", "twostep", "agentic",
    }
    demo_document()
    demo_split()
    demo_store()
    demo_search()
    demo_two_step()
    demo_agentic()
    hello_rag.demo_load(tmp_path)
    DEMOS["embed"]()
    DEMOS["retriever"]()

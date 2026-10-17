"""第 6 篇完整案例测试：个人知识库雏形（kb.py）。

全部用 HashEmbeddings 与 ScriptedChatModel 驱动，不依赖任何 API。
"""

import json
from pathlib import Path

import kb
from kb import STORE_FILE, SPLITTER, load_docs
from fake_models import HashEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


def clean_store(monkeypatch, tmp_path: Path) -> Path:
    """把 kb 的落盘位置换到测试目录，避免污染真实库文件。"""
    target = tmp_path / "kb_store.json"
    monkeypatch.setattr(kb, "STORE_FILE", target)
    return target


def seed_store(path: Path) -> None:
    """预置一个两块内容的知识库文件。"""
    emb = HashEmbeddings(size=64)
    store = InMemoryVectorStore(embedding=emb)
    store.add_documents([
        Document(page_content="安装 LangChain 用 pip install langchain，需要 Python 3.9 以上版本。",
                 metadata={"source": "basics.md"}),
        Document(page_content="LangGraph 用节点和边组织流程，条件边让流程按状态分叉。",
                 metadata={"source": "graph.md"}),
    ])
    store.dump(str(path))


# ---------- 加载与分割 ----------

def test_load_docs_single_file(tmp_path: Path):
    """单个 md 文件变成一个带来源的 Document。"""
    f = tmp_path / "note.md"
    f.write_text("# 标题\n\n正文内容。", encoding="utf-8")
    docs = load_docs(f)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "note.md"


def test_load_docs_directory_filters_suffixes(tmp_path: Path):
    """目录模式只收 txt/md，别的扩展名跳过。"""
    (tmp_path / "a.md").write_text("markdown", encoding="utf-8")
    (tmp_path / "b.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "c.py").write_text("print(1)", encoding="utf-8")
    docs = load_docs(tmp_path)
    assert sorted(d.metadata["source"] for d in docs) == ["a.md", "b.txt"]


def test_splitter_params_from_kb():
    """kb 的分割器 chunk_size=200 overlap=30。"""
    assert SPLITTER._chunk_size == 200
    assert SPLITTER._chunk_overlap == 30


# ---------- 落盘与重载 ----------

def test_seed_store_roundtrip(tmp_path: Path):
    """dump 出的文件重载后内容与元数据都在。"""
    path = tmp_path / "kb.json"
    seed_store(path)
    store = InMemoryVectorStore.load(str(path), embedding=HashEmbeddings(size=64))
    assert len(store.store) == 2
    texts = {item["text"] for item in store.store.values()}
    assert any("pip install langchain" in t for t in texts)
    sources = {item["metadata"]["source"] for item in store.store.values()}
    assert sources == {"basics.md", "graph.md"}


def test_dump_file_is_json(tmp_path: Path):
    """kb_store.json 是可读 JSON，顶层键是块 id。"""
    path = tmp_path / "kb.json"
    seed_store(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 2
    first = data[next(iter(data))]
    assert set(first.keys()) == {"id", "vector", "text", "metadata"}


# ---------- ingest / list / ask 流程 ----------

def test_ingest_creates_store(monkeypatch, tmp_path: Path):
    """ingest 之后落盘文件出现，块数与来源正确。"""
    target = clean_store(monkeypatch, tmp_path)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("第一篇笔记讲 LangChain 安装。" * 3, encoding="utf-8")

    kb.ingest([str(notes)])
    assert target.exists()
    store = InMemoryVectorStore.load(str(target), embedding=HashEmbeddings(size=64))
    assert len(store.store) >= 1
    assert all(item["metadata"]["source"] == "a.md" for item in store.store.values())


def test_ingest_without_files(monkeypatch, tmp_path: Path, capsys):
    """路径里没有 txt/md 时给友好提示，不落盘。"""
    target = clean_store(monkeypatch, tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    kb.ingest([str(empty)])
    assert "没有找到" in capsys.readouterr().out
    assert not target.exists()


def test_list_store_empty(monkeypatch, tmp_path: Path, capsys):
    """库不存在时 list 给出先 ingest 的提示。"""
    clean_store(monkeypatch, tmp_path)
    kb.list_store()
    assert "还是空的" in capsys.readouterr().out


def test_list_store_entries(monkeypatch, tmp_path: Path, capsys):
    """list 输出每块的来源与开头。"""
    clean_store(monkeypatch, tmp_path)
    seed_store(tmp_path / "kb_store.json")
    kb.list_store()
    out = capsys.readouterr().out
    assert "basics.md" in out
    assert "graph.md" in out


def test_ask_retrieves_and_answers(monkeypatch, tmp_path: Path, capsys):
    """ask 检索 top-k 并回答，出处跟问题相关的那块排前面。"""
    monkeypatch.delenv("YI_API_KEY", raising=False)
    monkeypatch.delenv("YI_BASE_URL", raising=False)
    clean_store(monkeypatch, tmp_path)
    seed_store(tmp_path / "kb_store.json")

    # 用库内原文当问题，HashEmbeddings 下该块相似度必然 1.0 排第一
    kb.ask("安装 LangChain 用 pip install langchain，需要 Python 3.9 以上版本。", k=2)
    out = capsys.readouterr().out
    assert "basics.md" in out
    assert "回答:" in out


def test_ask_without_store(monkeypatch, tmp_path: Path, capsys):
    """库不存在时 ask 给出先 ingest 的提示。"""
    clean_store(monkeypatch, tmp_path)
    kb.ask("任何问题", k=2)
    assert "还是空的" in capsys.readouterr().out


def test_build_embeddings_local_fallback(monkeypatch):
    """没有 YI 环境变量时回退到 HashEmbeddings。"""
    monkeypatch.delenv("YI_API_KEY", raising=False)
    monkeypatch.delenv("YI_BASE_URL", raising=False)
    emb = kb.build_embeddings()
    assert isinstance(emb, HashEmbeddings)

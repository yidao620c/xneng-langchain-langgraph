"""第 11 篇完整案例测试：个人知识库（library.py）。"""

import library


# ---------- 固件 ----------

def _isolated(tmp_path, monkeypatch):
    """隔离环境：重定向库文件 + 清掉 YI 变量强制走本地替身。"""
    monkeypatch.setattr(library, "STORE_FILE", tmp_path / "store.json")
    for k in ("YI_API_KEY", "YI_BASE_URL", "AGENT_MODEL"):
        monkeypatch.delenv(k, raising=False)


def _ingest_texts(monkeypatch, texts: dict[str, str]):
    """绕过文件系统直接导入内存文档，返回库。"""
    store = library.load_store()
    for source, content in texts.items():
        doc = library.Document(page_content=content, metadata={"source": source})
        chunks = library.SPLITTER.split_documents([doc])
        for i, c in enumerate(chunks):
            c.metadata["chunk"] = i
        store.add_documents(chunks,
                            ids=[f"{source}#{i}" for i in range(len(chunks))])
    store.dump(str(library.STORE_FILE))
    return store


# ---------- 文档导入管道 ----------

def test_ingest_idempotent(tmp_path, monkeypatch):
    """同一目录导入两次，块数不翻倍。"""
    _isolated(tmp_path, monkeypatch)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("第一段内容，讲安装。", encoding="utf-8")

    library.ingest([str(notes)])
    store = library.load_store()
    assert len(store.store) == 1

    library.ingest([str(notes)])
    store = library.load_store()
    assert len(store.store) == 1


def test_ingest_replaces_stale_chunks(tmp_path, monkeypatch):
    """文件内容更新导致块数变少时，旧的多余块被清掉。"""
    _isolated(tmp_path, monkeypatch)
    f = tmp_path / "b.md"
    para = "这是一段足够长的笔记内容，详细记录某个主题的方方面面，用来撑过分割器的块大小阈值，保证能切成多个块。"
    f.write_text("\n\n".join(para + str(i) for i in range(8)), encoding="utf-8")
    library.ingest([str(f)])
    first_count = len(library.load_store().store)
    assert first_count > 1

    f.write_text("重写后的短内容", encoding="utf-8")
    library.ingest([str(f)])
    store = library.load_store()
    assert len(store.store) == 1
    ids = [i for i in store.store]
    assert ids == ["b.md#0"]


def test_chunks_carry_source_and_index(tmp_path, monkeypatch):
    """每块的 metadata 带来源文件名与块号，id 规则一致。"""
    _isolated(tmp_path, monkeypatch)
    long = tmp_path / "long.md"
    long.write_text("段一。\n\n段二。\n\n段三。", encoding="utf-8")
    library.ingest([str(long)])
    store = library.load_store()
    for item in store.store.values():
        assert item["metadata"]["source"] == "long.md"
        assert isinstance(item["metadata"]["chunk"], int)
        assert item["id"] == f"long.md#{item['metadata']['chunk']}"


# ---------- 引用解析与溯源 ----------

def test_parse_citations_dedup_keep_order():
    """引用标记按出现顺序去重。"""
    text = "先看 [a.md#0]，再对比 [b.md#2]，最后回到 [a.md#0]。"
    assert library.parse_citations(text) == ["a.md#0", "b.md#2"]


def test_parse_citations_ignores_plain_brackets():
    """普通方括号不带 #块号 的不算引用。"""
    assert library.parse_citations("参考 [链接] 和 [a.md#0]") == ["a.md#0"]


def test_source_trace_hits_store(tmp_path, monkeypatch):
    """有效引用翻回库内原文，无效引用标注找不到。"""
    _isolated(tmp_path, monkeypatch)
    store = _ingest_texts(monkeypatch, {"x.md": "笔记内容在这里"})
    trace = library.source_trace(store, ["x.md#0", "ghost.md#9"])
    assert trace[0] == ("x.md#0", "笔记内容在这里")
    assert "找不到" in trace[1][1]


# ---------- Agent 问答 ----------

def test_ask_with_fake_agent(tmp_path, monkeypatch, capsys):
    """无 key 走替身 agent，回答带引用，溯源翻得回原文。"""
    _isolated(tmp_path, monkeypatch)
    _ingest_texts(monkeypatch, {
        "install.md": "LangChain 用 pip install langchain 安装。",
        "vector.md": "向量库存文本的嵌入表示。",
    })
    library.ask("LangChain 怎么安装？")
    out = capsys.readouterr().out
    assert "回答:" in out
    assert "引用溯源:" in out
    assert "[install.md#0]" in out or "[vector.md#0]" in out
    assert "（模型检索了 1 轮" in out


def test_ask_multi_round_tool_calls(tmp_path, monkeypatch, capsys):
    """模型连续检索两轮的脚本，检索轮数如实计数。"""
    _isolated(tmp_path, monkeypatch)
    store = _ingest_texts(monkeypatch, {
        "install.md": "pip install langchain",
        "vector.md": "向量库存嵌入",
    })
    scripted = library.ScriptedChatModel(messages=iter([
        library.AIMessage(content="", tool_calls=[
            {"name": "search_notes", "args": {"query": "安装", "k": 3},
             "id": "t1", "type": "tool_call"}]),
        library.AIMessage(content="", tool_calls=[
            {"name": "search_notes", "args": {"query": "向量库", "k": 3},
             "id": "t2", "type": "tool_call"}]),
        library.AIMessage(content="结合 [install.md#0] 与 [vector.md#0]。"),
    ]))
    agent = library.build_agent(store, scripted)
    result = agent.invoke({"messages": ["对比一下"]})
    rounds = sum(1 for m in result["messages"]
                 if type(m).__name__ == "AIMessage" and m.tool_calls)
    assert rounds == 2
    assert "[install.md#0]" in result["messages"][-1].content


def test_format_hits_carries_citations(tmp_path, monkeypatch):
    """检索结果格式化带 [来源#块号] 标记，空结果有兜底文案。"""
    _isolated(tmp_path, monkeypatch)
    store = _ingest_texts(monkeypatch, {"s.md": "要被检索到的内容"})
    hits = store.similarity_search_with_score("要被检索到", k=3)
    assert library.format_hits(hits).startswith("[s.md#0]")
    assert library.format_hits([]) == "（没有检索到相关资料）"


def test_ask_empty_store(tmp_path, monkeypatch, capsys):
    """空库提问给出提示，不抛异常。"""
    _isolated(tmp_path, monkeypatch)
    library.ask("任何问题")
    assert "还是空的" in capsys.readouterr().out


# ---------- 统计 ----------

def test_stats_counts_by_source(tmp_path, monkeypatch, capsys):
    """stats 按文件聚合块数。"""
    _isolated(tmp_path, monkeypatch)
    _ingest_texts(monkeypatch, {"a.md": "内容 A", "b.md": "内容 B"})
    library.stats()
    out = capsys.readouterr().out
    assert "a.md: 1 块" in out and "b.md: 1 块" in out

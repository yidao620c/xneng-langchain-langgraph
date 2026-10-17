"""第 8 篇完整案例测试：文本体检流水线（textcheck.py）。"""

from textcheck import State, build_graph, clean, detect, en_count, summary, zh_count


# ---------- 节点单测 ----------

def test_clean_collapses_whitespace():
    """clean 压平空白并去首尾。"""
    out = clean({"text": "  a   b\n\nc  "})
    assert out["cleaned"] == "a b c"


def test_detect_routes_by_ratio():
    """中文字符占比过半判中文，否则英文。"""
    assert detect({"cleaned": "今天天气不错"})["lang"] == "zh"
    assert detect({"cleaned": "hello world today"})["lang"] == "en"


def test_zh_count_stats_shape():
    """中文统计产出三条，含字数与字频。"""
    out = zh_count({"cleaned": "测试测试文本。"})
    assert len(out["stats"]) == 3
    assert out["stats"][0] == "中文字数 6"


def test_en_count_stats_shape():
    """英文统计产出三条，含词数与词频。"""
    out = en_count({"cleaned": "hello hello world"})
    assert len(out["stats"]) == 3
    assert out["stats"][0] == "english words 3"


# ---------- 图级行为 ----------

def test_zh_input_flows_through_zh_branch():
    """中文输入走 zh_count 分支，报告由 summary 汇出。"""
    report = build_graph().invoke({"text": " 图状态    是共享的。"})["report"]
    assert "语言判定 zh" in report
    assert "中文字数" in report


def test_en_input_flows_through_en_branch():
    """英文输入走 en_count 分支。"""
    report = build_graph().invoke({"text": "Nodes do the work."})["report"]
    assert "语言判定 en" in report
    assert "english words" in report


def test_empty_text_short_circuits_via_command():
    """空文本经 summary 的 Command 直接跳 END，报告提示为空。"""
    report = build_graph().invoke({"text": "   "})["report"]
    assert "空文本" in report


def test_stats_merge_across_steps():
    """stats 是 reducer 合并字段，统计节点写几条最终就有几条。"""
    result = build_graph().invoke({"text": "图论基础入门。"})
    assert len(result["stats"]) == 3

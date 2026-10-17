"""第 10 篇完整案例测试：编辑部多智能体流水线（newsroom.py）。"""

import newsroom
from newsroom import MAX_REVISIONS, build_graph, print_trace


def _inputs(topic: str) -> dict:
    return {"topic": topic, "notes": [], "draft": "", "review_note": "",
            "revisions": 0, "steps": [], "final": ""}


# ---------- 正常场景 ----------

def test_happy_path_publishes():
    """无禁词一遍过：research -> writer -> reviewer -> 定稿。"""
    result = build_graph().invoke(_inputs("LangGraph 是什么"))
    assert result["final"].startswith("定稿：")
    assert result["revisions"] == 0
    assert "reviewer: 审校通过" in result["steps"]


def test_researcher_runs_before_writer():
    """notes 为空时 supervisor 先派 researcher。"""
    assert newsroom.supervisor({
        "notes": [], "draft": "x", "review_note": "",
        "revisions": 0, "final": "y"}).goto == "research"


# ---------- 禁词打回场景 ----------

def test_banned_word_gets_replaced_and_passes():
    """初稿带禁词被打回，writer 换词后二审通过，不再死循环。"""
    result = build_graph().invoke(_inputs("写一篇绝对要用赋能这个词的稿子"))
    assert result["final"].startswith("定稿：")
    assert "赋能" not in result["final"]
    assert "助力" in result["final"]
    assert result["revisions"] == 1
    # 轨迹里能看见完整协作链：初稿 -> 打回 -> 改稿 -> 通过
    assert "reviewer: 打回（发现禁词 赋能）" in result["steps"]
    assert "writer: 按批注出第 2 版" in result["steps"]
    assert "reviewer: 审校通过" in result["steps"]


def test_writer_replacement_covers_all_banned_words():
    """writer 按批注改会替换全部禁词，不只是批注点名的那个。"""
    state = _inputs("t")
    state.update(draft="又赋能又抓手还想闭环", review_note="禁词「赋能」请替换",
                 revisions=1)
    update = newsroom.writer(state)
    for word in newsroom.BANNED:
        assert word not in update["draft"]
    assert update["review_note"] == ""


def test_all_banned_words_detected():
    """reviewer 对三个禁词都会打回。"""
    for word in newsroom.BANNED:
        state = _inputs("t")
        state.update(draft=f"稿子里有{word}", revisions=0)
        cmd = newsroom.reviewer(state)
        assert cmd.goto == "supervisor"
        assert word in cmd.update["review_note"]


# ---------- 返工上限兜底 ----------

def test_supervisor_stops_at_revision_cap():
    """返工达上限时 supervisor 终止流水线转人工，不无限打回。"""
    state = _inputs("t")
    state.update(notes=["n"], draft="带禁词的稿子", review_note="禁词「赋能」请替换",
                 revisions=MAX_REVISIONS)
    cmd = newsroom.supervisor(state)
    assert cmd.goto == "__end__"
    assert "转人工" in cmd.update["final"]


def test_pipeline_step_count_bounded():
    """正常禁词场景一轮返工就收敛，协作步骤数有上界，不再死循环。"""
    result = build_graph().invoke(_inputs("写一篇绝对要用赋能这个词的稿子"))
    assert len(result["steps"]) == 5  # 调研、初稿、打回、改稿、通过


# ---------- trace 打印 ----------

def test_print_trace_accumulates_state(capsys):
    """stream updates 模式下 print_trace 能累积出完整终态。"""
    app = build_graph()
    state = print_trace(app.stream(_inputs("trace 测试"), stream_mode="updates"))
    out = capsys.readouterr().out
    assert "[reviewer]" in out
    assert state["final"].startswith("定稿：")
    assert "researcher" in " ".join(state["steps"])

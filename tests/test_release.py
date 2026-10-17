"""第 9 篇完整案例测试：带人工审批的发布流水线（release.py）。"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import release
from release import Decision, build_graph


def paused_app(task="把搜索阈值从 0.8 调到 0.75"):
    """发起一次发布申请并停在中断点。"""
    app = build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "release-1"}}
    result = app.invoke(
        {"change": task, "summary": "", "decision": None, "result": ""}, cfg)
    return app, cfg, result


def test_plan_builds_summary_locally():
    """没接模型时 plan 本地拼装摘要。"""
    _, cfg, result = paused_app()
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert "把搜索阈值从 0.8 调到 0.75" in payload["summary"]


def test_interrupt_payload_carries_schema():
    """中断对象带 response_schema，恢复值会被校验。"""
    _, _, result = paused_app()
    interrupt_obj = result["__interrupt__"][0]
    assert interrupt_obj.response_schema is not None


def test_approved_routes_to_deploy():
    """批准恢复走 deploy，备注带进结论。"""
    app, cfg, _ = paused_app()
    final = app.invoke(Command(resume=Decision(approved=True, note="低风险")), cfg)
    assert "已发布" in final["result"]
    assert "低风险" in final["result"]


def test_rejected_routes_to_reject():
    """驳回恢复走 reject，结论注明已驳回。"""
    app, cfg, _ = paused_app()
    final = app.invoke(Command(resume=Decision(approved=False, note="再想想")), cfg)
    assert "已驳回" in final["result"]
    assert "再想想" in final["result"]


def test_dict_resume_without_note_defaults():
    """恢复值用字典也能过校验，note 缺省时备注写无备注。"""
    app, cfg, _ = paused_app()
    final = app.invoke(Command(resume={"approved": True}), cfg)
    assert "无备注" in final["result"]


def test_bad_resume_rejected_then_retried():
    """乱按（类型不对）被 schema 拦下，同线程还能再按一次。"""
    app, cfg, _ = paused_app()
    with pytest.raises(Exception):
        app.invoke(Command(resume={"approved": "sure"}), cfg)
    final = app.invoke(Command(resume=Decision(approved=True, note="补按")), cfg)
    assert "已发布" in final["result"]


def test_decision_model_defaults():
    """Decision 的 note 有默认值，approved 必填。"""
    d = Decision(approved=False)
    assert d.note == ""
    with pytest.raises(Exception):
        Decision()

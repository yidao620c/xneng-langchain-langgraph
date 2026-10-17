"""第 1 篇示例测试：单节点图的执行。"""

from hello_graph import build_graph


def test_graph_runs_from_start_to_end():
    """图从 START 走到 END，用户消息在前，AI 回答在后。"""
    app = build_graph()
    result = app.invoke({"messages": [{"role": "user", "content": "很高兴认识你"}]})

    assert result["messages"][0].type == "human"
    assert result["messages"][-1].type == "ai"
    assert "第一个节点" in result["messages"][-1].content

"""第 12 篇完整案例测试：个人旅游助手（travel.py）。"""

import travel


# ---------- 固件 ----------

def _store():
    return travel.InMemoryStore()


def _isolated(tmp_path, monkeypatch):
    """重定向行程目录 + 清掉模型环境变量强制走替身。"""
    monkeypatch.setattr(travel, "PLAN_DIR", tmp_path / "travel_plans")
    for k in ("YI_API_KEY", "YI_BASE_URL", "AGENT_MODEL"):
        monkeypatch.delenv(k, raising=False)


def _rt(store):
    """手工构造 ToolRuntime（测试直接调工具时用）。"""
    return travel.ToolRuntime(state=None, context=None, config=None,
                              stream_writer=None, tool_call_id="t",
                              store=store)


# ---------- 信息工具（模拟数据） ----------

def test_weather_known_city():
    """有数据的城市返回两行预报。"""
    out = travel.get_weather.invoke({"city": "杭州"})
    assert "第一天" in out and "第二天" in out
    assert "22℃" in out


def test_weather_unknown_city():
    """没有数据的城市如实说没有，并列出现有城市。"""
    out = travel.get_weather.invoke({"city": "火星"})
    assert "没有 火星" in out and "杭州" in out


def test_attractions_filter_by_tag():
    """按城市查全部，再按 tag 过滤。"""
    all_hangzhou = travel.search_attractions.invoke({"city": "杭州"})
    assert "西湖" in all_hangzhou and "灵隐寺" in all_hangzhou
    indoor = travel.search_attractions.invoke({"city": "杭州", "tag": "室内"})
    assert "丝绸博物馆" in indoor and "西湖" not in indoor


def test_attractions_unknown_city():
    out = travel.search_attractions.invoke({"city": "火星"})
    assert "没有找到" in out


# ---------- 长期记忆 ----------

def test_remember_and_recall_roundtrip():
    """remember 写进 store，recall 读得回来。"""
    store = _store()
    travel.remember_preference.invoke(
        {"key": "饮食", "value": "不吃辣", "runtime": _rt(store)})
    out = travel.recall_preferences.invoke({"runtime": _rt(store)})
    assert "饮食=不吃辣" in out


def test_recall_empty_store():
    """空记忆给出兜底提示，不抛异常。"""
    out = travel.recall_preferences.invoke({"runtime": _rt(_store())})
    assert "还没有" in out


# ---------- 行程落盘 ----------

def test_save_itinerary_writes_file_and_memory(tmp_path, monkeypatch):
    """save_itinerary 真写文件，同时在 store 留档。"""
    _isolated(tmp_path, monkeypatch)
    store = _store()
    out = travel.save_itinerary.invoke({
        "city": "杭州", "text": "# 杭州两日游\n\nDay1 西湖",
        "runtime": _rt(store)})
    assert "杭州.md" in out
    saved = (tmp_path / "travel_plans" / "杭州.md").read_text(encoding="utf-8")
    assert "Day1 西湖" in saved
    keys = [i.key for i in store.search(travel.MEM_NAMESPACE)]
    assert "last_plan_杭州" in keys


# ---------- Agent 组装与替身流程 ----------

def test_fake_plan_end_to_end(tmp_path, monkeypatch):
    """无 key 走四步替身脚本，工具全部真执行，行程文件落地。"""
    _isolated(tmp_path, monkeypatch)
    store = _store()
    agent = travel.build_agent(store, travel.fake_model())
    result = agent.invoke({"messages": ["十一去杭州玩两天"]})

    rounds = sum(1 for m in result["messages"]
                 if type(m).__name__ == "AIMessage" and m.tool_calls)
    assert rounds == 4
    # 每个工具的真实返回都在 ToolMessage 里
    tool_contents = [m.content for m in result["messages"]
                     if type(m).__name__ == "ToolMessage"]
    assert any("第一天" in c for c in tool_contents)
    assert any("西湖" in c for c in tool_contents)
    assert any("已保存" in c for c in tool_contents)
    # 行程文件真落地
    assert (tmp_path / "travel_plans" / "杭州.md").exists()


def test_build_agent_has_five_tools():
    """agent 挂齐五个工具，工具列表是模块常量。"""
    assert len(travel.TOOLS) == 5
    names = {t.name for t in travel.TOOLS}
    assert names == {"get_weather", "search_attractions",
                     "remember_preference", "recall_preferences",
                     "save_itinerary"}

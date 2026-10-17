"""第 3 篇完整案例测试：结构化信息抽取台 extract()。

覆盖 extract() 的返回值结构、嵌套 cast 解析、解析失败路径、
build_model 分支。全部用 ScriptedChatModel 驱动，不依赖外部 API。
"""

import extract as extract_mod
from extract import Actor, MovieInfo, build_model, extract
from fake_models import ScriptedChatModel
from langchain.messages import AIMessage


def scripted_tool_call(args):
    """快捷构造：按 schema 名 MovieInfo 供应工具调用的 fake 模型。"""
    return ScriptedChatModel(
        messages=iter([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "MovieInfo",
                    "args": args,
                    "id": "call_1",
                    "type": "tool_call",
                }],
            )
        ])
    )


def test_extract_returns_parsed_movie_with_nested_cast():
    """extract() 解析出 MovieInfo 实例，嵌套的 cast 列表同步到位。"""
    model = scripted_tool_call({
        "title": "盗梦空间",
        "year": 2010,
        "director": "诺兰",
        "cast": [{"name": "莱昂纳多", "role": "Cobb"}],
        "genres": ["科幻"],
    })

    result = extract(model, "诺兰执导的《盗梦空间》2010年上映")

    movie = result["parsed"]
    assert isinstance(movie, MovieInfo)
    assert movie.title == "盗梦空间"
    assert movie.year == 2010
    assert movie.cast == [Actor(name="莱昂纳多", role="Cobb")]
    assert movie.genres == ["科幻"]
    assert result["parsing_error"] is None
    assert result["elapsed_s"] >= 0
    assert result["usage"] is None  # fake 不产用量，真实模型才有


def test_extract_reports_failure_without_raising():
    """模型没按 schema 输出时，extract() 不抛异常，把原始文本交出来排查。"""
    model = ScriptedChatModel(messages=iter([AIMessage(content="我不知道")]))

    result = extract(model, "随便一段文本")

    assert result["parsed"] is None
    assert result["raw_text"] == "我不知道"


def test_extract_prompt_contains_text_and_fields():
    """提示词把待抽取文本与字段名都带上（json_mode 依赖提示词传 schema）。"""
    prompt = extract_mod.EXTRACT_PROMPT.format(text="测试文本")

    assert "测试文本" in prompt
    for field in ("title", "year", "director", "cast", "genres"):
        assert field in prompt


def test_build_model_defaults_to_scripted(monkeypatch):
    """未指定模型名时用本地替身。"""
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    assert isinstance(build_model(), ScriptedChatModel)


def test_build_model_forwards_params_to_real_model(monkeypatch):
    """指定真实模型时，模型参数与网关参数随 init_chat_model 原样透传。"""
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-5.5")
    monkeypatch.delenv("YI_BASE_URL", raising=False)
    monkeypatch.delenv("YI_API_KEY", raising=False)
    seen = {}

    def fake_init(model_name, **kwargs):
        seen["model_name"] = model_name
        seen["kwargs"] = kwargs
        return "REAL_MODEL"

    monkeypatch.setattr(extract_mod, "init_chat_model", fake_init)

    assert build_model(temperature=0, max_tokens=500) == "REAL_MODEL"
    assert seen["model_name"] == "openai:gpt-5.5"
    assert seen["kwargs"] == {"temperature": 0, "max_tokens": 500}

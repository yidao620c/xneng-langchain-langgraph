"""第 3 篇完整案例：结构化信息抽取台（extract.py）。

一条命令，把一段自然语言抽成结构化数据：电影名、年份、导演、演员表、类型。
接一个新模型，先跑它十几条文本，看字段准不准、快不快、烧多少 token，
再决定用不用。

用法：
    uv run python extract.py "诺兰执导的《盗梦空间》2010年上映，莱昂纳多饰 Cobb"
    uv run python extract.py "..." --model openai:gpt-5.5 --temperature 0

模型来源与第 1、2 篇一致：--model 或环境变量 AGENT_MODEL 指定「提供商:模型名」
时使用真实模型；都没指定时用 ScriptedChatModel 本地演示。

⚠️ method 参数的选择（本案例的关键背景）：
    with_structured_output 有三种 method，能力由模型与网关决定：
    - json_schema     提供商原生结构化输出，最可靠（OpenAI / Anthropic 等）
    - function_calling 强制一次工具调用，schema 由工具参数承载
    - json_mode       只把 schema 写进提示词，不强制结构，最兼容也最松散
    本案例面向 OpenAI 兼容网关（API易）调参：实测该网关不放行
    json_schema（response_format 不可用）与 function_calling（推理模式
    不支持强制 tool_choice），只放行 json_mode，故显式指定。
    直连支持原生结构化输出的提供商时，换成默认的 json_schema 更稳。
    json_mode 不强制结构，提示词里必须钉死字段名（见 EXTRACT_PROMPT）。
"""

import argparse
import json
import os
import time

from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage

from fake_models import ScriptedChatModel


class Actor(BaseModel):
    """一名演员及其饰演的角色。"""

    name: str = Field(description="演员姓名")
    role: str = Field(description="饰演的角色名")


class MovieInfo(BaseModel):
    """一部电影的结构化信息。"""

    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")
    director: str = Field(description="导演")
    cast: list[Actor] = Field(default_factory=list, description="主要演员与角色")
    genres: list[str] = Field(default_factory=list, description="类型标签，如 科幻")


EXTRACT_PROMPT = (
    "从下面的文本里抽取电影信息，只输出 JSON，字段名必须严格使用 "
    "title、year、director、cast、genres，其中 cast 的每项用 name 和 role "
    "两个字段，不要增加任何其他字段，抽不到的元素给空列表。"
    "文本：{text}"
)


def build_model(model_name=None, temperature=None, max_tokens=None):
    """按参数组装模型（约定同第 2 篇 ask.py）。"""
    name = model_name or os.environ.get("AGENT_MODEL")
    params = {}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if name:
        if os.environ.get("YI_BASE_URL"):
            params["base_url"] = os.environ["YI_BASE_URL"]
        if os.environ.get("YI_API_KEY"):
            params["api_key"] = os.environ["YI_API_KEY"]
        return init_chat_model(name, **params)
    return ScriptedChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "MovieInfo",
                        "args": {
                            "title": "盗梦空间",
                            "year": 2010,
                            "director": "克里斯托弗·诺兰",
                            "cast": [{"name": "莱昂纳多·迪卡普里奥", "role": "Cobb"}],
                            "genres": ["科幻", "悬疑"],
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }],
                )
            ]
        ),
        **params,
    )


def extract(model, text):
    """抽取结构化电影信息，返回解析结果与统计数据。

    返回值：
        parsed        MovieInfo 实例（解析失败为 None）
        parsing_error 解析错误信息（成功为 None）
        raw_text      模型原始输出文本（排查字段跑偏用）
        usage         用量元数据，fake 模型为 None
        elapsed_s     端到端耗时（秒）
    """
    structured = model.with_structured_output(MovieInfo, method="json_mode", include_raw=True)
    started = time.perf_counter()
    out = structured.invoke(EXTRACT_PROMPT.format(text=text))
    elapsed = time.perf_counter() - started
    raw = out["raw"]
    return {
        "parsed": out["parsed"],
        "parsing_error": out["parsing_error"],
        "raw_text": raw.text if raw is not None else "",
        "usage": raw.usage_metadata if raw is not None else None,
        "elapsed_s": round(elapsed, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="结构化信息抽取台：自然语言抽成结构化数据")
    parser.add_argument("text", help="待抽取的自然语言文本")
    parser.add_argument("--model", help="提供商:模型名，例如 openai:gpt-5.5")
    parser.add_argument("--temperature", type=float, help="采样温度，0~1 越高越发散")
    parser.add_argument("--max-tokens", type=int, help="限制回答的最大 token 数")
    args = parser.parse_args()

    model = build_model(args.model, args.temperature, args.max_tokens)
    name = args.model or os.environ.get("AGENT_MODEL") or "ScriptedChatModel（本地替身）"
    print(f"模型: {name}")
    print(f"参数: temperature={args.temperature} max_tokens={args.max_tokens}")
    print(f"文本: {args.text}")
    print()

    result = extract(model, args.text)

    print("---- 抽取结果 ----")
    if result["parsed"] is not None:
        movie = result["parsed"]
        print(f"标题: {movie.title}")
        print(f"年份: {movie.year}")
        print(f"导演: {movie.director}")
        print(f"演员: {'、'.join(f'{a.name} 饰 {a.role}' for a in movie.cast) or '（无）'}")
        print(f"类型: {'、'.join(movie.genres) or '（无）'}")
        print(f"\nJSON: {movie.model_dump_json(indent=2, ensure_ascii=False)}")
    else:
        print("解析失败。raw 输出如下，检查字段名是否跑偏。")
        print(result["raw_text"])
        if result["parsing_error"]:
            print(f"\n错误: {result['parsing_error']}")

    print()
    print("---- 统计 ----")
    print(f"耗时: {result['elapsed_s']}s")
    if result["usage"]:
        u = result["usage"]
        detail = u.get("output_token_details") or {}
        reasoning = detail.get("reasoning", 0)
        print(f"token 用量: 输入 {u['input_tokens']} / 输出 {u['output_tokens']}"
              f" / 合计 {u['total_tokens']}"
              + (f"（其中思考 {reasoning}）" if reasoning else ""))
    else:
        print("token 用量: 无（fake 模型不产用量，真实模型才有）")


if __name__ == "__main__":
    main()

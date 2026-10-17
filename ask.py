"""第 2 篇完整案例：模型试用台（ask.py）。

一条命令，把新模型的四个问题一次问清：
1. 接进来要写几行代码？——一行 init_chat_model；
2. temperature 调高调低，回答怎么变？——构造参数实时生效；
3. 流式输出长什么样？——chunk 逐块到达，终端边收边显；
4. 一次调用花多少 token、等多久？——usage_metadata 加计时。

用法：
    uv run python ask.py "北京天气怎么样？"
    uv run python ask.py "给我三个项目名" --temperature 1
    uv run python ask.py "你好" --model openai:gpt-5.5
    uv run python ask.py "你好" --model deepseek-v4.1-flash

模型来源与第 1、2 篇的示例一致：--model 或环境变量 AGENT_MODEL 指定
「提供商:模型名」时使用真实模型；都没指定时用 ScriptedChatModel 本地演示。
"""

import argparse
import os
import time

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage

from fake_models import ScriptedChatModel


def build_model(model_name=None, temperature=None, max_tokens=None):
    """按参数组装模型。

    模型参数（temperature / max_tokens）在构造时传入，对整个实例生效；
    调用时还可以用 config 传运行配置（见 hello_models.demo_params）。

    约定了 YI_BASE_URL / YI_API_KEY（API易 等 OpenAI 兼容网关）时，
    它们作为 base_url / api_key 随模型一起传入。
    """
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
            [AIMessage(content="北京 今天 晴天，气温 26℃，适合出行。")]
        ),
        **params,
    )


def ask(model, question, on_token=None):
    """流式提问，返回回答文本与统计数据。

    返回值：
        text      完整回答（流式 chunk 用加法聚合而成）
        chunks    收到的 chunk 数（真实模型≈token 数，fake 模型按空白切片）
        usage     用量元数据（input/output/total tokens），fake 模型为 None
        elapsed_s 端到端耗时（秒）
    """
    started = time.perf_counter()
    full = None  # None | AIMessageChunk
    n_chunks = 0
    for chunk in model.stream(question):
        full = chunk if full is None else full + chunk
        n_chunks += 1
        if on_token is not None:
            on_token(chunk.text)
    elapsed = time.perf_counter() - started
    usage = full.usage_metadata if full is not None else None
    return {
        "text": full.text if full is not None else "",
        "chunks": n_chunks,
        "usage": usage,
        "elapsed_s": round(elapsed, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="模型试用台：流式提问并统计用量")
    parser.add_argument("question", help="要问模型的问题")
    parser.add_argument("--model", help="提供商:模型名，例如 openai:gpt-5.5")
    parser.add_argument("--temperature", type=float, help="采样温度，0~1 越高越发散")
    parser.add_argument("--max-tokens", type=int, help="限制回答的最大 token 数")
    args = parser.parse_args()

    model = build_model(args.model, args.temperature, args.max_tokens)
    name = args.model or os.environ.get("AGENT_MODEL") or "ScriptedChatModel（本地替身）"
    print(f"模型: {name}")
    print(f"参数: temperature={args.temperature} max_tokens={args.max_tokens}")
    print(f"问题: {args.question}")
    print("回答: ", end="", flush=True)

    result = ask(model, args.question, on_token=lambda t: print(t, end="", flush=True))
    print()
    print()
    print("---- 统计 ----")
    print(f"chunk 数: {result['chunks']}")
    print(f"耗时: {result['elapsed_s']}s")
    if result["usage"]:
        u = result["usage"]
        print(f"token 用量: 输入 {u['input_tokens']} / 输出 {u['output_tokens']}"
              f" / 合计 {u['total_tokens']}")
    else:
        print("token 用量: 无（fake 模型不产用量，真实模型才有）")


if __name__ == "__main__":
    main()

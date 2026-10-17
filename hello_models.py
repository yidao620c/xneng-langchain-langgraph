"""第 2 篇示例：模型抽象——一个接口，百家模型。

围绕 init_chat_model 演示 LangChain 1.x 模型层的五个核心用法：

1. invoke：单次调用，入参可以是字符串，也可以是带历史的消息列表；
2. stream：流式输出，AIMessageChunk 逐块产出，用加法聚合成完整消息；
3. batch：批量并行调用，一次问多个问题；
4. 参数：模型参数（temperature 等）在构造时传，运行配置在调用时传；
5. 提供商切换：换模型只改 "提供商:模型名" 这一个字符串。

模型来源与第 1 篇相同：设置环境变量 AGENT_MODEL（例如 deepseek-v4.1-flash）
时使用真实模型；未设置时用 ScriptedChatModel 在本地演示完整流程，
不需要任何 API key。

⚠️ fake 模型按脚本顺序供应消息，每次调用消耗一条，脚本条数必须覆盖
main() 里的全部调用次数（当前为 6 条）。
"""

import os

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from fake_models import ScriptedChatModel


def build_model():
    """组装模型：优先使用环境变量配置的真实模型。

    模型名用「提供商:模型名」格式，例如：
        openai:gpt-5.5
        openai:deepseek-v4.1-flash   （API易 网关，见下方约定）
        anthropic:claude-sonnet-4-6

    约定了 YI_BASE_URL / YI_API_KEY（API易 等 OpenAI 兼容网关）时，
    它们作为 base_url / api_key 随模型一起传入，模型名写 openai:<模型名>。
    """
    model_name = os.environ.get("AGENT_MODEL")
    if model_name:
        kwargs = {}
        if os.environ.get("YI_BASE_URL"):
            kwargs["base_url"] = os.environ["YI_BASE_URL"]
        if os.environ.get("YI_API_KEY"):
            kwargs["api_key"] = os.environ["YI_API_KEY"]
        return init_chat_model(model_name, **kwargs)
    # 本地演示脚本：按 main() 的调用顺序供应 6 条消息
    return ScriptedChatModel(
        messages=iter(
            [
                AIMessage(content="北京今天晴天，气温 26℃，适合出行。"),          # 1 字符串调用
                AIMessage(content="北京位于华北平原北部，是一座历史悠久的城市。"),  # 2 消息列表调用
                AIMessage(content="流式 输出 把 回答 切成 小块 逐块 送达"),        # 3 流式调用
                AIMessage(content="第一个批量问题的回答。"),                      # 4 批量调用（一）
                AIMessage(content="第二个批量问题的回答。"),                      # 5 批量调用（二）
                AIMessage(content="这一趟调用的 run_name 是 demo_params。"),      # 6 带 config 调用
            ]
        )
    )


def demo_invoke_string(model):
    """用法 1a：invoke 传字符串，框架自动包成 HumanMessage。"""
    response = model.invoke("北京天气怎么样？")
    print("== invoke（字符串入参） ==")
    print(f"返回类型: {type(response).__name__}")
    print(f"回答: {response.text}")
    print()


def demo_invoke_messages(model):
    """用法 1b：invoke 传消息列表，携带多轮对话历史。"""
    conversation = [
        SystemMessage("你是一个简洁的地理助手，回答不超过两句话。"),
        HumanMessage("北京是哪座城市？"),
        AIMessage("北京是中华人民共和国的首都，位于华北平原北部。"),
        HumanMessage("它有多少年的建城史？"),
    ]
    response = model.invoke(conversation)
    print("== invoke（消息列表入参） ==")
    print(f"历史消息数: {len(conversation)}")
    print(f"回答: {response.text}")
    print()


def demo_stream(model):
    """用法 2：stream 流式输出，chunk 逐块到达，加法聚合成完整消息。

    注意：推理模型（如 deepseek-v4.1-flash）会先输出一批空文本 chunk
    （思考 token），chunk.text 为空属正常现象，聚合结果不受影响；
    用量里 output_token_details.reasoning 单独统计这部分 token。
    """
    print("== stream（流式输出） ==")
    full = None  # None | AIMessageChunk
    n_chunks = 0
    for chunk in model.stream("演示一下流式输出"):
        full = chunk if full is None else full + chunk
        n_chunks += 1
        if chunk.text:  # 空文本 chunk（思考 token）不打印，但照常聚合
            print(f"  收到文本: {chunk.text!r}")
    print(f"共 {n_chunks} 个 chunk（可能含空文本 chunk），聚合后: {full.text}")
    print(f"chunk 类型: {type(chunk).__name__}（AIMessage 的子类）")
    print()


def demo_batch(model):
    """用法 3：batch 批量并行调用，返回顺序与输入一致。"""
    questions = ["第一个批量问题", "第二个批量问题"]
    responses = model.batch(questions)
    print("== batch（批量调用） ==")
    for q, r in zip(questions, responses):
        print(f"  问: {q} -> 答: {r.text}")
    print()


def demo_params(model):
    """用法 4：模型参数在构造时传，运行配置（config）在调用时传。"""
    response = model.invoke(
        "这一趟调用带了哪些运行配置？",
        config={
            "run_name": "demo_params",           # 这次调用在追踪里的名字
            "tags": ["lesson-02", "models"],     # 标签，便于检索
            "metadata": {"source": "hello_models"},  # 自定义元数据
        },
    )
    print("== invoke（带 config 运行配置） ==")
    print(f"回答: {response.text}")
    print("config 不改变模型行为，供追踪与检索使用（LangSmith 等）")
    print()


def main() -> None:
    model = build_model()
    print(f"当前模型: {type(model).__name__}")
    print(f"模型名: {getattr(model, 'model_name', None) or getattr(model, 'model', 'ScriptedChatModel（本地替身）')}")
    print()
    demo_invoke_string(model)
    demo_invoke_messages(model)
    demo_stream(model)
    demo_batch(model)
    demo_params(model)


if __name__ == "__main__":
    main()

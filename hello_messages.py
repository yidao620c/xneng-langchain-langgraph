"""第 3 篇示例：消息与结构化输出——模型的通用货币，应用的结构化接口。

围绕 langchain.messages 与 with_structured_output 演示七个核心用法：

1. 消息类型：SystemMessage / HumanMessage / AIMessage / ToolMessage，三种入参形式；
2. content 与 content_blocks：松散 content 与标准内容块的惰性解析；
3. 多模态：HumanMessage 的 content blocks 图片输入（url / base64 / file_id）；
4. 元数据：AIMessage 的 id / usage_metadata / response_metadata，聚合后再读；
5. ToolMessage：工具结果回传，artifact 携带不发给模型的附加数据；
6. 序列化：dumpd / load 持久化对话历史；
7. 结构化输出：with_structured_output 三种 schema、method 参数、include_raw。

模型来源与第 1、2 篇相同：设置环境变量 AGENT_MODEL 时使用真实模型；
未设置时用 ScriptedChatModel 在本地演示，不需要任何 API key。

⚠️ fake 模型按脚本顺序供应消息，每次调用消耗一条；结构化输出演示的
脚本消息带 tool_calls（基础实现走工具调用解析路径，与 method 参数无关），
脚本条数必须覆盖 main() 里的全部调用次数（当前为 9 条）。
"""

import json
import os

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.load import dumpd, load

from fake_models import ScriptedChatModel


class Movie(BaseModel):
    """A movie with details."""

    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")


class MovieDict(TypedDict):
    """A movie with details."""

    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "The year the movie was released"]
    director: Annotated[str, ..., "The director of the movie"]


MOVIE_JSON_SCHEMA = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The title of the movie"},
        "year": {"type": "integer", "description": "The year the movie was released"},
        "director": {"type": "string", "description": "The director of the movie"},
    },
    "required": ["title", "year", "director"],
}

# json_mode 只把 schema 写进提示词、不强制结构，字段名容易跑偏，
# 所以提示词里显式钉死字段名（实测 deepseek-v4.1-flash 必须这样才稳定）。
EXTRACT_PROMPT = (
    "从这句话里抽取电影信息，只输出 JSON，"
    "字段名必须严格使用 title、year、director，不要增加任何其他字段。"
    "句子：诺兰执导的《盗梦空间》2010年上映"
)


def build_model():
    """组装模型：优先使用环境变量配置的真实模型（约定同第 2 篇）。"""
    model_name = os.environ.get("AGENT_MODEL")
    if model_name:
        kwargs = {}
        if os.environ.get("YI_BASE_URL"):
            kwargs["base_url"] = os.environ["YI_BASE_URL"]
        if os.environ.get("YI_API_KEY"):
            kwargs["api_key"] = os.environ["YI_API_KEY"]
        return init_chat_model(model_name, **kwargs)
    # 本地演示脚本：按 main() 的调用顺序供应 9 条消息
    return ScriptedChatModel(
        messages=iter(
            [
                AIMessage(content="北京位于华北平原北部，是一座历史悠久的城市。"),  # 1 消息类型·对象列表
                AIMessage(content="北京是中华人民共和国的首都，位于华北平原北部。"),  # 2 消息类型·字典列表
                AIMessage(content="这是一张模型抽象架构图，展示了统一接口与多家提供商的关系。"),  # 3 多模态
                AIMessage(content="消息是对话的最小单元，由角色、内容和元数据组成。"),  # 4 元数据
                AIMessage(  # 5 ToolMessage·模型发起工具调用
                    content="",
                    tool_calls=[{
                        "name": "get_weather",
                        "args": {"location": "北京"},
                        "id": "call_1", "type": "tool_call",
                    }],
                ),
                AIMessage(content="北京今天晴，气温 26℃，适合出行。"),              # 6 ToolMessage·回传后回答
                AIMessage(  # 5 结构化：Pydantic
                    content="",
                    tool_calls=[{
                        "name": "Movie",
                        "args": {"title": "盗梦空间", "year": 2010, "director": "诺兰"},
                        "id": "call_1", "type": "tool_call",
                    }],
                ),
                AIMessage(  # 6 结构化：TypedDict
                    content="",
                    tool_calls=[{
                        "name": "MovieDict",
                        "args": {"title": "盗梦空间", "year": 2010, "director": "诺兰"},
                        "id": "call_2", "type": "tool_call",
                    }],
                ),
                AIMessage(  # 7 结构化：JSON Schema
                    content="",
                    tool_calls=[{
                        "name": "Movie",
                        "args": {"title": "盗梦空间", "year": 2010, "director": "诺兰"},
                        "id": "call_3", "type": "tool_call",
                    }],
                ),
            ]
        )
    )


def demo_message_types(model):
    """用法 1：四种消息类型与三种入参形式（字符串 / 对象列表 / 字典列表）。"""
    print("== 消息类型与三种入参 ==")
    conversation = [
        SystemMessage("你是一个简洁的地理助手，回答不超过两句话。"),
        HumanMessage("北京是哪座城市？"),
        AIMessage("北京是中华人民共和国的首都。"),
        HumanMessage("它有多少年的建城史？"),
    ]
    response = model.invoke(conversation)
    print(f"对象列表入参: {[type(m).__name__ for m in conversation]}")
    print(f"回答: {response.text}")

    # 字典格式：OpenAI chat completions 风格，框架自动转换
    as_dicts = [
        {"role": "system", "content": "你是一个简洁的地理助手。"},
        {"role": "user", "content": "北京是哪座城市？"},
    ]
    response = model.invoke(as_dicts)
    print(f"字典列表入参也能跑: {response.text[:30]}...")
    print()


def demo_content_blocks():
    """用法 2：content 的三种构造方式与 content_blocks 的惰性解析。

    不调模型，手工构造一条「提供商原生格式」的 AIMessage，
    看 content_blocks 怎么把它归一成标准内容块。
    """
    print("== content 与 content_blocks ==")
    message = AIMessage(
        content=[
            {"type": "thinking", "thinking": "用户问的是消息的定义。", "signature": "WaUjzkyp"},
            {"type": "text", "text": "消息是对话的最小单元。"},
        ],
        response_metadata={"model_provider": "anthropic"},
    )
    print(f"content 原样（提供商原生）: {message.content}")
    print(f"content_blocks（标准块）: {message.content_blocks}")
    print("content_blocks 是惰性解析的属性，跨提供商格式归一，content 不受影响")
    print()


def demo_multimodal(model):
    """用法 3：多模态输入，HumanMessage 的 content blocks 图片三种来源。"""
    print("== 多模态：图片输入 ==")
    # url 来源（base64 / file_id 来源见文章正文，结构相同）
    message = HumanMessage(content=[
        {"type": "text", "text": "用一句话描述这张图。"},
        {"type": "image", "url": "https://static.xiongneng.me/model-abstraction-20261017122518.png"},
    ])
    response = model.invoke([message])
    print(f"图片输入也能跑: {response.text[:40]}...")
    print("content 是块列表时框架按多模态消息处理，文本块与图片块可以混排")
    print()


def demo_metadata(model):
    """用法 4：AIMessage 元数据，id / usage_metadata / response_metadata。"""
    print("== AIMessage 元数据 ==")
    response = model.invoke("用一句话说明什么是消息")
    print(f"id: {response.id}")
    print(f"usage_metadata: {response.usage_metadata}")
    meta = response.response_metadata or {}
    print(f"response_metadata 键: {sorted(meta)[:5]}")
    print("usage 里 output_token_details.reasoning 是推理模型的思考 token，单独计费")
    print()


def demo_tool_message(model):
    """用法 5：ToolMessage 把工具结果回传给模型，artifact 携带附加数据。

    先用 bind_tools 让模型发起一次真实的工具调用，再把结果包成
    ToolMessage 回传（完整的工具调用循环是第 4 篇的主题，这里只讲消息）。
    ⚠️ 回传的 AIMessage 必须是模型当轮产出的原消息：思考模型
    （如 deepseek-v4.1-flash）要求 reasoning_content 随工具调用一起回传，
    手工构造的 AIMessage 会被网关拒绝。
    """
    print("== ToolMessage 与 artifact ==")

    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "晴，26℃"

    ai_message = model.bind_tools([get_weather]).invoke("北京天气怎么样？")
    call = ai_message.tool_calls[0]
    print(f"模型发起工具调用: {call['name']}({call['args']})")

    tool_message = ToolMessage(
        content="晴，26℃",
        tool_call_id=call["id"],     # 必须与 tool_call 的 id 匹配
        name=call["name"],
        artifact={"source": "本地替身", "cached": False},  # 不发给模型，下游可读
    )
    response = model.invoke([
        HumanMessage("北京天气怎么样？"),
        ai_message,
        tool_message,
    ])
    print(f"工具结果回传后模型回答: {response.text}")
    print(f"artifact 不下发给模型: {tool_message.artifact}")
    print()


def demo_serialization():
    """用法 6：dumpd / load 序列化消息，持久化对话历史。"""
    print("== 序列化：dumpd / load ==")
    message = HumanMessage("北京是哪座城市？", id="msg_123")
    serialized = dumpd(message)
    restored = load(serialized)
    print(f"序列化成普通 dict: type={serialized['type']}（可存 JSON / 数据库）")
    print(f"反序列化回消息对象: {type(restored).__name__}，内容一致: {restored.content == message.content}")
    print("⚠️ load() 会实例化 Python 对象，切勿对不可信数据调用")
    print()


def demo_structured_pydantic(model):
    """用法 7a：Pydantic schema，带字段校验，返回模型实例。"""
    print("== 结构化输出：Pydantic ==")
    structured = model.with_structured_output(Movie, method="json_mode")
    movie = structured.invoke(EXTRACT_PROMPT)
    print(f"解析结果: {movie!r}")
    print(f"类型: {type(movie).__name__}（带字段校验，返回模型实例）")
    print()


def demo_structured_typeddict(model):
    """用法 7b：TypedDict schema，无运行时校验，返回普通 dict。"""
    print("== 结构化输出：TypedDict ==")
    structured = model.with_structured_output(MovieDict, method="json_mode")
    movie = structured.invoke(EXTRACT_PROMPT)
    print(f"解析结果: {movie!r}")
    print(f"类型: {type(movie).__name__}（无运行时校验，返回 dict）")
    print()


def demo_structured_json_schema(model):
    """用法 7c：JSON Schema dict，最大控制权，必须显式指定 method。"""
    print("== 结构化输出：JSON Schema ==")
    structured = model.with_structured_output(MOVIE_JSON_SCHEMA, method="json_mode")
    movie = structured.invoke(EXTRACT_PROMPT)
    print(f"解析结果: {json.dumps(movie, ensure_ascii=False)}")
    print("schema 是普通 dict，可以来自文件或接口，跨语言复用")
    print()


def main() -> None:
    model = build_model()
    print(f"当前模型: {type(model).__name__}")
    print(f"模型名: {getattr(model, 'model_name', None) or 'ScriptedChatModel（本地替身）'}")
    print()
    demo_message_types(model)
    demo_content_blocks()
    demo_multimodal(model)
    demo_metadata(model)
    demo_tool_message(model)
    demo_serialization()
    demo_structured_pydantic(model)
    demo_structured_typeddict(model)
    demo_structured_json_schema(model)


if __name__ == "__main__":
    main()

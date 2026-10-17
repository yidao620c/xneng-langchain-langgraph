"""第 1 篇示例：第一个 LangChain Agent。

用 create_agent 把「模型 + 工具 + 系统提示」组装成一个会调用工具的智能体。

模型来源：
- 设置环境变量 AGENT_MODEL（例如 openai:gpt-5.5）时使用真实模型，
  需要对应提供商的 API key；
- 未设置时使用 ScriptedChatModel 在本地演示完整的工具调用循环，
  不需要任何 API key。
"""

import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from fake_models import ScriptedChatModel


def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return f"{city}今天晴天，气温 26℃，适合出行。"


def build_agent():
    """组装 Agent：优先使用环境变量配置的真实模型。"""
    model_name = os.environ.get("AGENT_MODEL")
    if model_name:
        from langchain.chat_models import init_chat_model

        model = init_chat_model(model_name)
    else:
        # 未配置真实模型时，用脚本化模型演示完整循环：
        # 第一轮模型决定调用工具，第二轮拿到工具结果后生成回答
        model = ScriptedChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_weather", "args": {"city": "北京"}, "id": "call_1"}
                        ],
                    ),
                    AIMessage(content="北京今天晴天，气温 26℃，适合出行。"),
                ]
            )
        )

    return create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="你是一个简洁的天气助手，回答不超过两句话。",
    )


def main() -> None:
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": "北京天气怎么样？"}]})

    print("== 消息流 ==")
    for message in result["messages"]:
        print(f"[{message.type}] {message.content}")
        for call in getattr(message, "tool_calls", None) or []:
            print(f"    -> 调用工具 {call['name']}({call['args']})")


if __name__ == "__main__":
    main()

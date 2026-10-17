"""第 4 篇完整案例：工具箱（toolbox.py）。

把工具当资产管理。ToolBox 统一注册、查看、调用、绑定一组工具，
接一个新模型先 --list 看工具集，再 --call 单点测试每个工具，
最后走 create_agent 让模型自己挑工具。工具出错不当异常炸循环，
转成模型读得懂的文字，让它自己调整。

用法：
    uv run python toolbox.py --list
    uv run python toolbox.py --call get_weather --args '{"city": "北京"}'
    uv run python toolbox.py "北京天气怎么样？算上到上海的距离"
    uv run python toolbox.py "..." --model openai:gpt-5.5 --temperature 0

模型来源与前三篇一致：--model 或环境变量 AGENT_MODEL 指定时用真实模型；
都没指定时用 ScriptedChatModel 本地演示。
"""

import argparse
import ast
import json
import operator
import os
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langchain.tools import ToolException, tool
from langchain_core.tools import BaseTool, StructuredTool

from fake_models import ScriptedChatModel


# ---------- 四个示例工具 ----------

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。

    Args:
        city: 城市名
    """
    return f"{city}今天晴天，气温 26℃，适合出行。"


_ARITH = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _safe_eval(node):
    """只放行四则运算与幂运算的递归求值。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH:
        return _ARITH[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ToolException(f"不支持的表达式元素: {ast.dump(node)[:60]}")


def calculate(expression: str) -> str:
    """计算四则运算表达式，支持加减乘除、乘方和括号。

    Args:
        expression: 算式，例如 (3 + 4) * 2
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
    except ToolException:
        raise
    except SyntaxError as e:
        raise ToolException(f"表达式语法错误: {e.msg}") from e
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"


# 外部依赖型的工具，异常统一兜底成模型读得懂的话
calculate_tool = StructuredTool.from_function(
    calculate, handle_tool_error="表达式算不了，只支持数字与四则运算，请简化后重试"
)

_LENGTH = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0}
_TEMP = {"c": ("c",), "f": ("f",)}


def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
    """在常用单位间换算，支持长度 mm/cm/m/km 与温度 c/f。

    Args:
        value: 数值
        from_unit: 原单位
        to_unit: 目标单位
    """
    fu, tu = from_unit.lower(), to_unit.lower()
    if fu in _LENGTH and tu in _LENGTH:
        result = value * _LENGTH[fu] / _LENGTH[tu]
    elif fu in _TEMP and tu in _TEMP:
        if fu == tu:
            result = value
        elif fu == "c":
            result = value * 9 / 5 + 32
        else:
            result = (value - 32) * 5 / 9
    else:
        raise ToolException(f"不支持的单位组合: {from_unit} -> {to_unit}")
    return f"{value} {from_unit} = {round(result, 4)} {to_unit}"


convert_tool = StructuredTool.from_function(
    convert_unit, handle_tool_error="单位不认识，长度支持 mm/cm/m/km，温度支持 c/f"
)


@tool
def now_time() -> str:
    """获取当前本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------- ToolBox：工具集管理 ----------

class ToolBox:
    """工具的注册表。统一登记、查看、直调、绑定，工具集一目了然。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, *tools: BaseTool) -> None:
        """登记工具，重名直接报错，避免静默覆盖。"""
        for t in tools:
            if t.name in self._tools:
                raise ValueError(f"工具重名: {t.name}")
            self._tools[t.name] = t

    def unregister(self, name: str) -> None:
        """下线一个工具。"""
        self._tools.pop(name, None)

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def table(self) -> str:
        """工具集一览表，名称、入参、描述各一列。"""
        rows = []
        for t in self._tools.values():
            args = ", ".join(t.args) or "（无入参）"
            desc = t.description.splitlines()[0]
            rows.append(f"{t.name:<16} {args:<28} {desc}")
        header = f"{'名称':<16} {'入参':<28} 描述"
        return "\n".join([header, "-" * 72, *rows])

    def call(self, name: str, args: dict) -> str:
        """直调单个工具，返回内容文本；找不到工具给明确报错。"""
        t = self.get(name)
        if t is None:
            raise ToolException(f"工具箱里没有 {name}，现有工具: {', '.join(self._tools)}")
        return str(t.invoke(args))

    def bind(self, model):
        """把整个工具集绑定到模型上。"""
        return model.bind_tools(self.tools)


# ---------- 模型与 CLI ----------

def build_model(model_name=None, temperature=None, max_tokens=None):
    """按参数组装模型（约定同前三篇）。"""
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
                AIMessage(  # 1 发起天气调用
                    content="",
                    tool_calls=[{
                        "name": "get_weather", "args": {"city": "北京"},
                        "id": "call_1", "type": "tool_call",
                    }],
                ),
                AIMessage(content="北京今天晴，气温 26℃，适合出行。"),  # 2 拿到结果回答
            ]
        ),
        **params,
    )


def build_agent(model, box: ToolBox):
    """工具箱 + 模型 = 会自己挑工具的助手。"""
    from langchain.agents import create_agent

    return create_agent(
        model=model,
        tools=box.tools,
        system_prompt="你是一个简洁的助手，优先使用工具箱里的工具回答，回答不超过三句话。",
    )


def main() -> None:
    box = ToolBox()
    box.register(get_weather, calculate_tool, convert_tool, now_time)

    parser = argparse.ArgumentParser(description="工具箱：注册、查看、直调、交给模型调度")
    parser.add_argument("question", nargs="?", help="交给模型的问题")
    parser.add_argument("--list", action="store_true", help="打印工具集一览表")
    parser.add_argument("--call", help="直调指定工具")
    parser.add_argument("--args", help="直调时的入参 JSON，配合 --call")
    parser.add_argument("--model", help="提供商:模型名，例如 openai:gpt-5.5")
    parser.add_argument("--temperature", type=float, help="采样温度")
    parser.add_argument("--max-tokens", type=int, help="最大输出 token")
    cli = parser.parse_args()

    if cli.list:
        print(box.table())
        return
    if cli.call:
        args = json.loads(cli.args) if cli.args else {}
        print(f"工具: {cli.call}")
        print(f"入参: {args}")
        print(f"结果: {box.call(cli.call, args)}")
        return

    if not cli.question:
        parser.error("给一个问题，或者用 --list / --call")

    model = build_model(cli.model, cli.temperature, cli.max_tokens)
    name = cli.model or os.environ.get("AGENT_MODEL") or "ScriptedChatModel（本地替身）"
    print(f"模型: {name}")
    print(f"工具集: {', '.join(t.name for t in box.tools)}")
    print(f"问题: {cli.question}")
    print()

    agent = build_agent(model, box)
    result = agent.invoke({"messages": [{"role": "user", "content": cli.question}]})
    for m in result["messages"]:
        if m.type == "ai" and m.tool_calls:
            for call in m.tool_calls:
                print(f"-> 调用 {call['name']}({call['args']})")
    print(f"回答: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()

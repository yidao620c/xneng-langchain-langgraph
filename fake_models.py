"""本地测试替身：脚本化聊天模型 + 哈希嵌入。

两个用途：
1. 本地没有模型 API key 时，完整演示 Agent 的工具调用循环与 RAG 检索；
2. 作为测试替身，让单元测试不依赖外部 API，随时可跑。

ScriptedChatModel 补了 GenericFakeChatModel 缺失的 bind_tools。
HashEmbeddings 用文本哈希做种生成定长向量，不依赖 numpy
（core 自带的 DeterministicFakeEmbedding 在没装 numpy 的环境会炸）。
"""

import hashlib
import random
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.tools import BaseTool


class ScriptedChatModel(GenericFakeChatModel):
    """按脚本依次返回消息，并支持 bind_tools 的聊天模型。"""

    def bind_tools(
        self, tools: Sequence[BaseTool], **kwargs: object
    ) -> "ScriptedChatModel":
        """create_agent 会调用该方法绑定工具，脚本化模型直接返回自身。

        是否触发工具调用由预设消息里的 tool_calls 决定，与工具绑定无关。
        """
        return self


class HashEmbeddings(Embeddings):
    """按文本哈希生成定长向量的嵌入替身。

    同一段文本永远得到同一个向量，所以「用原文当 query」一定命中原文，
    单元测试靠这个性质断言检索结果。向量本身没有语义，只保证确定性。
    """

    def __init__(self, size: int = 64) -> None:
        self.size = size

    def _vec(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.size)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

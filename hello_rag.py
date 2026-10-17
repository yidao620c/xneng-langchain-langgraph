"""第 6 篇示例：检索增强 RAG（hello_rag.py）。

把「文档进库」与「按问题取料」拆成九个 demo：Document 本质、加载、
分割、嵌入、向量库、相似检索、检索器、两步式 RAG、Agentic RAG。

嵌入模型来源：环境变量 YI_API_KEY 与 YI_BASE_URL 可用时用真实嵌入
（模型由 EMBED_MODEL 指定，默认 text-embedding-3-small）；未设置时
用本地 HashEmbeddings 替身，向量确定性可复现，不依赖任何 API。
"""

import os
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fake_models import HashEmbeddings, ScriptedChatModel
from langchain.messages import AIMessage


# ---------- 嵌入与模型的来源约定 ----------

def build_embeddings():
    """有 YI 环境变量就用真实嵌入，否则用本地哈希替身。"""
    if os.environ.get("YI_API_KEY") and os.environ.get("YI_BASE_URL"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
            openai_api_base=os.environ["YI_BASE_URL"],
            openai_api_key=os.environ["YI_API_KEY"],
            check_embedding_ctx_length=False,
        )
    return HashEmbeddings(size=64)


def build_model():
    """聊天模型：有 AGENT_MODEL 用真实模型，否则本地替身。"""
    name = os.environ.get("AGENT_MODEL")
    if name:
        from langchain.chat_models import init_chat_model

        params = {}
        if os.environ.get("YI_BASE_URL"):
            params["base_url"] = os.environ["YI_BASE_URL"]
        if os.environ.get("YI_API_KEY"):
            params["api_key"] = os.environ["YI_API_KEY"]
        return init_chat_model(name, **params)
    return ScriptedChatModel(
        messages=iter(
            [
                AIMessage(content="根据笔记，Python 3.9 以上版本即可安装 LangChain。"),
            ]
        )
    )


# ---------- 知识库样例文本 ----------

NOTES = """LangChain 安装与初始化

安装 LangChain 用 pip install langchain，需要 Python 3.9 以上版本。
装完建议先跑一个 init_chat_model 的最小示例确认模型连通。

LangChain 消息体系

对话由消息构成，SystemMessage 装人设，HumanMessage 装用户输入，
AIMessage 装模型输出，ToolMessage 装工具执行结果。

LangChain 检索增强

文档先分割成块，每块算出嵌入向量存进向量库。提问时把问题也变成
向量，按余弦相似度找最相关的几块，塞进提示词再让模型回答。

LangGraph 图编排

LangGraph 用节点和边组织流程，节点是函数，边决定走向，
条件边让流程能按状态分叉，Agent 循环就是一张这样组成的图。
"""


def write_sample(path: Path) -> None:
    """把样例笔记写进临时文件，供加载 demo 使用。"""
    path.write_text(NOTES, encoding="utf-8")


# ---------- demo 1：Document 的本质 ----------

def demo_document() -> None:
    """一个 Document 就是 page_content 加 metadata。"""
    doc = Document(
        page_content="LangChain 用 Document 统一承载各种来源的文本。",
        metadata={"source": "notes.md", "page": 3},
    )
    print("== Document ==")
    print("  内容:", doc.page_content)
    print("  元数据:", doc.metadata)


# ---------- demo 2：加载 ----------

def demo_load(tmp_dir: Path) -> None:
    """加载就是「任何来源都变成 Document 列表」。手写一个 txt 加载器。"""
    file_path = tmp_dir / "notes.txt"
    write_sample(file_path)

    def load_text(path: Path) -> list[Document]:
        text = Path(path).read_text(encoding="utf-8")
        return [Document(page_content=text, metadata={"source": str(path)})]

    docs = load_text(file_path)
    print("== 加载 ==")
    print(f"  {file_path.name} -> {len(docs)} 个 Document，"
          f"{len(docs[0].page_content)} 字符")


# ---------- demo 3：分割 ----------

def demo_split() -> None:
    """RecursiveCharacterTextSplitter 按分隔符递归切，块间留重叠。"""
    write_sample(Path("rag_sample.txt"))
    text = Path("rag_sample.txt").read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=90, chunk_overlap=15,
        separators=["\n\n", "\n", "。", "，"],
    )
    docs = splitter.create_documents([text])
    print("== 分割 chunk_size=90 overlap=15 ==")
    print(f"  {len(text)} 字符 -> {len(docs)} 块")
    for i, d in enumerate(docs, 1):
        print(f"  块 {i}（{len(d.page_content)} 字符）:", d.page_content[:26], "...")
    Path("rag_sample.txt").unlink()


# ---------- demo 4：嵌入 ----------

def demo_embed() -> None:
    """Embeddings 接口就两个方法，入库用 embed_documents，提问用 embed_query。"""
    emb = build_embeddings()
    name = type(emb).__name__
    vectors = emb.embed_documents(["北京今天晴", "上海多云"])
    query = emb.embed_query("天气怎么样")
    print("== 嵌入 ==")
    print(f"  {name}: 文档 {len(vectors)} 条，每条 {len(vectors[0])} 维；"
          f"问题向量 {len(query)} 维")


# ---------- demo 5：向量库 ----------

def demo_store() -> None:
    """InMemoryVectorStore 零部署，dump/load 能把库存成文件。"""
    emb = build_embeddings()
    store = InMemoryVectorStore.from_documents(
        [Document(page_content="LangChain 是做 AI 应用的框架。",
                  metadata={"source": "a.txt"}),
         Document(page_content="LangGraph 用图编排流程。",
                  metadata={"source": "b.txt"})],
        emb,
    )
    print("== 向量库 ==")
    print("  库内条目:", len(store.store))
    store.dump("rag_demo_store.json")
    reloaded = InMemoryVectorStore.load("rag_demo_store.json", embedding=emb)
    print("  落盘重载后条目:", len(reloaded.store))
    Path("rag_demo_store.json").unlink()


# ---------- demo 6：相似检索 ----------

def demo_search() -> None:
    """similarity_search 返回 Document，with_score 附带相似度，越大越相关。"""
    emb = build_embeddings()
    store = InMemoryVectorStore.from_texts(
        ["今天北京天气晴朗。", "上海明天有雨。", "Python 是一门编程语言。"], emb,
    )
    query = "今天北京天气晴朗。"
    hits = store.similarity_search(query, k=2)
    print("== 相似检索 k=2 ==")
    for d in hits:
        print("  命中:", d.page_content)

    scored = store.similarity_search_with_score(query, k=1)
    print(f"  首位相似度: {scored[0][1]:.4f}（1.0 表示完全一致）")


# ---------- demo 7：检索器 ----------

def demo_retriever() -> None:
    """as_retriever 把向量库变成统一接口，MMR 模式兼顾相关性多样性。"""
    emb = build_embeddings()
    store = InMemoryVectorStore.from_texts(
        ["LangChain 提供模型抽象。", "LangChain 提供模型抽象，还支持工具调用。",
         "今天的午餐是面条。"], emb,
    )
    retriever = store.as_retriever(search_kwargs={"k": 2})
    hits = retriever.invoke("LangChain 提供模型抽象。")
    print("== 检索器 k=2 ==")
    print("  命中", len(hits), "条，首条:", hits[0].page_content[:20])

    mmr = store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 3})
    hits2 = mmr.invoke("LangChain 提供模型抽象。")
    print("== MMR 检索 ==")
    for d in hits2:
        print("  命中:", d.page_content[:20])


# ---------- demo 8：两步式 RAG ----------

def demo_two_step() -> None:
    """先检索后生成，检索结果拼进提示词，模型只依据料回答。"""
    emb = build_embeddings()
    store = InMemoryVectorStore.from_texts(
        ["安装 LangChain 需要 Python 3.9 以上版本。",
         "LangGraph 用节点和边组织流程。"], emb,
    )
    question = "装 LangChain 对 Python 版本有什么要求？"
    context_docs = store.similarity_search(question, k=1)
    context = "\n".join(d.page_content for d in context_docs)

    prompt = (
        "只依据下面的资料回答问题，资料里没有就直说不知道。\n\n"
        f"资料：\n{context}\n\n问题：{question}"
    )
    model = build_model()
    answer = model.invoke([SystemMessage("你是简洁的问答助手"), HumanMessage(prompt)])
    print("== 两步式 RAG ==")
    print("  检索到", len(context_docs), "块:", context_docs[0].page_content)
    print("  回答:", answer.content)


# ---------- demo 9：Agentic RAG ----------

def demo_agentic() -> None:
    """把检索器包成工具交给 agent，让模型自己决定查不查、查几轮。"""
    emb = build_embeddings()
    store = InMemoryVectorStore.from_texts(
        ["安装 LangChain 需要 Python 3.9 以上版本。"], emb,
    )
    retriever = store.as_retriever(search_kwargs={"k": 1})
    search_tool = create_retriever_tool(
        retriever, "search_notes", "搜索 LangChain 学习笔记，返回相关段落",
    )
    agent = create_agent(
        model=ScriptedChatModel(messages=iter([
            AIMessage(content="好的", tool_calls=[{
                "name": "search_notes",
                "args": {"query": "Python 版本要求"},
                "id": "call_1", "type": "tool_call",
            }]),
            AIMessage(content="根据笔记，安装 LangChain 需要 Python 3.9 以上版本。"),
        ])),
        tools=[search_tool],
        system_prompt="回答问题前先查笔记，只依据笔记内容作答。",
    )
    result = agent.invoke({"messages": [HumanMessage("装 LangChain 要什么版本的 Python？")]})
    print("== Agentic RAG ==")
    for m in result["messages"]:
        if getattr(m, "tool_calls", None):
            for c in m.tool_calls:
                print(f"  模型发起检索: {c['name']}({c['args']})")
        if m.type == "tool":
            print(f"  工具返回: {str(m.content)[:40]}")
    print("  回答:", result["messages"][-1].content)


# ---------- 入口 ----------

DEMOS = {
    "document": demo_document,
    "load": demo_load,
    "split": demo_split,
    "embed": demo_embed,
    "store": demo_store,
    "search": demo_search,
    "retriever": demo_retriever,
    "twostep": demo_two_step,
    "agentic": demo_agentic,
}


def main() -> None:
    import sys
    import tempfile

    names = sys.argv[1:] or list(DEMOS)
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            if name == "load":
                DEMOS[name](Path(tmp))
            else:
                DEMOS[name]()


if __name__ == "__main__":
    main()

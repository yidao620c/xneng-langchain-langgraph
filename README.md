# xneng-langchain-langgraph · 源码目录

公众号「飞污熊」系列教程《使用LangChain+LangGraph开发AI应用》全部 12 篇的配套源码。前 10 篇按知识点递进（模型、消息、工具、Agent、RAG、记忆、LangGraph 基础与进阶、多智能体），最后 2 篇是完整案例（个人知识库、个人旅游助手）。

所有示例都配了脚本化测试替身（`fake_models.py`），**不设 API key 也能直接跑通**；填好环境变量后可一键切换为真实模型。

## 技术栈

| 依赖 | 版本 | 用途 |
| --- | --- | --- |
| Python | >= 3.14 | 运行环境 |
| uv | 最新 | 依赖管理与虚拟环境 |
| langchain | 1.4.3 | 核心框架（模型、消息、工具、Agent、RAG、记忆） |
| langchain-core | 1.6.12 | 底层抽象（消息类型、可运行对象） |
| langchain-openai | 1.7.2 | OpenAI 兼容模型接入 |
| langgraph | 1.2.13 | 图编排（状态图、子图、持久化、多智能体） |
| langgraph-checkpoint-sqlite | 3.1.2 | SQLite 持久化检查点 |
| langchain-text-splitters | >= 1.1.3 | 文档切分 |
| numpy | >= 2.5.4 | 向量相似度计算 |
| pytest | 9.1.1 | 单元测试（全仓 187 条） |

## 快速开始

环境要求：Python 3.14+ 与 [uv](https://docs.astral.sh/uv/)。

```bash
# 1. 安装依赖（自动创建虚拟环境）
uv sync

# 2. 不带任何配置，直接跑第一个 Agent（使用内置脚本化模型）
uv run python hello_agent.py

# 3. 跑全部测试
uv run pytest
```

### 接入真实模型（可选）

复制 `.env.example` 为 `.env` 并填入值（`.env` 已被 gitignore，不会提交）：

```bash
cp .env.example .env
```

两种接入方式：

- **直连提供商**：`AGENT_MODEL=openai:gpt-5.5`（格式为 `提供商:模型名`），对应 key 填 `OPENAI_API_KEY`
- **统一网关（API易）**：填 `YI_API_KEY` 与 `YI_BASE_URL`

模型名支持 `--model` 参数逐次覆盖，例如 `uv run python ask.py "你好" --model openai:gpt-5.5`。不指定时一律回落到脚本化模型，保证示例永远能跑。

## 目录结构

```
source/
├── fake_models.py        # 测试替身：ScriptedChatModel + HashEmbeddings（无 key 可跑的关键）
├── hello_*.py            # 10 个知识点示例文件（第 1 篇 2 个 + 第 2-4、6-10 篇各 1 个）
├── agent_deep.py         # 第 5 篇 Agent 深入示例
├── ask.py / extract.py 等 # 9 个配套小案例（每篇 1 个，均为根目录下的单文件脚本）
├── library.py            # 案例：个人知识库（第 11 篇）
├── travel.py             # 案例：个人旅游助手（第 12 篇）
├── notes/                # 第 11 篇演示笔记（知识库语料）
├── rag_notes/            # 第 6 篇演示笔记（RAG 语料）
├── tests/                # 单元测试，与源码文件一一对应
├── pyproject.toml        # uv 项目定义（含清华 PyPI 镜像）
└── uv.lock               # 依赖锁文件（51 包）
```

## 12 篇教程与源码对照

每篇一个示例系列 + 一个可交互的小案例。示例可整体跑（`uv run python 文件名`），也支持按 demo 名单跑（`uv run python 文件名 demo名`）。

| 篇 | 主题 | 示例 | 案例 | 案例跑法示例 |
| --- | --- | --- | --- | --- |
| 1 | 项目初始化 | `hello_agent.py` / `hello_graph.py` | — | `uv run python hello_agent.py` |
| 2 | 模型抽象 | `hello_models.py` | `ask.py` | `uv run python ask.py "用一句话解释RAG"` |
| 3 | 消息与结构化输出 | `hello_messages.py` | `extract.py` | `uv run python extract.py "张三下周二去上海出差"` |
| 4 | 工具定义与调用 | `hello_tools.py` | `toolbox.py` | `uv run python toolbox.py --list` / `toolbox.py --call get_weather --args '{"city":"杭州"}'` |
| 5 | Agent 深入 | `agent_deep.py` | `briefing.py` | `uv run python briefing.py "沪深300最近走势如何" --stream` |
| 6 | 检索增强 RAG | `hello_rag.py` | `kb.py` | `uv run python kb.py ingest rag_notes/` → `kb.py ask "什么是RAG"` |
| 7 | 记忆系统 | `hello_memory.py` | `assistant.py` | `uv run python assistant.py --demo` / `assistant.py --mems` |
| 8 | LangGraph 基础 | `hello_graph_deep.py` | `textcheck.py` | `uv run python textcheck "这个产品真的太好用了！！" --trace` |
| 9 | LangGraph 进阶 | `hello_persist.py` | `release.py` | `uv run python release.py "升级支付SDK" --db release.db` |
| 10 | 多智能体编排 | `hello_multi.py` | `newsroom.py` | `uv run python newsroom.py "AI手机兴起的观察" --trace` |
| 11 | 案例：个人知识库 | — | `library.py` | `uv run python library.py ingest notes/` → `library.py ask "向量库存的是什么" --show-sources` |
| 12 | 案例：个人旅游助手 | — | `travel.py` | `uv run python travel.py plan "十一去杭州玩两天" --trace` |

几个案例的补充说明：

- `kb.py` / `library.py`：首次使用先 `ingest` 导入语料，再 `ask` 提问；`library.py` 支持引用溯源（`--show-sources` 展开引用块原文），重复导入自动按文件翻新、不留僵尸块
- `release.py`：人机协同审批示例，带 `--db` 可将审批状态落 SQLite，重启进程也能接着审；`--auto-approve` 用于无人值守演示
- `briefing.py`：`--stream` 逐步观察 Agent 执行轨迹；网关不支持 `tool_choice` 时用 `--plain` 降级
- `assistant.py` / `travel.py`：演示长期记忆（跨线程保存与召回用户偏好），`--demo` 一键跑完整闭环
- `textcheck.py` / `newsroom.py`：`--trace` 流式打印图中每一步的状态变化，适合观察 LangGraph 的执行过程

## 无 key 运行的秘密：fake_models.py

- `ScriptedChatModel`：按脚本顺序返回预设回复，支持绑定工具并返回工具调用，用于演示 Agent 循环
- `HashEmbeddings`：基于 MD5 种子生成确定性向量，同一文本必然自命中，用于演示 RAG 检索

全部 187 条测试都构建在这两个替身上，CI 与离线环境均可用。

## 测试

```bash
uv run pytest              # 跑全部 187 条
uv run pytest tests/test_travel.py -v   # 只跑某一篇的测试
```

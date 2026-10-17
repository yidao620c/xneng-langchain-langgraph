# LangChain 安装与初始化

LangChain 用 pip 安装，命令是 pip install langchain，需要 Python 3.10 以上版本。

推荐用 uv 管理虚拟环境，uv add langchain 一条命令完成安装并写入锁文件。

安装完成后设置两个环境变量，OPENAI_API_KEY 和 OPENAI_BASE_URL，模型调用就走自定义网关。

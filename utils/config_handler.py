"""
yaml
k: v
"""
import yaml
from utils.path_tool import get_abs_path

#加载和rag相关的配置文件
#这段代码定义了一个加载 RAG 配置的函数，通过默认参数获取config/rag.yml的绝对路径，
# 使用utf-8编码读取文件，再通过yaml库安全解析配置文件内容，最终返回 Python 字典格式的配置信息，
# 方便项目中读取 RAG 相关参数。
#with open(...) : Python 安全打开文件的写法（自动关闭文件，不会报错）。
#config_path :要打开的文件路径（rag.yml）。
#"r"  read，只读模式打开。
# encoding=encoding  使用 utf-8 编码读取。
# as f  把打开的文件对象取名为 f。

#yaml.load(f)  把 .yml / .yaml 配置文件内容读取并解析成 Python 字典。
#Loader=yaml.FullLoader  安全解析器，防止 yaml 漏洞，官方推荐写法。

# 假设你的 rag.yml 内容是：
# model: qwen-max
# temperature: 0.1
# top_k: 5

# 函数返回的就是 Python 字典：
# {
#     "model": "qwen-max",
#     "temperature": 0.1,
#     "top_k": 5
# }

def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

#加载和向量数据库相关的配置文件
def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

#加载和提示词相关的配置文件
def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

#加载和agent相关的配置文件
def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# 把结果加载存入变量中,通过变量可以获取结果  相当于Java中的 int[] a = 函数（）; a[1]=结果的内容.
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


if __name__ == '__main__':
    print(rag_conf["chat_model_name"])

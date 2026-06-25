from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf

#定义了一个抽象类 BaseModelFactory，(ABC)是 Abstract Base Class（抽象基类）的缩写。它的作用是：告诉 Python “这个类是一个抽象类，不能直接拿来创建对象，只能被其他类继承”
#因为我们要定义一个“规范”，规定所有子类都必须实现某些方法。ABC 是从 Python 标准库 abc 模块导入的，需要提前写 from abc import ABC, abstractmethod。
#定义生产规范--必须用generate方法
class BaseModelFactory(ABC):
    @abstractmethod   #声明方法是抽象方法
    # Optional[Embeddings | BaseChatModel]  Optional[...] 表示可选的，说明返回值类型是Embeddings 或 BaseChatModel 或 None
    def generator(self) -> Optional[Embeddings | BaseChatModel]:    # 定义一个抽象方法generator，并说明返回值类型
        pass  # 表示空语句，什么都不做。因为抽象方法没有实现体，只是占个位置。

#生产聊天模型工程
class ChatModelFactory(BaseModelFactory):  # (BaseModelFactory) 表示该类继承BaseModelFactory
    def generator(self) -> Optional[Embeddings | BaseChatModel]:  #重写方法，并加上自己的语句，返回ChatTongyi模型
        return ChatTongyi(model=rag_conf["chat_model_name"])

#生产嵌入模型工程
class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()

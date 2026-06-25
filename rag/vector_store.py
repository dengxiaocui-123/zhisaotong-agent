import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List,Optional
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger

import pickle
import hashlib

from rank_bm25 import BM25Okapi
# from FlagEmbedding import FlagReranker
from sentence_transformers import CrossEncoder
import jieba
from pydantic import ConfigDict, PrivateAttr


#完成向量存储服务

class VectorStoreService:
    #下面的函数相当于Java的如下：
    #public class VectorStoreService {
#     // 成员变量（全局属性）
#     private Chroma vectorStore;
#
#     // 构造函数
#     public VectorStoreService() {
#         // 给成员变量赋值
#         this.vectorStore = new Chroma(...);
#     }
# }
    # 创建一个 Chroma 向量数据库 对象，并把它存到当前类的 vector_store 属性里，方便后面做检索。
    #相当于java中的 Student s = new Student(参数)   self.vector_store 就是s  new Student就是Chroma
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],  #向量数据库中存入数据的表名
            embedding_function=embed_model,                  #嵌入模型--从模型工厂factory中获取（工厂设计模式）
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),  #存储路径
        )
        #这段代码是在创建一个文本分割器（Text Splitter），专门用于将长文本切分成较小的“块（chunk）”，以便后续存入向量数据库（如 Chroma）或交给大模型处理。
        #1. RecursiveCharacterTextSplitter 是什么？
        #它来自 langchain 库（一个流行的 LLM 应用开发框架），位于 langchain.text_splitter 模块。
        #作用：用一种“递归”的方式，按照指定的分隔符（如换行符、句号、逗号等）尽可能地把文本切分到不超过 chunk_size 的大小，同时保持语义的完整性（尽量在自然边界处切割）。
        #“递归”的意思：它会先尝试用最大的分隔符（比如 "\n\n"）切分，如果切出来的块还是太大，就再用小一点的分隔符（比如 "\n"）继续切分，以此类推，直到每个块都符合大小要求。
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],          #每个文本块的最大长度
            chunk_overlap=chroma_conf["chunk_overlap"],    #相邻两个块之间重叠的字符数。重叠可以避免关键信息恰好被切断在边界处，提高检索效果
            separators=chroma_conf["separators"],          #分隔符优先级列表，比如 ["\n\n", "\n", "。", "，", " "]。切分时按顺序尝试使用这些分隔符
            length_function=len,                           #计算文本长度的函数。这里传入 len，表示用 Python 内置的 len() 计算字符数
        )

        # ========== 新增：BM25 索引初始化 ==========
        self.bm25_docs = []  # 文档列表：[{page_content, metadata, id}]
        self.bm25_index = None  # BM25Okapi 实例
        self.bm25_persist_path = get_abs_path(chroma_conf["bm25_persist_file"])
        self._load_bm25()  # 启动时加载持久化索引

    #定义一个获取检索器的方法
    def get_retriever(self):
        #向量数据库对象 self.vector_store 调用as_retriever方法，作用是将“数据库对象”转换成一个“检索器对象”。专门用于接收查询文本，然后返回最相似的文档块
        # 参数search_kwargs 是一个字典，这里只设置了一个键 "k"
        # k 表示“返回最相关的文档块的数量（top‑k）”。例如，k=3 表示每次查询返回最相似的 3 个文本块。
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    # ==================== 新增：BM25 相关方法 ====================
    def _tokenize(self, text: str) -> List[str]:
        # 用 jieba.cut + list，兼容任何版本的 jieba
        words = list(jieba.cut(text))
        return [w for w in words if w and not w.isspace()]
        #return [w for w in jieba.lcut(text) if w and not w.isspace()]

    def _load_bm25(self):
        if not os.path.exists(self.bm25_persist_path):
            return
        with open(self.bm25_persist_path, "rb") as f:
            data = pickle.load(f)
        self.bm25_docs = data.get("docs", [])
        if self.bm25_docs:
            tokens = [self._tokenize(d["page_content"]) for d in self.bm25_docs]
            self.bm25_index = BM25Okapi(tokens)
        logger.info(f"[BM25] 已加载 {len(self.bm25_docs)} 篇文档")

    def _save_bm25(self):
        with open(self.bm25_persist_path, "wb") as f:
            pickle.dump({"docs": self.bm25_docs}, f)

    def _add_to_bm25(self, split_document: List[Document]):
        for doc in split_document:
            self.bm25_docs.append({
                "page_content": doc.page_content,
                "metadata": dict(doc.metadata or {}),
                "id": doc.id or hashlib.md5(doc.page_content.encode()).hexdigest(),
            })
        tokens = [self._tokenize(d["page_content"]) for d in self.bm25_docs]
        self.bm25_index = BM25Okapi(tokens)
        self._save_bm25()

    # ========== 新增：获取混合检索器 ==========
    def get_hybrid_retriever(self):
        return HybridRetriever(self)

    #加载知识库
    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """
        #用于检查给定的 MD5 哈希值是否已经存在于一个存储文件中。如果不存在，则返回 False，表示未处理过；如果存在，返回 True，表示已处理过。
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])): #如果路径所指向的文件不存在
                # 创建一个空的MD5记录文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()  # w 写
                return False            # md5 没处理过
            #如果文件存在，执行下面的语句
            #使用 with 语句的好处是代码块结束后会自动关闭文件，不用手动写 .close()。
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:  # r 读
                for line in f.readlines():
                    line = line.strip()  #去掉该行字符串首尾的空白字符，包括换行符 \n、空格、制表符等。这样就能得到干净的MD5字符串
                    if line == md5_for_check:
                        return True     # md5 处理过

                return False            # md5 没处理过

        #将str保存到md5.txt文件（记录哪些md5已经被处理过了）中
        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:  #a 追加
                f.write(md5_for_check + "\n")

        #根据文件扩展名（后缀）选择不同的文档加载器，读取文件内容后返回文档对象列表。
        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        #获取符合类型的文件列表
        #相当于 a: int = 函数（）
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        #从符合类型的文件列表中依次获取文件，计算md5值，判断是否已经处理，
        # 没处理就加载文档，获取文件内容，然后判断文件是否有有效内容，有就进行文本切割、
        # 然后判断切割后的内容是否有有效内容，有再存入向量库，然后将处理过的文件的md5值存入md5文件中
        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:   #判断文档是不是 没有效内容
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue
                #文本有内容，进行文本切割
                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document: #判断分片后是不是 没有效内容
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)
                # ========== 新增：同步写入 BM25 索引（与向量库共用 MD5 去重）==========
                self._add_to_bm25(split_document)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue

# =================================================================
# 新增：混合检索器（向量 + BM25 + Rerank）
# =================================================================
class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # -- 在类顶部声明字段，Pydantic 才会接受它们 --
    vs: object = None  # VectorStoreService 实例
    vector_k: int = 0
    bm25_k: int = 0
    fusion_k: int = 0
    rerank_top_n: int = 0
    vector_w: float = 0.0
    bm25_w: float = 0.0
    # FlagReranker 标记为私有属性，Pydantic 不校验它的类型
    _reranker = PrivateAttr(default=None)

     # vs：自定义形参名，接收外部传入的对象。
     # : "VectorStoreService"：字符串形式的类型注解，标注vs要求是VectorStoreService类的实例。
     # 用引号包裹是前向引用：定义当前类时，VectorStoreService还未加载 / 定义，加引号避免代码报错。
    def __init__(self, vs: "VectorStoreService"):
        super().__init__()
        self.vs = vs   # 接收外部传入的VectorStoreService实例,如果不这样做，vs只能在__init__方法中使用，不能在其他方法中（比如_vector_search）使用
        self.vector_k = int(chroma_conf["vector_k"])
        self.bm25_k = int(chroma_conf["bm25_k"])
        self.fusion_k = int(chroma_conf["fusion_k"])
        self.rerank_top_n = int(chroma_conf["rerank_top_n"])
        self.vector_w = float(chroma_conf["vector_weight"])
        self.bm25_w = float(chroma_conf["bm25_weight"])
        # Reranker 初始化（FlagEmbedding 会自动从 HuggingFace 下载或从缓存加载）
        self._reranker = CrossEncoder(chroma_conf["reranker_model_name"])

     # ---- 子检索 1：向量检索 ----
     # 这里调用了向量数据库self.vs.vector_store 里的方法similarity_search_with_score，方法返回 List[(Document, distance_score)] ，
     # 也就是文档列表和与它们的相似度得分（L2距离），之间调用.as_retriever底层也调用了这个方法，只不过只返回了文档列表。
     #因为混合检索需要需要把「向量检索的得分」和「BM25 的得分」做加权融合，所以用的这个方法
    def _vector_search(self, query: str) -> List[Document]:
        docs = self.vs.vector_store.similarity_search_with_score(query, k=self.vector_k)
        result = []
        for doc, dist in docs:
            score = 1.0 / (1.0 + float(dist))   # L2 距离 → 相似度
            result.append(Document(
                page_content=doc.page_content,
                metadata={**(doc.metadata or {}), "_vec": score},
            ))
        return result

    # ---- 子检索 2：BM25 检索 ----
    def _bm25_search(self, query: str) -> List[Document]:
        if self.vs.bm25_index is None or not self.vs.bm25_docs:
            return []
        scores = self.vs.bm25_index.get_scores(self.vs._tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:self.bm25_k]
        max_s, min_s = float(max(scores)), float(min(scores))
        span = (max_s - min_s) or 1.0
        out = []
        for idx, raw in ranked:
            if raw <= 0 and out:
                continue
            d = self.vs.bm25_docs[idx]
            out.append(Document(
                page_content=d["page_content"],
                metadata={**d["metadata"], "_bm25": (float(raw) - min_s) / span},
                id=d["id"],
            ))
        return out

    # ---- 合并：文本去重 + 加权融合 ----
    def _merge(self, vec_docs: List[Document], bm_docs: List[Document]) -> List[Document]:
        merged = {}                           # key: 文本内容
        for doc in vec_docs:
            key = doc.page_content.strip()
            e = merged.setdefault(key, {"doc": doc, "vs": 0.0, "bm": 0.0})
            e["vs"] = max(e["vs"], float(doc.metadata.get("_vec", 0.0)))
        for doc in bm_docs:
            key = doc.page_content.strip()
            e = merged.setdefault(key, {"doc": doc, "vs": 0.0, "bm": 0.0})
            e["bm"] = max(e["bm"], float(doc.metadata.get("_bm25", 0.0)))
        # 按加权得分排序
        items = [(e["doc"], e["vs"] * self.vector_w + e["bm"] * self.bm25_w) for e in merged.values()]
        items.sort(key=lambda x: x[1], reverse=True)
        return [Document(page_content=d.page_content, metadata={**d.metadata, "_fused": s}, id=d.id)
                for d, s in items[:self.fusion_k]]

    # ---- 重排序：BGE Reranker ----
    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self._reranker.predict(pairs)  # 返回 numpy 数组
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        pairs = sorted(zip(docs, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)[:self.rerank_top_n]
        return [Document(page_content=d.page_content, metadata={**d.metadata, "_rerank": s}, id=d.id)
                for d, s in pairs]

    # ---- LangChain 标准接口（用 .invoke 调用）----
    def _get_relevant_documents(self, query: str, *,
                                run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        vec_docs = self._vector_search(query)
        bm_docs = self._bm25_search(query)
        merged = self._merge(vec_docs, bm_docs)
        return self._rerank(query, merged)

#测试功能
if __name__ == '__main__':
    #创建一个向量存储服务对象
    vs = VectorStoreService()
   #加载知识库
    vs.load_document()
   #创建一个检索器
    retriever = vs.get_hybrid_retriever()
    #retriever = vs.get_retriever()
   #通过检索器
    res = retriever.invoke("陀螺仪")
    for r in res:
        print(r.page_content)
        print("-"*20)

import os,hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 获取文件的md5的十六进制字符串  把数据存入向量数据库中时用于去重判断
def get_file_md5_hex(filepath: str):

    if not os.path.exists(filepath):  #如果路径所指的地方不存在文件/文件夹
        logger.error(f"[md5计算]文件{filepath}不存在")
        return

    if not os.path.isfile(filepath): #如果路径所指的地方不是文件
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return

    #创建一个 MD5 哈希对象
    # # 创建 MD5 对象
    # md5_obj = hashlib.md5()
    #
    # # 向对象中喂入数据（必须是 bytes 类型）
    # md5_obj.update(b"hello world")  # 方式1：直接传入 bytes
    # md5_obj.update("你好".encode("utf-8"))  # 方式2：字符串先编码成 bytes
    #
    # # 获取最终的 MD5 值（十六进制字符串）
    # md5_hex = md5_obj.hexdigest()
    # print(md5_hex)  # 输出类似 "5eb63bbbe01eeed093cb22bb8f5acdc3"
    #
    # # 也可以一步到位（直接对数据计算）
    # md5_once = hashlib.md5(b"hello world").hexdigest()
    md5_obj = hashlib.md5()

    chunk_size = 4096  # 4KB分片，避免文件过大爆内存
    # :=
    # 把右边的值赋值给左边变量  chunk = f.read(...)
    # 把这个值作为判断条件 while chunk:

    try:
        with open(filepath, "rb") as f:  # 读取文件路径所指的文件，而且必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

            """
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest() #把之前分片读取、更新好的文件内容，计算出最终的 MD5 字符串（32 位十六进制）。
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None

# 设置我们读取的文件夹后缀 比如 pdf、txt，那么就会只读取并返回这两个后缀的文件
#这个函数的作用是遍历指定目录（str参数表示要遍历的路径，比如"data/"），筛选出指定后缀名的文件，(过滤掉子文件夹和不符合后缀的文件)
# 接收文件夹路径和允许的文件类型元组作为参数，
# 返回该目录下所有符合类型的文件列表，
def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):  # 返回文件夹内的文件列表（允许的文件后缀）
    files = []

    if not os.path.isdir(path):  #判断传入的 path 是不是一个【不存在】或【不是文件夹】的东西。
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types

    for f in os.listdir(path): #遍历文件夹
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files) #将list转成元组，让其不能改变

#加载PDF文档
def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()

#加载txt文档
def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()
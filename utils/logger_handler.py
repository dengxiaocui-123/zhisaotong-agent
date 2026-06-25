import logging
from utils.path_tool import get_abs_path
import os
from datetime import datetime

# 定义日志保存的根目录
LOG_ROOT = get_abs_path("logs")

# 确保日志的目录存在  exist_ok=True表示如果目录不存在则创建
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的格式配置  日志级别：error info debug（从高到低 如果是error那么只输出error,级别为debug则都输出）
DEFAULT_LOG_FORMAT = logging.Formatter(
    # 日志输出时间     日志名字     日志级别         文件名，哪一行             正文
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file = None,
) -> logging.Logger:
    logger = logging.getLogger(name)  #创建Logger（日志）对象
    logger.setLevel(logging.DEBUG)    #设置日志级别 高于级别的会输出

    # 避免重复添加Handler
    if logger.handlers:
        return logger

    # 控制台Handler
    console_handler = logging.StreamHandler()  #创建一个控制台处理器对象
    console_handler.setLevel(console_level)    #设置控制台输出日志的级别
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)   #设置输出的日志格式

    logger.addHandler(console_handler)  #让logger这个日志器，把日志输出到控制台

    # 文件Handler
    if not log_file:        # 日志文件的存放路径
        #datetime.now()：获取当前系统时间  .strftime('%Y%m%d')：把时间格式化成字符串
        #最终拼出来的效果： C:\xxx\logs\agent_20250529.log
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')  #创建一个文件处理器对象，并设置编码方式
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler) #让logger这个日志器，把日志输出到文件中

    return logger


# 快捷获取日志器
logger = get_logger()


if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")

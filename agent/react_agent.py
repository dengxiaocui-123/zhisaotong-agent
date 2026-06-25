from langchain.agents import create_agent
from langchain_classic.memory import ConversationBufferWindowMemory # 导入记忆模块
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from typing import Dict, Optional
import uuid
from datetime import datetime, timedelta

# ========== 记忆类型配置 ==========
# 可选择的记忆类型：
# 1. ConversationBufferMemory - 完整记忆所有对话
# 2. ConversationBufferWindowMemory - 只保留最近k条消息（推荐）
# 3. ConversationSummaryMemory - 自动总结历史对话
MEMORY_TYPE = "window"  # 切换记忆类型："full", "window", "summary"
WINDOW_SIZE = 5  # 窗口大小（仅window模式有效）
SESSION_TIMEOUT_MINUTES = 30  # 会话超时时间（分钟）30分钟


class SessionMemoryManager:
    """会话级别的记忆管理器，支持多用户隔离"""

    def __init__(self):
        self.sessions: Dict[str, object] = {}  # 存储各会话的记忆对象
        self.last_access: Dict[str, datetime] = {}  # 记录会话最后访问时间

    def _create_memory(self):
        """根据配置创建对应类型的记忆对象"""
        if MEMORY_TYPE == "full":
            from langchain.memory import ConversationBufferMemory
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        elif MEMORY_TYPE == "window":
            return ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=WINDOW_SIZE
            )
        elif MEMORY_TYPE == "summary":
            from langchain.memory import ConversationSummaryMemory
            return ConversationSummaryMemory(
                llm=chat_model,
                memory_key="chat_history",
                return_messages=True
            )
        else:
            # 默认使用窗口记忆
            return ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=WINDOW_SIZE
            )

    def get_memory(self, session_id: str):
        """获取或创建会话记忆"""
        self._cleanup_expired()
        self.last_access[session_id] = datetime.now()

        if session_id not in self.sessions:
            self.sessions[session_id] = self._create_memory()
        return self.sessions[session_id]

    def create_new_session(self) -> str:
        """创建新会话并返回会话ID"""
        session_id = str(uuid.uuid4())
        self.get_memory(session_id)  # 触发创建
        return session_id

    def _cleanup_expired(self):
        """清理过期会话"""
        now = datetime.now()
        expired = [sid for sid, last in self.last_access.items()
                   if now - last > timedelta(minutes=SESSION_TIMEOUT_MINUTES)]

        for sid in expired:
            del self.sessions[sid]
            del self.last_access[sid]

    def get_session_list(self):
        """获取所有活跃会话列表"""
        self._cleanup_expired()
        sessions = []
        for session_id, memory in self.sessions.items():
            # 获取会话的最近访问时间
            last_access = self.last_access.get(session_id, datetime.now())
            # 获取会话的第一条消息作为标题
            history = memory.load_memory_variables({}).get("chat_history", [])
            if history:
                first_message = history[0].content if hasattr(history[0], 'content') else str(history[0])
                title = first_message[:30] + "..." if len(first_message) > 30 else first_message
            else:
                title = "空会话"

            sessions.append({
                "session_id": session_id,
                "title": title,
                "last_access": last_access.strftime("%Y-%m-%d %H:%M:%S")
            })

        # 按最后访问时间排序（最新的在前）
        sessions.sort(key=lambda x: x["last_access"], reverse=True)
        return sessions

    def get_session_history(self, session_id: str):
        """获取指定会话的历史消息"""
        if session_id not in self.sessions:
            return []

        memory = self.sessions[session_id]
        chat_history = memory.load_memory_variables({}).get("chat_history", [])
        messages = []
        for msg in chat_history:
            role = "assistant" if hasattr(msg, 'type') and msg.type == "ai" else "user"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            messages.append({"role": role, "content": content})
        return messages

#创建ReAct智能体并流式输出（支持多轮对话记忆和多用户隔离）
class ReactAgent:
    def __init__(self):  #创建智能体
        self.session_manager = SessionMemoryManager()  # 初始化会话管理器
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    # def execute_stream(self, query: str):
    #     #定义一个字典，之后要注入agent中
    #     input_dict = {
    #         "messages": [
    #             {"role": "user", "content": query},
    #         ]
    #     }
    #
    #     # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
    #     for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
    #         latest_message = chunk["messages"][-1]
    #         if latest_message.content:
    #             yield latest_message.content.strip() + "\n"
            #yield 使该方法成为一个生成器函数，每产生一条有效消息就立即输出（流式输出）。

        # 下面是yield的示例
        # def execute_stream(self, query):
        #     for i in range(3):
        #         yield f"第{i + 1}块内容\n"
        #
        # # 调用
        # gen = execute_stream("test")  # 得到生成器对象，此时函数未执行
        #
        # # 方式1：for 循环获取
        # for chunk in gen:
        #     print(chunk, end="")  # 依次输出：第1块内容\n 第2块内容\n 第3块内容\n
        #
        # # 方式2：手动调用 next()
        # gen = execute_stream("test")
        # print(next(gen))  # 第1块内容
        # print(next(gen))  # 第2块内容
        # print(next(gen))  # 第3块内容
        # # 再调用 next(gen) 会抛出 StopIteration

    def execute_stream(self, query: str, session_id: Optional[str] = None):
        """
        执行带记忆的流式对话

        Args:
            query: 用户查询内容
            session_id: 会话ID（可选，不传则创建新会话）

        Returns:
            流式响应生成器
        """
        # 获取或创建会话
        if not session_id:
            session_id = self.session_manager.create_new_session()

        # 获取该会话的记忆
        memory = self.session_manager.get_memory(session_id)

        # 从记忆中加载历史对话
        chat_history = memory.load_memory_variables({}).get("chat_history", [])

        # 构建消息列表
        messages = []
        for msg in chat_history:
            # 判断消息角色
            if hasattr(msg, 'type') and msg.type == "ai":
                role = "assistant"
            else:
                role = "user"
            #hasattr 是 Python 的一个内置函数，用于检查一个对象是否拥有指定的属性或方法。
            # 语法：
            # hasattr(object, name)
            # object：要检查的对象
            # name：属性名的字符串（如 'type'）
            # 返回值：
            # 如果对象存在该属性，返回 True；否则返回 False。

            # 获取消息内容
            content = msg.content if hasattr(msg, 'content') else str(msg)
            # 这行代码是一个条件表达式（三元运算符），用于安全地提取消息的文本内容。
            # 逻辑拆解：
            # hasattr(msg, 'content')：检查msg对象是否拥有content属性。
            # 如果为True，则取msg.content作为content的值。
            # 如果为False，则调用str(msg)将msg对象转换成普通字符串，作为content的值。

            messages.append({"role": role, "content": content}) #将历史对话中的每一条消息添加到messages列表中

        # 添加当前查询
        messages.append({"role": "user", "content": query})  #将当前用户查询添加到messages列表中

        input_dict = {
            "messages": messages
        }

        # 收集完整响应用于保存到记忆
        full_response = ""
        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                full_response += latest_message.content
                yield latest_message.content.strip() + "\n"

        # 将对话保存到记忆
        if full_response:
            memory.save_context({"input": query}, {"output": full_response.strip()})


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)

    # ========== 测试多轮对话记忆 ==========
    print("=" * 60)
    print("测试1: 多轮对话记忆")
    print("=" * 60)

    # 创建会话
    session_id = agent.session_manager.create_new_session()
    print(f"会话ID: {session_id}")
    print()

    # 第一轮对话
    print("用户: 推荐一款小户型扫地机器人")
    for chunk in agent.execute_stream("推荐一款小户型扫地机器人", session_id):
        print(chunk, end="", flush=True)
    print()

    # 第二轮对话（引用上文）
    print("用户: 它的续航怎么样？")
    for chunk in agent.execute_stream("它的续航怎么样？", session_id):
        print(chunk, end="", flush=True)
    print()

    # ========== 测试多用户隔离 ==========
    print("\n" + "=" * 60)
    print("测试2: 多用户隔离")
    print("=" * 60)

    # 用户A的对话
    session_a = agent.session_manager.create_new_session()
    print(f"\n用户A - 会话ID: {session_a}")
    print("用户A: 扫地机器人需要定期维护吗？")
    for chunk in agent.execute_stream("扫地机器人需要定期维护吗？", session_a):
        print(chunk, end="", flush=True)
    print()

    # 用户B的对话（新会话）
    session_b = agent.session_manager.create_new_session()
    print(f"\n用户B - 会话ID: {session_b}")
    print("用户B: 扫地机器人有哪些品牌推荐？")
    for chunk in agent.execute_stream("扫地机器人有哪些品牌推荐？", session_b):
        print(chunk, end="", flush=True)
    print()

    # 用户A继续对话（验证记忆保持）
    print(f"\n用户A继续 - 会话ID: {session_a}")
    print("用户A: 维护频率是多久一次？")
    for chunk in agent.execute_stream("维护频率是多久一次？", session_a):
        print(chunk, end="", flush=True)
    print()

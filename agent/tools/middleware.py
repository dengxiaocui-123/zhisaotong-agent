from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger

#agent中间件类

#监控工具的执行
#@wrap_tool_call
# @wrap_tool_call 是一个装饰器，通常来自 LangGraph 或类似的 Agent 框架（如 langgraph.prebuilt 中的工具调用中间件 API）。
# 它的作用是将一个普通函数转换为“工具调用中间件”，让你能够在每个工具执行前后插入自定义逻辑，而无需修改工具本身的代码。

#总结：
#这个 monitor_tool 是一个工具调用监控中间件，被 @wrap_tool_call 装饰后，会自动拦截 Agent 对任意工具的调用。
# 它主要做三件事：记录完整的调用日志、统一处理异常、以及在调用特定工具 fill_context_for_report 时向运行时上下文注入 report=True 标记，
# 为后续动态切换提示词或行为模式提供依据。这种设计实现了横切关注点与业务逻辑的解耦，是构建可观测、可扩展 Agent 系统的常见模式

@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,  #ToolCallRequest 是它的类型，表示这个请求单里包含了：要调用哪个工具(name)、参数是什么(args)、运行时上下文(runtime.context)等信息。
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],

        #Callable 是 Python 内置的一个特殊标记，用来表示 “一个函数”。
        #任何你可以用括号调用并传参的东西，比如 my_function()，都可以称为 Callable。
        # Callable[[ToolCallRequest], ToolMessage | Command] 表示这个 Callable 的详细规格
        # [ToolCallRequest]代表 这个 handler 接受一个参数，并且这个参数的类型必须是 ToolCallRequest
        #ToolMessage | Command 表示返回值类型，意思是：handler 执行完之后，要么返回一个 ToolMessage 对象，要么返回一个 Command 对象。
        #ToolMessage 是工具返回给 Agent 的正常消息（比如 "城市深圳天气为晴天"）。
        #Command 可能是控制 Agent 下一步动作的指令（比如 "停止"、"继续" 等）。

        #总结：handler 是一个函数，这个函数要求传一个 ToolCallRequest，并且它会返回一个 ToolMessage 或 Command。

) -> ToolMessage | Command:             # 工具执行的监控                  #如果调用的工具是：get_weather(city="深圳")
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")  #输出执行的工具名，比如：[tool monitor]执行工具：get_weather
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")  #输出工具被调用时传入的参数。比如 {"city": "深圳"}

    try:
      #handler(request) 就是在真正执行那个工具，比如调用 get_weather，并得到它的返回结果（可能是字符串 "城市深圳天气为晴天..."）。
      #将返回结果保存到变量 result 中。
        result = handler(request)#这行是核心：调用 handler，并把 request 传给它
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":    #判断当前调用的工具名字是不是 "fill_context_for_report"
            request.runtime.context["report"] = True                #如果是，就执行这条语句

            # request.runtime是一个对象，里面包含运行时的各种状态。
            # .context 是一个字典（可以看作一个记事本，存储了上下文信息），用于在程序不同部分之间传递信息。
            # ["report"] = True 表示在这个记事本里写上一个标记，repor这个键对应的值设为rue。
            # 效果：后续其他代码（比如提示词选择器）可以读取这个标记，知道“用户想要生成报告了”，从而切换行为模式。

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e   #raise 表示重新抛出这个错误。意思是：虽然我记录了日志，但错误依然存在，要让上层调用者知道出错了。如果不写这一行，错误会被“吞掉”，程序可能错误地继续运行。


#在model模型允许前打印日志
#这是一个模型调用前置钩子函数，通过 @before_model 装饰器注册，在 Agent 每次调用大语言模型之前自动执行，用于记录当前对话状态和最后一条消息的详细信息。
@before_model   #模型执行前调用此函数
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None

# AgentState 通常是一个包含对话历史、当前步骤、临时变量等的字典或对象。
# 这里特别提到 state['messages'] 是一个消息列表，记录了用户和助手之间的所有对话。
#Runtime 对象保存了本次执行的上下文信息，比如配置、中间件共享的数据、工具调用状态等。
#state['messages'][-1] 取最后一条消息（即最新的用户输入或助手回复）。
#type(...).__name__ 获取该消息对象的类型名称（比如 HumanMessage、AIMessage、ToolMessage）。



#如果用户有生成报告的意图，就要动态的把原本的提示词切换成生成报告的提示词 --完成提示词切换，更好的生成报告
#这是一个动态提示词切换器，通过 @dynamic_prompt 装饰器注册，Agent 在每次调用模型前会自动执行它，根据运行时上下文中的 report 标记决定使用报告专用提示词还是普通提示词。
@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()

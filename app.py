import time

import streamlit as st
from agent.react_agent import ReactAgent

#有几点需要改善，1是如果新增了一个会话却什么也没问，那么就不要添加到会话列表中。2是现在每问一个问题，消耗的token值都很多，（有时一个问题要花费20000个token）

#需要安装streamlit库

#Streamlit 是一个开源的 Python 库，专门用来快速构建数据应用、机器学习演示、交互式仪表板等 Web 界面，无需编写 HTML/CSS/JavaScript。

# st.title("智扫通机器人智能客服") – 显示页面标题
# st.divider() – 绘制分割线
# st.session_state – 管理会话状态（跨页面/交互保存变量）
# st.chat_message(...) – 创建聊天消息气泡
# st.chat_input() – 创建文本输入框供用户输入
# st.spinner(...) – 显示加载动画
# st.rerun() – 重新运行脚本以更新界面

# ========== 页面配置 ==========
st.set_page_config(
    page_title="智扫通机器人智能客服",
    page_icon="🤖",
    layout="wide"
)
# ========== 侧边栏 - 会话管理 ==========
with st.sidebar:
    st.title("🤖 会话管理")
    st.divider()

    # 初始化智能体
    if "agent" not in st.session_state:
        st.session_state["agent"] = ReactAgent()

    # 初始化会话ID
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = st.session_state["agent"].session_manager.create_new_session()

    # 显示当前会话ID
    st.subheader("当前会话")
    st.code(st.session_state["session_id"][:8] + "...", language="text")

    # 创建新会话按钮（模拟新用户）
    if st.button("🔄 创建新会话", use_container_width=True):
        st.session_state["session_id"] = st.session_state["agent"].session_manager.create_new_session()
        st.session_state["message"] = []  # 清空当前页面的消息显示
        st.rerun()

    # 会话信息
    st.divider()
    st.subheader("测试说明")
    # 会话列表
    st.subheader("会话列表")
    session_list = st.session_state["agent"].session_manager.get_session_list()
    if session_list:
        for session in session_list:
            is_active = session["session_id"] == st.session_state["session_id"]
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{session['title']}**")
                    st.markdown(f"`{session['session_id'][:8]}...`")
                with col2:
                    st.markdown(f"_{session['last_access']}_")
                if is_active:
                    st.button(
                        "✓ 当前会话",
                        disabled=True,
                        use_container_width=True,
                        key=f"btn_{session['session_id']}"
                    )
                else:
                    if st.button(
                            "切换到此会话",
                            use_container_width=True,
                            key=f"btn_{session['session_id']}"
                    ):
                        st.session_state["session_id"] = session["session_id"]
                        history = st.session_state["agent"].session_manager.get_session_history(session["session_id"])
                        st.session_state["message"] = history
                        st.rerun()
                st.divider()
    else:
        st.markdown("暂无会话")
    # 使用提示
    st.subheader("💡 使用提示")
    st.markdown("""
       **测试多用户隔离：**
       1. 发送第一条消息（如：推荐一款扫地机器人）
       2. 点击「创建新会话」模拟新用户
       3. 发送新消息，验证与上一会话隔离
       4. 刷新页面回到原会话，验证历史保留
       - **创建新会话**：点击按钮开始新的对话
       - **切换会话**：点击会话列表中的「切换到此会话」按钮
       - **会话隔离**：每个会话独立保存对话历史
       - **自动清理**：30分钟不活动的会话会被自动清理
        """)


# 标题
st.title("智扫通机器人智能客服")
st.divider()  #分隔符

#检查 Streamlit 的会话状态（st.session_state）中是否已存在键 "agent"。如果不存在，则创建一个 ReactAgent 实例并存入会话状态。
#为什么这么写：Streamlit 在每次用户交互（如点击、输入）后会重新运行整个脚本。
# 若没有 st.session_state，ReactAgent 会被反复创建，导致对话历史、模型状态丢失。通过会话状态持久化智能体实例，确保对话连续性。
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# 解释：类似上面，如果会话状态中没有 "message" 键，则初始化为空列表，用于存储所有聊天记录。
# 为什么这么写：消息列表需要在多次脚本运行间保留，否则每次重绘都会丢失历史对话。st.session_state 充当了“内存数据库”。
if "message" not in st.session_state:
    st.session_state["message"] = []

# 解释：遍历存储的消息列表，对每条消息调用 st.chat_message(role) 创建一个气泡（role 通常是 "user" 或 "assistant"），
#      然后调用 .write() 填入内容。
# 为什么这么写：这一行是渲染历史聊天记录的关键。每次页面重绘（例如用户发送新消息后），都需要重新显示所有历史消息，
#            否则用户只能看到最新的一条。通过循环动态生成，保证了界面与 session_state 同步。
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])


# 解释：在页面底部生成一个聊天输入框，用户输入文本后，prompt 变量会获得输入内容；如果未输入，则为 None。
# 为什么这么写：st.chat_input() 是专为聊天应用设计的输入组件，自动处理回车提交、占位符等，且能在每次脚本运行时返回新输入。
# 用户输入提示词
prompt = st.chat_input()

#
if prompt:
    st.chat_message("user").write(prompt) #立即在聊天界面中显示用户发送的消息气泡，角色为 "user"。 为什么这么写：为了获得即时响应感，先将自己输入的内容显示出来，再调用后端。Streamlit 是同步模型，所以这里先更新界面，然后继续执行。
    st.session_state["message"].append({"role": "user", "content": prompt}) #将用户消息存入会话状态的消息列表中。

    response_messages = [] #创建一个空列表，用于后续流式生成过程中收集完整的回复内容。
    with st.spinner("智能客服思考中..."):   #显示一个加载动画及文字“智能客服思考中...”。在 with 块内的代码执行期间，spinner 保持可见；结束后自动消失。
        res_stream = st.session_state["agent"].execute_stream(prompt, st.session_state["session_id"])  #调用智能体 ReactAgent 的 execute_stream 方法，传入用户提示词，传入session_id 保持会话，返回一个生成器（generator），用于流式产出回复的碎片（chunk）。
        # st.chat_message("User").write_Stream(res_stream) 不能这样写，因为这样写无法保存agent回复的消息
        def capture(generator, cache_list):   #定义一个内部函数 capture，接受一个生成器和一个列表作为参数。该函数会遍历生成器产生的每个 chunk，一边将其存入 cache_list，一边逐字符 yield 出来。

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:   # for char in chunk: time.sleep(0.01) 故意每字符暂停 0.01 秒，再yield打印出来，让显示速度适合人眼阅读，避免瞬间爆出全部文字。
                    time.sleep(0.01)
                    yield char
        #调用了capture(res_stream, response_messages)后得到的也是一个生成器，里面是不断yield一个个字符，
        # 调用用write_stream，就是不断遍历这个生成器，让里面的字符不断输出到气泡中
        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()

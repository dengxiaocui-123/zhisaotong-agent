from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from agent.react_agent import ReactAgent

# pip install fastapi uvicorn   需要安装这两个库

# 创建FastAPI应用实例
app = FastAPI(title="智扫通Agent API服务", description="提供智能体对外接口调用功能")

# 初始化React智能体
agent = ReactAgent()

#BaseModel是Pydantic 库中的核心基类，用于定义数据模型（Data Model），主要用于数据验证和序列化/反序列化。FastAPI 还会根据 BaseModel 自动生成交互式文档
#安装 FastAPI 时会自动安装 Pydantic，因为 FastAPI 强烈依赖 Pydantic 来实现数据验证和类型提示功能。
#BaseModel可实现数据验证、自动类型转换、数据序列化（内置了.dict() 和 .json() 方法）/反序列化、文档自动生成等功能。
# request = QueryRequest(query="你好", user_id="123")
# print(request.dict())  # 输出: {'query': '你好', 'user_id': '123', 'session_id': None}
# print(request.json())  # 输出: {"query": "你好", "user_id": "123", "session_id": null}


# 请求体模型
class QueryRequest(BaseModel):
    query: str                          # 必填字段，字符串类型
    user_id: Optional[str] = None       # 可选字段，字符串类型，默认值为 None
    session_id: Optional[str] = None    # 可选字段，字符串类型，默认值为 None


# 响应体模型
class QueryResponse(BaseModel):
    status: str
    message: str
    data: Optional[str] = None


#FastAPI 路由就是定义 URL 地址 + 对应处理函数，告诉程序：用户访问某个路径时，执行什么代码。
# 由装饰器（@app.get）和具体的函数构成 下面的代码就是一个路由
@app.get("/", tags=["健康检查"])
async def health_check():
    """健康检查接口"""
    return {"status": "success", "message": "智扫通Agent服务运行正常"}


@app.post("/api/query", tags=["智能查询"])
async def query(request: QueryRequest):
    """
    智能查询接口 - 流式响应

    Args:
        query: 用户查询内容
        user_id: 用户ID（可选）
        session_id: 会话ID（可选）

    Returns:
        流式响应的文本内容
    """
    try:
        # 使用ReactAgent执行查询并返回流式响应
        return StreamingResponse(
            agent.execute_stream(request.query),
            media_type="text/plain",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务内部错误: {str(e)}")


@app.post("/api/query_sync", tags=["智能查询"], response_model=QueryResponse)
async def query_sync(request: QueryRequest):
    """
    智能查询接口 - 同步响应（非流式）

    Args:
        query: 用户查询内容
        user_id: 用户ID（可选）
        session_id: 会话ID（可选）

    Returns:
        完整的响应结果
    """
    try:
        # 收集所有流式输出并返回
        result = ""
        for chunk in agent.execute_stream(request.query):
            result += chunk

        return QueryResponse(status="success", message="查询成功", data=result.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务内部错误: {str(e)}")


@app.get("/api/tools", tags=["工具列表"])
async def get_tools():
    """获取当前智能体支持的工具列表"""
    tools = [
        {"name": "rag_summarize", "description": "RAG总结工具"},
        {"name": "get_weather", "description": "获取天气信息"},
        {"name": "get_user_location", "description": "获取用户位置"},
        {"name": "get_user_id", "description": "获取用户ID"},
        {"name": "get_current_month", "description": "获取当前月份"},
        {"name": "fetch_external_data", "description": "获取外部数据"},
        {"name": "fill_context_for_report", "description": "填充报告上下文"}
    ]
    return {"status": "success", "data": tools}

# 启动该fastapi项目的方式有两种：
#    1.直接运行该文件（点击运行按钮）
#    2.在命令行中运行： uvicorn fastapi_serve:app --reload
#    (加上--reload（热部署）参数后，改变代码后，页面刷新后会自动更新；否则只能停止项目，再重新运行才能看到改变)

# 通过输入 http://127.0.0.1:8000/docs  可以访问FastApI交互式文档，查看接口详情


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

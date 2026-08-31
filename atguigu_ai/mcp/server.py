# -*- coding: utf-8 -*-
"""
MCP Server 基类

基于 FastAPI 实现 MCP 协议服务端，提供工具注册接口和 JSON-RPC 路由分发。
子类或调用方通过 register_tool() 注册工具，run() 启动服务。

协议端点：
    POST {endpoint_path}   主入口，处理 JSON-RPC 2.0 请求
    GET  /health           健康检查

支持的方法：
    initialize     能力协商，返回 protocolVersion / capabilities / serverInfo
    tools/list     返回已注册的工具列表
    tools/call     调用工具执行，返回 {"content": [...], "isError": bool}
    ping           心跳，返回 {"pong": true}

错误处理：
    - JSON 解析失败 → -32700 ParseError + HTTP 400
    - 非法请求结构 → -32600 InvalidRequest + HTTP 400
    - 方法不存在   → -32601 MethodNotFound + HTTP 200（JSON-RPC 错误响应）
    - 工具不存在   → -32003 ToolNotFound + HTTP 200
    - 工具执行异常 → -32603 InternalError + HTTP 500

handler 签名：
    async def handler(arguments: Dict[str, Any]) -> Any
    # 也支持 sync 函数（自动包装到线程池执行）

返回值标准化（_normalize_result）：
    - 已是 {"content": [...]} 格式：补全 isError=False 后直接返回
    - 普通 dict：包装为 {"content": [{"type": "result", "data": dict}], "isError": False}
    - 其他类型：同上包装
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from atguigu_ai.mcp.exceptions import (
    MCPError,
    MCPMethodNotFound,
    MCPToolNotFoundError,
)
from atguigu_ai.mcp.protocol import (
    MCP_PROTOCOL_VERSION,
    JsonRpcError,
    JsonRpcErrorResponse,
    JsonRpcRequest,
    JsonRpcResponse,
    parse_message,
)

logger = logging.getLogger(__name__)

# 工具 handler 类型：接受 arguments dict，返回任意可 JSON 序列化的值
ToolHandler = Callable[[Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass
class Tool:
    """工具定义。

    Attributes:
        name:         工具名（含命名空间，如 "ecommerce__query_order"）
        description:  工具描述（用于 tools/list 返回，帮客户端理解工具用途）
        input_schema: JSON Schema 描述参数（用于 tools/list 返回）
        handler:      处理函数，接受 arguments dict，返回结果
    """
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[ToolHandler] = None


class MCPServer:
    """MCP Server 基类。

    生命周期：
        server = MCPServer(name="ecommerce-mcp", version="1.0.0")
        server.register_tool(tool1)
        server.register_tool(tool2)
        server.run(host="127.0.0.1", port=8765)

    也可作为 ASGI app 嵌入到其他 FastAPI 应用：
        app = FastAPI()
        mcp = MCPServer(name="embedded")
        mcp.register_tool(...)
        app.mount("/mcp", mcp.app)
    """

    def __init__(
        self,
        name: str = "mcp-server",
        version: str = "1.0.0",
        endpoint_path: str = "/mcp",
    ) -> None:
        self.name = name
        self.version = version
        self.endpoint_path = endpoint_path
        self._tools: Dict[str, Tool] = {}

        # FastAPI 应用（可被外部 mount）
        self.app = FastAPI(
            title=f"MCP Server: {name}",
            version=version,
        )
        self._setup_routes()

    # ---------- 路由设置 ----------

    def _setup_routes(self) -> None:
        """注册 FastAPI 路由。"""
        self.app.add_api_route(
            self.endpoint_path,
            self._handle_mcp,
            methods=["POST"],
            name="mcp_endpoint",
        )
        self.app.add_api_route(
            "/health",
            self._handle_health,
            methods=["GET"],
            name="health",
        )

    # ---------- 工具注册接口 ----------

    def register_tool(self, tool: Tool) -> None:
        """注册工具。重复注册同名工具会覆盖。"""
        if tool.handler is None:
            raise ValueError(f"工具 {tool.name} 缺少 handler")
        self._tools[tool.name] = tool
        logger.info(
            f"注册工具: {tool.name}（{tool.description}，"
            f"参数: {list(tool.input_schema.get('properties', {}).keys())}）"
        )

    def register_tool_simple(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        """简化的工具注册方法。"""
        self.register_tool(Tool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        ))

    @property
    def tools(self) -> Dict[str, Tool]:
        """已注册的工具字典（只读视图）。"""
        return dict(self._tools)

    # ---------- FastAPI 路由处理 ----------

    async def _handle_health(self, request: Request) -> JSONResponse:
        """健康检查端点。"""
        return JSONResponse({
            "status": "ok",
            "server": self.name,
            "version": self.version,
            "tools_count": len(self._tools),
            "tools": list(self._tools.keys()),
        })

    async def _handle_mcp(self, request: Request) -> JSONResponse:
        """MCP 主入口：处理 JSON-RPC 2.0 请求。"""
        # 1. 解析请求体
        try:
            body = await request.json()
        except Exception as e:
            err = JsonRpcErrorResponse(
                id=None,
                error=JsonRpcError(
                    code=-32700,
                    message=f"请求体 JSON 解析失败: {e}",
                ),
            )
            return JSONResponse(err.to_dict(), status_code=400)

        # 2. 解析 JSON-RPC 消息结构
        try:
            msg = parse_message(body)
        except MCPError as e:
            req_id = body.get("id") if isinstance(body, dict) else None
            err = JsonRpcErrorResponse(
                id=req_id,
                error=JsonRpcError(
                    code=e.code if e.code is not None else -32700,
                    message=e.message,
                    data=e.data,
                ),
            )
            return JSONResponse(err.to_dict(), status_code=400)

        # 3. 只处理 Request（Notification 本期未实现）
        if not isinstance(msg, JsonRpcRequest):
            err = JsonRpcErrorResponse(
                id=getattr(msg, "id", None),
                error=JsonRpcError(
                    code=-32600,
                    message=f"只支持 Request，收到 {type(msg).__name__}",
                ),
            )
            return JSONResponse(err.to_dict(), status_code=400)

        # 4. 分发到对应方法
        try:
            result = await self._dispatch(msg)
            response = JsonRpcResponse(id=msg.id, result=result)
            return JSONResponse(response.to_dict())
        except MCPError as e:
            # 业务层抛出的 MCP 异常，按对应错误码返回（HTTP 200，错误在 body 里）
            err = JsonRpcErrorResponse(
                id=msg.id,
                error=JsonRpcError(
                    code=e.code if e.code is not None else -32603,
                    message=e.message,
                    data=e.data,
                ),
            )
            return JSONResponse(err.to_dict())
        except Exception as e:
            # 未预期异常
            logger.exception(f"工具执行未预期异常: {e}")
            err = JsonRpcErrorResponse(
                id=msg.id,
                error=JsonRpcError(
                    code=-32603,
                    message=f"服务端内部错误: {e}",
                    data={"traceback": traceback.format_exc()},
                ),
            )
            return JSONResponse(err.to_dict(), status_code=500)

    async def _dispatch(self, request: JsonRpcRequest) -> Any:
        """根据 method 路由到对应处理函数。"""
        method = request.method
        params = request.params or {}

        if method == "initialize":
            return self._handle_initialize(params)
        if method == "tools/list":
            return self._handle_list_tools(params)
        if method == "tools/call":
            return await self._handle_call_tool(params)
        if method == "ping":
            return {"pong": True}

        # 未知方法
        raise MCPMethodNotFound(f"方法不存在: {method}")

    # ---------- MCP 标准方法实现 ----------

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 initialize 请求：返回服务端能力信息。"""
        client_info = params.get("clientInfo", {})
        client_protocol = params.get("protocolVersion", "unknown")
        logger.info(
            f"客户端 initialize: {client_info}（protocol={client_protocol}）"
        )
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": self.name, "version": self.version},
        }

    def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 tools/list 请求：返回已注册的工具列表。"""
        tools_list = [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self._tools.values()
        ]
        return {"tools": tools_list}

    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理 tools/call 请求：调用工具执行。"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {}) or {}

        if not isinstance(tool_name, str):
            raise MCPToolNotFoundError(tool_name=str(tool_name))

        tool = self._tools.get(tool_name)
        if tool is None:
            raise MCPToolNotFoundError(tool_name=tool_name)

        if tool.handler is None:
            raise MCPToolNotFoundError(tool_name=tool_name)

        # 调用 handler（支持 sync 和 async 两种）
        handler = tool.handler
        if asyncio.iscoroutinefunction(handler):
            result = await handler(arguments)
        else:
            # sync 函数包装到线程池避免阻塞事件循环
            result = await asyncio.to_thread(handler, arguments)

        # 标准化结果格式
        return self._normalize_result(result)

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        """将工具 handler 返回值标准化为 MCP 响应格式。

        标准格式：{"content": [{"type": "...", "data": ...}, ...], "isError": bool}

        支持的输入：
        - dict 已含 content 字段：补全 isError=False 后直接返回
        - 普通 dict：包装为 {"content": [{"type": "result", "data": dict}], "isError": False}
        - list：包装为 {"content": [{"type": "result", "data": item} for item in list], "isError": False}
        - 其他类型：包装为 {"content": [{"type": "result", "data": result}], "isError": False}
        """
        if isinstance(result, dict) and "content" in result:
            # 已是标准格式，补全 isError
            if "isError" not in result:
                result["isError"] = False
            return result
        if isinstance(result, dict):
            return {
                "content": [{"type": "result", "data": result}],
                "isError": False,
            }
        if isinstance(result, list):
            return {
                "content": [
                    {"type": "result", "data": item} for item in result
                ],
                "isError": False,
            }
        return {
            "content": [{"type": "result", "data": result}],
            "isError": False,
        }

    # ---------- 启动 ----------

    def run(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        """启动服务（同步阻塞）。

        Args:
            host: 监听地址
            port: 监听端口
        """
        import uvicorn

        logger.info(
            f"MCP Server '{self.name}' 启动: "
            f"http://{host}:{port}{self.endpoint_path} "
            f"（已注册 {len(self._tools)} 个工具）"
        )
        uvicorn.run(self.app, host=host, port=port)


__all__ = [
    "Tool",
    "ToolHandler",
    "MCPServer",
]

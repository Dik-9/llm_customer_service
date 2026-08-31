# -*- coding: utf-8 -*-
"""
atguigu_ai.mcp - MCP 工具协议层

自研 MCP（Model Context Protocol）实现，将业务 Action 从同进程函数调用
提升为跨进程、跨语言的标准协议接口。

核心组件：
- exceptions:     MCP 异常体系（遵循 JSON-RPC 2.0 错误码）
- protocol:       JSON-RPC 2.0 消息封装（Request/Response/Error/Notification）
- client:         MCPClient（initialize / tools/list / tools/call + 超时重试熔断）
- server:         MCPServer 基类（FastAPI 路由 + 工具注册接口）
- tool_registry:  ToolRegistry（命名空间路由 + MCP 失败降级到本地 Action）

设计原则：
1. 协议语义借用 JSON-RPC 2.0 + tools/list|call，不追求与官方规范 100% 兼容
2. 传输层用 HTTP（FastAPI 服务端 + httpx 客户端），支撑跨语言演进
3. 多路径保障：MCP 不可达时自动降级回本地 Action 直调，保证主链路不中断
"""

from atguigu_ai.mcp.exceptions import (
    MCPError,
    MCPConnectionError,
    MCPTimeoutError,
    MCPProtocolError,
    MCPParseError,
    MCPInvalidRequest,
    MCPMethodNotFound,
    MCPInvalidParams,
    MCPInternalError,
    MCPToolNotFoundError,
    FallbackTriggered,
    error_from_code,
)
from atguigu_ai.mcp.protocol import (
    JSONRPC_VERSION,
    MCP_PROTOCOL_VERSION,
    RpcId,
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    JsonRpcErrorResponse,
    JsonRpcNotification,
    is_request,
    is_notification,
    is_response,
    is_error_response,
    parse_message,
)
from atguigu_ai.mcp.client import (
    CircuitBreakerConfig,
    MCPClientConfig,
    MCPClient,
)
from atguigu_ai.mcp.server import (
    Tool,
    ToolHandler,
    MCPServer,
)

__all__ = [
    # 异常
    "MCPError",
    "MCPConnectionError",
    "MCPTimeoutError",
    "MCPProtocolError",
    "MCPParseError",
    "MCPInvalidRequest",
    "MCPMethodNotFound",
    "MCPInvalidParams",
    "MCPInternalError",
    "MCPToolNotFoundError",
    "FallbackTriggered",
    "error_from_code",
    # 协议常量
    "JSONRPC_VERSION",
    "MCP_PROTOCOL_VERSION",
    "RpcId",
    # 消息类型
    "JsonRpcRequest",
    "JsonRpcResponse",
    "JsonRpcError",
    "JsonRpcErrorResponse",
    "JsonRpcNotification",
    # 判定函数
    "is_request",
    "is_notification",
    "is_response",
    "is_error_response",
    "parse_message",
    # 客户端
    "CircuitBreakerConfig",
    "MCPClientConfig",
    "MCPClient",
    # 服务端
    "Tool",
    "ToolHandler",
    "MCPServer",
]

# -*- coding: utf-8 -*-
"""
JSON-RPC 2.0 消息封装

定义自研 MCP 协议使用的消息类型，遵循 JSON-RPC 2.0 规范。
所有消息通过 HTTP body 传输（FastAPI 服务端 + httpx 客户端）。

消息类型：
- JsonRpcRequest          请求：带 id，期望收到响应
- JsonRpcResponse        成功响应：包含 result
- JsonRpcErrorResponse   错误响应：包含 error 对象
- JsonRpcNotification     通知：无 id，不需响应（预留）

设计说明：
- jsonrpc 字段固定为 "2.0"（JSON-RPC 2.0 规范要求）
- 协议自身的版本号（protocolVersion）放在 initialize 的 params 里，
  这是 MCP 官方规范的做法，不破坏 JSON-RPC 兼容性
- 所有消息提供 to_dict / from_dict 双向转换
- parse_message 工厂函数根据 dict 结构自动识别消息类型

参考：
- JSON-RPC 2.0 规范: https://www.jsonrpc.org/specification
- MCP 协议规范:      https://modelcontextprotocol.io/specification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

# JSON-RPC 2.0 规范要求 jsonrpc 字段固定为该字符串
JSONRPC_VERSION: str = "2.0"

# MCP 自研协议版本（用于 initialize 能力协商，放在 params.protocolVersion）
MCP_PROTOCOL_VERSION: str = "atguigu-mcp/1.0"

# JSON-RPC id 类型：规范允许 integer / string / null
RpcId = Union[int, str, None]


# ========== 消息类型 ==========

@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 请求消息。

    客户端发起 method 调用，期望收到与 id 对应的响应。

    Attributes:
        method: 调用的方法名（如 "tools/list", "tools/call", "initialize"）
        params: 方法参数（dict 或 list，可选）
        id: 请求 ID，用于匹配响应（int 或 str，规范允许 null 但不推荐）
        jsonrpc: 协议版本标识，固定为 "2.0"
    """

    method: str
    params: Optional[Any] = None
    id: RpcId = None
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.params is not None:
            d["params"] = self.params
        if self.id is not None:
            d["id"] = self.id
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JsonRpcRequest":
        return cls(
            method=data["method"],
            params=data.get("params"),
            id=data.get("id"),
            jsonrpc=data.get("jsonrpc", JSONRPC_VERSION),
        )


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 成功响应。

    Attributes:
        id: 对应请求的 ID
        result: 调用结果（任意 JSON 可序列化值）
        jsonrpc: 协议版本标识
    """

    id: RpcId
    result: Any
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JsonRpcResponse":
        return cls(
            id=data.get("id"),
            result=data["result"],
            jsonrpc=data.get("jsonrpc", JSONRPC_VERSION),
        )


@dataclass
class JsonRpcError:
    """JSON-RPC 2.0 error 对象。

    错误响应的 error 字段结构，包含 code/message/可选 data。

    Attributes:
        code: 错误码（参考 exceptions.py 的错误码定义）
        message: 错误描述
        data: 附加数据（可选，对应异常的 data 字段）
    """

    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JsonRpcError":
        return cls(
            code=data["code"],
            message=data["message"],
            data=data.get("data"),
        )


@dataclass
class JsonRpcErrorResponse:
    """JSON-RPC 2.0 错误响应。

    Attributes:
        id: 对应请求的 ID（无法解析请求时为 None）
        error: 错误对象
        jsonrpc: 协议版本标识
    """

    id: RpcId
    error: JsonRpcError
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "error": self.error.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JsonRpcErrorResponse":
        return cls(
            id=data.get("id"),
            error=JsonRpcError.from_dict(data["error"]),
            jsonrpc=data.get("jsonrpc", JSONRPC_VERSION),
        )

    def to_exception(self):
        """将错误响应转换为对应的 MCP 异常实例。

        利用 exceptions.error_from_code 完成错误码到异常类的映射。
        """
        from atguigu_ai.mcp.exceptions import error_from_code

        return error_from_code(
            code=self.error.code,
            message=self.error.message,
            data=self.error.data,
        )


@dataclass
class JsonRpcNotification:
    """JSON-RPC 2.0 通知（无 id，不需响应）。

    用于服务端向客户端推送事件（如长任务进度）。
    本期未使用，预留以备后续扩展。

    Attributes:
        method: 通知的方法名
        params: 通知参数（可选）
        jsonrpc: 协议版本标识
    """

    method: str
    params: Optional[Any] = None
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.params is not None:
            d["params"] = self.params
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JsonRpcNotification":
        return cls(
            method=data["method"],
            params=data.get("params"),
            jsonrpc=data.get("jsonrpc", JSONRPC_VERSION),
        )


# ========== 类型判定辅助函数 ==========

def is_request(data: Any) -> bool:
    """判断 dict 是否为合法 JSON-RPC 请求（带 id）。"""
    return (
        isinstance(data, dict)
        and data.get("jsonrpc") is not None
        and isinstance(data.get("method"), str)
        and "id" in data
    )


def is_notification(data: Any) -> bool:
    """判断 dict 是否为 JSON-RPC 通知（无 id）。"""
    return (
        isinstance(data, dict)
        and data.get("jsonrpc") is not None
        and isinstance(data.get("method"), str)
        and "id" not in data
    )


def is_response(data: Any) -> bool:
    """判断 dict 是否为 JSON-RPC 成功响应。

    JSON-RPC 2.0 规范要求 result 和 error 互斥，同时存在视为非法。
    """
    return (
        isinstance(data, dict)
        and data.get("jsonrpc") is not None
        and "id" in data
        and "result" in data
        and "error" not in data  # 规范要求：result 和 error 互斥
    )


def is_error_response(data: Any) -> bool:
    """判断 dict 是否为 JSON-RPC 错误响应。

    JSON-RPC 2.0 规范要求 result 和 error 互斥，同时存在视为非法。
    """
    return (
        isinstance(data, dict)
        and data.get("jsonrpc") is not None
        and "id" in data
        and isinstance(data.get("error"), dict)
        and "result" not in data  # 规范要求：result 和 error 互斥
    )


def parse_message(data: Any) -> Union[
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcErrorResponse,
    JsonRpcNotification,
]:
    """根据 dict 结构解析为对应的消息类型。

    Args:
        data: 解析自 JSON 的 dict

    Returns:
        对应类型的消息实例

    Raises:
        MCPParseError: 消息结构不符合 JSON-RPC 2.0 规范
    """
    from atguigu_ai.mcp.exceptions import MCPParseError

    if not isinstance(data, dict):
        raise MCPParseError(f"消息必须是 dict，实际类型: {type(data).__name__}")
    if data.get("jsonrpc") is None:
        raise MCPParseError("消息缺少 jsonrpc 字段")

    if is_request(data):
        return JsonRpcRequest.from_dict(data)
    if is_response(data):
        return JsonRpcResponse.from_dict(data)
    if is_error_response(data):
        return JsonRpcErrorResponse.from_dict(data)
    if is_notification(data):
        return JsonRpcNotification.from_dict(data)

    raise MCPParseError(f"无法识别的消息结构: {data}")


__all__ = [
    # 常量
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
]

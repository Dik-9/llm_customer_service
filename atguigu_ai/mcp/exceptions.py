# -*- coding: utf-8 -*-
"""
MCP 异常体系

定义自研 MCP 协议层使用的所有异常类型。异常分类遵循 JSON-RPC 2.0
规范的错误码区间，便于将服务端返回的错误响应映射到对应的异常类型。

异常层级：
    MCPError                        ← 所有 MCP 异常的基类
    ├── MCPConnectionError          ← 网络层：连接失败、服务不可达
    ├── MCPTimeoutError             ← 网络层：调用超时
    ├── MCPProtocolError            ← 协议层：JSON-RPC 错误响应
    │   ├── MCPParseError           ← -32700 响应解析失败
    │   ├── MCPInvalidRequest       ← -32600 非法请求
    │   ├── MCPMethodNotFound       ← -32601 方法不存在
    │   ├── MCPInvalidParams        ← -32602 参数非法
    │   └── MCPInternalError       ← -32603 服务端内部错误
    ├── MCPToolNotFoundError        ← 业务层：请求的工具不存在
    └── FallbackTriggered           ← 降级信号：MCP 不可用，应降级到本地 Action

设计说明：
- FallbackTriggered 故意不继承 MCPError，因为它不代表 MCP 层出错，
  而是表示"已决定走降级路径"——属于上层控制流。
- 服务端错误码区间划分参考 JSON-RPC 2.0 规范：
    -32700 ~ -32600  协议层（解析/请求/方法/参数/内部）
    -32000 ~ -32099  应用层（连接/超时/工具未找到等自定义错误）
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type


class MCPError(Exception):
    """MCP 异常基类。

    所有自定义 MCP 异常都继承自此类，便于上层用
    `except MCPError` 统一捕获协议层所有错误。

    Attributes:
        message: 错误描述
        code: JSON-RPC 2.0 错误码（可选）
        data: 附加数据（可选，对应 JSON-RPC error.data）
    """

    def __init__(
        self,
        message: str = "",
        code: Optional[int] = None,
        data: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data

    def __str__(self) -> str:
        if self.code is not None:
            return f"[MCP {self.code}] {self.message}"
        return f"[MCP] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON-RPC 2.0 error 对象格式。"""
        err: Dict[str, Any] = {
            "code": self.code if self.code is not None else -32603,
            "message": self.message,
        }
        if self.data is not None:
            err["data"] = self.data
        return err


# ========== 网络层异常（-32xxx 区间）==========

class MCPConnectionError(MCPError):
    """连接 MCP Server 失败。

    触发场景：DNS 解析失败、TCP 连接被拒绝、服务未启动。
    出现此异常时，ToolRegistry 应触发降级到本地 Action。
    """

    def __init__(self, message: str = "MCP Server 连接失败", **kwargs: Any) -> None:
        super().__init__(message, code=-32001, **kwargs)


class MCPTimeoutError(MCPError):
    """MCP 调用超时。

    触发场景：客户端在配置的 timeout 内未收到响应。
    重试次数耗尽后转为该异常，ToolRegistry 应触发降级。
    """

    def __init__(self, message: str = "MCP 调用超时", **kwargs: Any) -> None:
        super().__init__(message, code=-32002, **kwargs)


# ========== 协议层异常（JSON-RPC 2.0 标准错误码）==========

class MCPProtocolError(MCPError):
    """JSON-RPC 协议错误基类。

    当服务端返回符合 JSON-RPC 2.0 规范的 error 响应时，
    根据错误码映射到对应的子类异常。
    """

    DEFAULT_CODE: int = -32603  # Internal error

    def __init__(
        self,
        message: str = "",
        code: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            code=code if code is not None else self.DEFAULT_CODE,
            **kwargs,
        )


class MCPParseError(MCPProtocolError):
    """-32700 响应解析失败（JSON 不合法或结构不符合协议）。"""

    DEFAULT_CODE = -32700

    def __init__(self, message: str = "响应解析失败", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class MCPInvalidRequest(MCPProtocolError):
    """-32600 非法请求（请求体不符合 JSON-RPC 2.0 规范）。"""

    DEFAULT_CODE = -32600

    def __init__(self, message: str = "非法请求", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class MCPMethodNotFound(MCPProtocolError):
    """-32601 方法不存在（调用了未实现的 RPC method）。"""

    DEFAULT_CODE = -32601

    def __init__(self, message: str = "方法不存在", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class MCPInvalidParams(MCPProtocolError):
    """-32602 参数非法（调用工具时参数不符合 inputSchema）。"""

    DEFAULT_CODE = -32602

    def __init__(self, message: str = "参数非法", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class MCPInternalError(MCPProtocolError):
    """-32603 服务端内部错误（工具执行异常等）。"""

    DEFAULT_CODE = -32603

    def __init__(self, message: str = "服务端内部错误", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


# ========== 业务层异常 ==========

class MCPToolNotFoundError(MCPError):
    """请求的工具在 MCP Server 的 tools/list 中不存在。

    触发场景：ToolRegistry 拿到 action_name 后查询映射表，
    发现对应的 MCP 工具不存在。应触发降级到本地 Action。

    Attributes:
        tool_name: 缺失的工具名
    """

    def __init__(self, tool_name: str = "", **kwargs: Any) -> None:
        msg = f"工具不存在: {tool_name}" if tool_name else "工具不存在"
        super().__init__(msg, code=-32003, **kwargs)
        self.tool_name = tool_name


# ========== 降级信号（控制流，非错误）==========

class FallbackTriggered(Exception):
    """降级信号：MCP 不可用，应降级到本地 Action 执行。

    当 ToolRegistry 捕获到 MCP 异常（连接失败/超时/工具不存在）时，
    抛出此信号通知调用方"应降级到本地 Action 执行"。

    设计为非 MCPError 子类，因为它不代表 MCP 层出错，
    而是表示"已决定走降级路径"——属于上层控制流。

    Attributes:
        reason: 降级原因描述
        action_name: 触发降级的 action 名
        cause: 原始异常（可选）
    """

    def __init__(
        self,
        reason: str = "",
        action_name: str = "",
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.action_name = action_name
        self.cause = cause

    def __str__(self) -> str:
        return f"降级到本地 Action: {self.action_name}（原因: {self.reason}）"


# ========== 错误码 → 异常类 映射表 ==========

# 用于从服务端 error 响应恢复异常类型
_ERROR_CODE_MAP: Dict[int, Type[MCPError]] = {
    -32700: MCPParseError,
    -32600: MCPInvalidRequest,
    -32601: MCPMethodNotFound,
    -32602: MCPInvalidParams,
    -32603: MCPInternalError,
    -32001: MCPConnectionError,
    -32002: MCPTimeoutError,
    -32003: MCPToolNotFoundError,
}


def error_from_code(
    code: int,
    message: str = "",
    data: Any = None,
) -> MCPError:
    """根据 JSON-RPC 错误码构造对应的异常实例。

    Args:
        code: JSON-RPC 2.0 错误码
        message: 错误描述（对 MCPToolNotFoundError 会作为 tool_name）
        data: 附加数据

    Returns:
        对应的 MCPError 子类实例；未知错误码返回 MCPInternalError
    """
    cls = _ERROR_CODE_MAP.get(code, MCPInternalError)
    # MCPToolNotFoundError 的第一参数是 tool_name 而非 message，需特殊处理
    if cls is MCPToolNotFoundError:
        return cls(tool_name=message, data=data)
    return cls(message=message, data=data)


__all__ = [
    # 基类
    "MCPError",
    # 网络层
    "MCPConnectionError",
    "MCPTimeoutError",
    # 协议层
    "MCPProtocolError",
    "MCPParseError",
    "MCPInvalidRequest",
    "MCPMethodNotFound",
    "MCPInvalidParams",
    "MCPInternalError",
    # 业务层
    "MCPToolNotFoundError",
    # 降级信号
    "FallbackTriggered",
    # 工具函数
    "error_from_code",
]

# -*- coding: utf-8 -*-
"""
MCP 客户端

实现自研 MCP 协议的客户端，封装三个核心 RPC 方法：
- initialize():    能力协商（交换 protocolVersion、capabilities、serverInfo）
- list_tools():    拉取工具列表（可缓存）
- call_tool():     调用工具执行

传输层使用 httpx.AsyncClient（HTTP + JSON-RPC 2.0 body），
内置三件套保护：
1. 超时：每个 RPC 调用按 config.timeout（默认 10s）
2. 重试：连接失败/超时按 config.retry 次重试（默认 2 次）
3. 熔断：连续失败达 failure_threshold 进入 OPEN，reset_timeout
        秒后转 HALF_OPEN 允许一次试探调用

异常分类处理：
- 网络层错误（MCPTimeoutError / MCPConnectionError）→ 触发重试，重试耗尽
  后由熔断器记录失败计数
- 协议层错误响应（MCPInvalidParams / MCPMethodNotFound 等）→ 不重试，
  直接抛出（业务错误重试无意义）
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from atguigu_ai.mcp.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPParseError,
    MCPTimeoutError,
)
from atguigu_ai.mcp.protocol import (
    MCP_PROTOCOL_VERSION,
    JsonRpcErrorResponse,
    JsonRpcRequest,
    JsonRpcResponse,
    parse_message,
)

logger = logging.getLogger(__name__)


# ========== 配置数据结构 ==========

@dataclass
class CircuitBreakerConfig:
    """熔断器配置。

    Attributes:
        failure_threshold: 连续失败 N 次后进入 OPEN 状态
        reset_timeout:     OPEN 状态维持多少秒后转 HALF_OPEN（半开探测）
    """
    failure_threshold: int = 5
    reset_timeout: float = 30.0


@dataclass
class MCPClientConfig:
    """MCPClient 配置。

    Attributes:
        base_url:        MCP Server 的完整请求 URL（如 http://127.0.0.1:8765/mcp）
        timeout:         单次 HTTP 调用超时（秒）
        retry:           网络层错误的重试次数（不含首次）
        circuit_breaker: 熔断器配置
        client_info:     客户端身份信息（用于 initialize 协商）
        transport:       httpx transport（测试用 MockTransport 注入；生产为 None）
    """
    base_url: str
    timeout: float = 10.0
    retry: int = 2
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    client_info: Dict[str, str] = field(
        default_factory=lambda: {"name": "atguigu-agent", "version": "1.0.0"}
    )
    transport: Optional[httpx.BaseTransport] = None


# ========== 熔断器状态机 ==========

class _CircuitBreaker:
    """熔断器状态机。

    三态转换：
        CLOSED  ──失败≥threshold──▶  OPEN
        OPEN    ──reset_timeout后──▶  HALF_OPEN
        HALF_OPEN ──成功──▶  CLOSED
        HALF_OPEN ──失败──▶  OPEN

    设计参考：Michael Nygard《Release It!》中的 Circuit Breaker 模式。
    """

    CLOSED = "closed"        # 正常调用
    OPEN = "open"           # 熔断，所有调用直接拒绝
    HALF_OPEN = "half_open" # 半开，允许一次试探

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.config = config
        self.state: str = self.CLOSED
        self.failure_count: int = 0
        self.opened_at: Optional[float] = None  # OPEN 状态开始时间

    def allow(self) -> bool:
        """是否允许调用。OPEN 状态下检查是否到 reset_timeout 转 HALF_OPEN。"""
        if self.state == self.OPEN:
            if self.opened_at is not None and (
                time.time() - self.opened_at
            ) >= self.config.reset_timeout:
                logger.info(
                    f"熔断器进入 HALF_OPEN（半开探测），"
                    f"已熔断 {time.time() - self.opened_at:.1f}s"
                )
                self.state = self.HALF_OPEN
                return True
            return False
        return True  # CLOSED 或 HALF_OPEN 都允许

    def record_success(self) -> None:
        """记录调用成功。HALF_OPEN 状态下成功 → CLOSED。"""
        if self.state == self.HALF_OPEN:
            logger.info("熔断器从 HALF_OPEN 恢复到 CLOSED")
        self.state = self.CLOSED
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self) -> None:
        """记录调用失败。HALF_OPEN 失败立即回 OPEN；CLOSED 累计失败次数。"""
        self.failure_count += 1
        if self.state == self.HALF_OPEN:
            self._trip_open()
            return
        if self.failure_count >= self.config.failure_threshold:
            self._trip_open()

    def _trip_open(self) -> None:
        """进入 OPEN 状态。"""
        self.state = self.OPEN
        self.opened_at = time.time()
        logger.warning(
            f"熔断器进入 OPEN 状态，连续失败 {self.failure_count} 次，"
            f"将熔断 {self.config.reset_timeout}s"
        )


# ========== MCPClient 实现 ==========

class MCPClient:
    """MCP 协议客户端。

    生命周期：
        client = MCPClient(config)
        await client.initialize()        # 能力协商
        tools = await client.list_tools()
        result = await client.call_tool("ecommerce__query_order", {"user_id": "1001"})
        await client.close()

    或使用 async context manager：
        async with MCPClient(config) as client:
            await client.initialize()
            result = await client.call_tool(...)

    线程安全：单实例并发调用未保证，建议每个 Agent 持有一个 client。
    """

    def __init__(self, config: MCPClientConfig) -> None:
        self.config = config
        self._breaker = _CircuitBreaker(config.circuit_breaker)
        self._id_counter = itertools.count(1)  # 每实例独立 id 序列
        self._initialized = False
        self._server_info: Optional[Dict[str, Any]] = None
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._http = httpx.AsyncClient(
            timeout=config.timeout,
            transport=config.transport,
        )

    # ---------- 内部：发送 JSON-RPC 请求 ----------

    def _next_id(self) -> int:
        return next(self._id_counter)

    async def _send_request(self, method: str, params: Any = None) -> Any:
        """发送 JSON-RPC 请求并返回 result（成功时）。

        处理流程：
        1. 熔断器检查（OPEN 直接拒绝）
        2. 构造 JsonRpcRequest
        3. 重试循环（最多 retry + 1 次）
        4. 解析响应：成功响应返回 result；错误响应抛对应异常
        5. 网络层错误重试；协议层错误（业务错误响应）不重试

        Raises:
            MCPConnectionError: 熔断中或连接失败重试耗尽
            MCPTimeoutError: 超时重试耗尽
            MCPProtocolError 及子类: 服务端返回错误响应
            MCPParseError: 响应无法解析
        """
        # 1. 熔断器检查
        if not self._breaker.allow():
            raise MCPConnectionError(
                f"MCP Server 熔断中: {self.config.base_url}（"
                f"已失败 {self._breaker.failure_count} 次，"
                f"等待 {self.config.circuit_breaker.reset_timeout}s 后半开探测）"
            )

        req = JsonRpcRequest(
            method=method,
            params=params,
            id=self._next_id(),
        )
        last_exc: Optional[Exception] = None
        total_attempts = self.config.retry + 1

        # 2. 重试循环
        for attempt in range(1, total_attempts + 1):
            try:
                resp = await self._http.post(
                    self.config.base_url,
                    json=req.to_dict(),
                )
                resp.raise_for_status()

                msg = parse_message(resp.json())

                if isinstance(msg, JsonRpcResponse):
                    self._breaker.record_success()
                    return msg.result

                if isinstance(msg, JsonRpcErrorResponse):
                    # 协议层错误响应：业务错误（如 method not found、参数错误）
                    # 重试无意义，直接抛出，但仍记录熔断失败计数
                    self._breaker.record_failure()
                    raise msg.to_exception()

                # 收到 Request/Notification 是协议错误
                self._breaker.record_failure()
                raise MCPParseError(
                    f"期望 Response/ErrorResponse，实际收到: {type(msg).__name__}"
                )

            # 网络层错误：可重试
            except httpx.TimeoutException as e:
                last_exc = MCPTimeoutError(f"HTTP 超时: {e}")
                logger.debug(
                    f"MCP {method} 超时（attempt {attempt}/{total_attempts}）: {e}"
                )
            except httpx.ConnectError as e:
                last_exc = MCPConnectionError(f"HTTP 连接失败: {e}")
                logger.debug(
                    f"MCP {method} 连接失败（attempt {attempt}/{total_attempts}）: {e}"
                )
            except httpx.HTTPStatusError as e:
                # 4xx/5xx：服务端不可用，转 MCPConnectionError 触发重试
                last_exc = MCPConnectionError(
                    f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                )
                logger.debug(
                    f"MCP {method} HTTP 错误（attempt {attempt}/{total_attempts}）: "
                    f"{e.response.status_code}"
                )
            except (MCPTimeoutError, MCPConnectionError) as e:
                # 已被前面 except 转换过的网络层错误，继续重试
                last_exc = e
            except MCPError:
                # 协议层错误（业务错误响应、解析错误）：不重试，直接抛
                raise

        # 3. 重试耗尽，记录熔断并抛
        self._breaker.record_failure()
        if last_exc is not None:
            raise last_exc
        raise MCPConnectionError(
            f"MCP {method} 重试 {self.config.retry} 次后仍失败（未知原因）"
        )

    # ---------- 三个核心 RPC 方法 ----------

    async def initialize(self) -> Dict[str, Any]:
        """能力协商。

        客户端发送自己的 protocolVersion 和 capabilities，服务端
        返回支持的 capabilities 和 serverInfo。

        Returns:
            服务端响应的 InitializeResult

        Raises:
            MCPError: 协议层或网络层错误
        """
        params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "clientInfo": self.config.client_info,
        }
        result = await self._send_request("initialize", params)
        self._initialized = True
        if isinstance(result, dict):
            self._server_info = result.get("serverInfo")
        logger.info(
            f"MCP 客户端 initialize 成功，server={self._server_info}"
        )
        return result if isinstance(result, dict) else {}

    async def list_tools(
        self,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """拉取工具列表。

        Args:
            use_cache: True 时复用首次拉取的缓存（推荐，工具列表很少变化）

        Returns:
            工具定义列表，每个元素含 name / description / inputSchema

        Raises:
            MCPError: 协议层或网络层错误
        """
        if use_cache and self._tools_cache is not None:
            return self._tools_cache

        result = await self._send_request("tools/list", {})
        if isinstance(result, dict) and "tools" in result:
            tools = result["tools"]
            if isinstance(tools, list):
                self._tools_cache = tools
                logger.info(f"MCP 客户端拉取到 {len(tools)} 个工具")
                return tools
        return []

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """调用工具。

        Args:
            name: 工具名（含命名空间，如 "ecommerce__query_order"）
            arguments: 工具参数（对应工具 inputSchema 的 properties）

        Returns:
            工具执行结果，结构遵循 MCP 标准：
            {"content": [{type, data}, ...], "isError": bool}

        Raises:
            MCPError: 协议层或网络层错误（含 MCPToolNotFoundError）
        """
        params = {"name": name, "arguments": arguments or {}}
        result = await self._send_request("tools/call", params)
        return result

    # ---------- 生命周期管理 ----------

    async def close(self) -> None:
        """关闭底层 HTTP 客户端。"""
        await self._http.aclose()
        logger.debug(f"MCP 客户端已关闭: {self.config.base_url}")

    async def __aenter__(self) -> "MCPClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # ---------- 调试辅助 ----------

    @property
    def is_initialized(self) -> bool:
        """是否已完成 initialize 协商。"""
        return self._initialized

    @property
    def server_info(self) -> Optional[Dict[str, Any]]:
        """服务端信息（initialize 协商后可用）。"""
        return self._server_info

    @property
    def breaker_state(self) -> str:
        """当前熔断器状态（closed / open / half_open）。"""
        return self._breaker.state


__all__ = [
    "CircuitBreakerConfig",
    "MCPClientConfig",
    "MCPClient",
]

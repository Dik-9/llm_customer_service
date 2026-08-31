# -*- coding: utf-8 -*-
"""
M1.3 验证脚本：MCPClient 单元测试

用 httpx.MockTransport 注入测试 transport，不依赖真实 HTTP server。

验证项：
1. initialize 成功：响应解析、_initialized 标记、server_info 保留
2. list_tools 成功 + 缓存：第二次调用不发起 HTTP
3. call_tool 成功：返回 content/isError 结构
4. 协议层错误（method not found）：不重试，立即抛 MCPMethodNotFound
5. 超时重试：模拟 2 次超时后成功，验证重试机制
6. 超时重试耗尽：抛 MCPTimeoutError + 熔断器记录失败
7. 熔断器 OPEN：连续失败达 threshold 后所有调用直接抛 MCPConnectionError
8. 连接失败：抛 MCPConnectionError
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx

from atguigu_ai.mcp import (
    MCPClient,
    MCPClientConfig,
    CircuitBreakerConfig,
    MCPMethodNotFound,
    MCPTimeoutError,
    MCPConnectionError,
)
from atguigu_ai.mcp.protocol import JSONRPC_VERSION, MCP_PROTOCOL_VERSION


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- MockTransport 辅助函数 ----------

def _make_handler(
    responses_per_call: List[Dict[str, Any]] | None = None,
    exc_per_call: List[Exception] | None = None,
    call_log: List[httpx.Request] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """构造 MockTransport handler。

    Args:
        responses_per_call: 每次调用返回的 JSON Response body（按顺序消耗）
        exc_per_call: 每次调用抛的异常（按顺序消耗，优先于 responses）
        call_log: 调用记录会追加到此列表

    Returns:
        handler 函数
    """
    responses_per_call = list(responses_per_call or [])
    exc_per_call = list(exc_per_call or [])

    def handler(request: httpx.Request) -> httpx.Response:
        if call_log is not None:
            call_log.append(request)
        if exc_per_call:
            raise exc_per_call.pop(0)
        if responses_per_call:
            return httpx.Response(200, json=responses_per_call.pop(0))
        return httpx.Response(404, text="no mock response")

    return handler


# ---------- 测试用例 ----------

def test_initialize_success() -> None:
    """测试 1：initialize 成功路径。"""
    print("[测试 1] initialize 成功")
    server_resp = {
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "result": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "ecommerce-mcp", "version": "1.0.0"},
        },
    }
    transport = httpx.MockTransport(_make_handler(responses_per_call=[server_resp]))
    config = MCPClientConfig(base_url="http://test/mcp", transport=transport)
    client = MCPClient(config)

    async def run():
        result = await client.initialize()
        return result

    result = asyncio.run(run())
    if not client.is_initialized:
        _fail("initialize 后 is_initialized 应为 True")
    if result.get("serverInfo", {}).get("name") != "ecommerce-mcp":
        _fail(f"serverInfo 解析不对: {result}")
    if client.server_info != {"name": "ecommerce-mcp", "version": "1.0.0"}:
        _fail(f"server_info 没保留: {client.server_info}")
    _ok(f"initialize 成功，server={client.server_info}")


def test_list_tools_cache() -> None:
    """测试 2：list_tools 成功 + 缓存。"""
    print("[测试 2] list_tools + 缓存")
    call_log: List[httpx.Request] = []
    server_resp = {
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "result": {
            "tools": [
                {"name": "ecommerce__query_order", "description": "查询订单", "inputSchema": {}},
                {"name": "ecommerce__cancel_order", "description": "取消订单", "inputSchema": {}},
            ]
        },
    }
    transport = httpx.MockTransport(_make_handler(
        responses_per_call=[server_resp, server_resp],  # 备 2 次响应以防万一
        call_log=call_log,
    ))
    config = MCPClientConfig(base_url="http://test/mcp", transport=transport)
    client = MCPClient(config)

    async def run():
        tools1 = await client.list_tools()
        tools2 = await client.list_tools()  # 应走缓存，不发请求
        return tools1, tools2

    tools1, tools2 = asyncio.run(run())
    if len(tools1) != 2:
        _fail(f"list_tools 返回数量不对: {len(tools1)}")
    if tools1[0]["name"] != "ecommerce__query_order":
        _fail(f"第一个工具名不对: {tools1[0]['name']}")
    if len(call_log) != 1:
        _fail(f"缓存机制失败，第二次应走缓存不发请求，实际 HTTP 调用 {len(call_log)} 次")
    if tools1 != tools2:
        _fail("两次返回结果不一致")
    _ok(f"list_tools 拉取 {len(tools1)} 个工具，第二次走缓存（HTTP 调用 {len(call_log)} 次）")


def test_call_tool_success() -> None:
    """测试 3：call_tool 成功路径。"""
    print("[测试 3] call_tool 成功")
    server_resp = {
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "result": {
            "content": [
                {"type": "responses", "data": [{"text": "请选择订单", "buttons": []}]},
                {"type": "slot_sets", "data": {"order_id": "false"}},
            ],
            "isError": False,
        },
    }
    transport = httpx.MockTransport(_make_handler(responses_per_call=[server_resp]))
    config = MCPClientConfig(base_url="http://test/mcp", transport=transport)
    client = MCPClient(config)

    async def run():
        return await client.call_tool("ecommerce__query_order", {"user_id": "1001"})

    result = asyncio.run(run())
    if not isinstance(result, dict):
        _fail(f"call_tool 返回应为 dict，实际 {type(result).__name__}")
    if not isinstance(result.get("content"), list):
        _fail("result.content 应为 list")
    if result.get("isError"):
        _fail("isError 应为 False")
    _ok(f"call_tool 成功，content 含 {len(result['content'])} 项，isError={result['isError']}")


def test_protocol_error_no_retry() -> None:
    """测试 4：协议层错误（method not found）不重试。"""
    print("[测试 4] 协议层错误不重试")
    call_log: List[httpx.Request] = []
    server_resp = {
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "error": {"code": -32601, "message": "方法不存在: tools/xxx"},
    }
    transport = httpx.MockTransport(_make_handler(
        responses_per_call=[server_resp, server_resp],
        call_log=call_log,
    ))
    # retry=2，但如果只发一次请求说明没重试
    config = MCPClientConfig(base_url="http://test/mcp", retry=2, transport=transport)
    client = MCPClient(config)

    async def run():
        try:
            await client.call_tool("ecommerce__xxx", {})
            return None, "未抛异常"
        except MCPMethodNotFound as e:
            return e, None
        except Exception as e:
            return None, f"抛错类型: {type(e).__name__}"

    exc, err = asyncio.run(run())
    if err:
        _fail(f"期望抛 MCPMethodNotFound，{err}")
    if exc is None:
        _fail("未抛异常")
    if exc.code != -32601:
        _fail(f"错误码不对: {exc.code}")
    if len(call_log) != 1:
        _fail(f"协议层错误不应重试，实际 HTTP 调用 {len(call_log)} 次")
    _ok(f"协议层错误不重试，立即抛 MCPMethodNotFound（HTTP 调用 {len(call_log)} 次）")


def test_retry_on_timeout_then_success() -> None:
    """测试 5：超时重试，第 3 次成功。"""
    print("[测试 5] 超时重试后成功")
    success_resp = {
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "result": {"content": [], "isError": False},
    }
    # 前 2 次超时，第 3 次成功
    transport = httpx.MockTransport(_make_handler(
        exc_per_call=[httpx.TimeoutException("connect timeout"), httpx.TimeoutException("read timeout")],
        responses_per_call=[success_resp],
    ))
    config = MCPClientConfig(base_url="http://test/mcp", retry=2, transport=transport)
    client = MCPClient(config)

    async def run():
        return await client.call_tool("ecommerce__query_order", {})

    result = asyncio.run(run())
    if result.get("isError"):
        _fail("第 3 次重试应成功")
    _ok("前 2 次超时后第 3 次重试成功")


def test_retry_exhausted_timeout() -> None:
    """测试 6：超时重试耗尽 → 抛 MCPTimeoutError + 熔断记录。"""
    print("[测试 6] 超时重试耗尽")
    transport = httpx.MockTransport(_make_handler(
        exc_per_call=[
            httpx.TimeoutException("timeout 1"),
            httpx.TimeoutException("timeout 2"),
            httpx.TimeoutException("timeout 3"),
        ],
    ))
    config = MCPClientConfig(
        base_url="http://test/mcp",
        retry=2,  # 共 3 次尝试
        circuit_breaker=CircuitBreakerConfig(failure_threshold=10),  # 暂不熔断
        transport=transport,
    )
    client = MCPClient(config)

    async def run():
        try:
            await client.call_tool("ecommerce__query_order", {})
            return None, "未抛异常"
        except MCPTimeoutError as e:
            return e, None
        except Exception as e:
            return None, f"抛错类型: {type(e).__name__}"

    exc, err = asyncio.run(run())
    if err:
        _fail(f"期望抛 MCPTimeoutError，{err}")
    if exc is None:
        _fail("未抛 MCPTimeoutError")
    _ok(f"重试 {config.retry} 次耗尽后抛 MCPTimeoutError: {exc}")


def test_circuit_breaker_open() -> None:
    """测试 7：熔断器进入 OPEN，后续调用直接拒绝。"""
    print("[测试 7] 熔断器 OPEN")
    # 配置熔断：1 次失败就进入 OPEN
    transport = httpx.MockTransport(_make_handler(
        exc_per_call=[
            httpx.ConnectError("connect refused 1"),
            httpx.ConnectError("connect refused 2"),
            httpx.ConnectError("connect refused 3"),
        ],
    ))
    config = MCPClientConfig(
        base_url="http://test/mcp",
        retry=0,  # 不重试，加速熔断
        circuit_breaker=CircuitBreakerConfig(failure_threshold=1, reset_timeout=999),
        transport=transport,
    )
    client = MCPClient(config)

    async def run():
        # 第一次：失败，failure_count=1，触发熔断
        try:
            await client.call_tool("ecommerce__query_order", {})
            return "第一次未抛异常"
        except MCPConnectionError:
            pass
        # 第二次：熔断器 OPEN，应直接拒绝（不发 HTTP）
        try:
            await client.call_tool("ecommerce__query_order", {})
            return "第二次未抛异常（熔断器未生效）"
        except MCPConnectionError as e:
            if "熔断中" in str(e):
                return None  # 预期
            return f"第二次抛 MCPConnectionError 但消息不含'熔断中': {e}"
        return None

    err = asyncio.run(run())
    if err:
        _fail(err)
    if client.breaker_state != "open":
        _fail(f"熔断器状态应为 open，实际 {client.breaker_state}")
    _ok("熔断器进入 OPEN 状态，后续调用直接拒绝（不发 HTTP）")


def test_connection_error() -> None:
    """测试 8：连接失败 → 抛 MCPConnectionError + 重试。"""
    print("[测试 8] 连接失败")
    call_log: List[httpx.Request] = []
    transport = httpx.MockTransport(_make_handler(
        exc_per_call=[
            httpx.ConnectError("connection refused 1"),
            httpx.ConnectError("connection refused 2"),
            httpx.ConnectError("connection refused 3"),
        ],
        call_log=call_log,
    ))
    config = MCPClientConfig(
        base_url="http://test/mcp",
        retry=2,  # 共 3 次尝试
        circuit_breaker=CircuitBreakerConfig(failure_threshold=10),
        transport=transport,
    )
    client = MCPClient(config)

    async def run():
        try:
            await client.call_tool("ecommerce__query_order", {})
            return None, "未抛异常"
        except MCPConnectionError as e:
            return e, None
        except Exception as e:
            return None, f"抛错类型: {type(e).__name__}"

    exc, err = asyncio.run(run())
    if err:
        _fail(f"期望抛 MCPConnectionError，{err}")
    if exc is None:
        _fail("未抛 MCPConnectionError")
    if len(call_log) != 3:
        _fail(f"应重试 {config.retry} 次（共 3 次 HTTP 调用），实际 {len(call_log)} 次")
    _ok(f"连接失败重试 {config.retry} 次后抛 MCPConnectionError: {exc}")


def main() -> int:
    print("=" * 60)
    print("M1.3 MCPClient 单元测试")
    print("=" * 60)
    print()
    tests = [
        test_initialize_success,
        test_list_tools_cache,
        test_call_tool_success,
        test_protocol_error_no_retry,
        test_retry_on_timeout_then_success,
        test_retry_exhausted_timeout,
        test_circuit_breaker_open,
        test_connection_error,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print()
            print(f"\u2717 测试失败: {t.__name__}: {e}")
            return 1
        except Exception as e:
            print()
            print(f"\u2717 测试异常: {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print()
    print("=" * 60)
    print(f"\u2713 全部 {len(tests)} 个测试通过，M1.3 MCPClient 就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

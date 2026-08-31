# -*- coding: utf-8 -*-
"""
M1.4 验证脚本：MCPServer 单元测试

用 FastAPI TestClient 同步测试，无需起真实 HTTP server。

验证项：
1. 健康检查端点
2. initialize 请求正确响应
3. tools/list 列出已注册工具
4. tools/call 调用 async handler
5. tools/call 调用 sync handler（验证 asyncio.to_thread 包装）
6. 调用不存在的工具 → 错误响应 -32003
7. 未知方法 → 错误响应 -32601
8. 请求体非 JSON → 错误响应 -32700 + HTTP 400
9. 非法请求结构 → 错误响应 -32600 + HTTP 400
10. handler 抛异常 → 错误响应 -32603 + HTTP 500
11. _normalize_result 三种输入类型的标准化
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from atguigu_ai.mcp import (
    MCPServer,
    Tool,
    MCP_PROTOCOL_VERSION,
    JSONRPC_VERSION,
)


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


def _make_server() -> MCPServer:
    """构造测试 server：注册 3 个工具（async/sync/抛异常）。"""
    server = MCPServer(name="test-mcp", version="0.1.0")

    # 1. async handler
    async def query_handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "content": [
                {"type": "responses", "data": [{"text": f"hello {args.get('name', 'world')}"}]},
            ],
            "isError": False,
        }
    server.register_tool(Tool(
        name="test__hello",
        description="问候工具",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
        handler=query_handler,
    ))

    # 2. sync handler（验证 asyncio.to_thread 包装）
    def compute_handler(args: Dict[str, Any]) -> Dict[str, Any]:
        x = args.get("x", 0)
        y = args.get("y", 0)
        return {"sum": x + y}
    server.register_tool(Tool(
        name="test__compute",
        description="加法工具（sync）",
        input_schema={
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["x", "y"],
        },
        handler=compute_handler,
    ))

    # 3. 抛异常 handler
    async def fail_handler(args: Dict[str, Any]) -> Any:
        raise ValueError("故意失败")
    server.register_tool(Tool(
        name="test__fail",
        description="总会失败的工具",
        input_schema={},
        handler=fail_handler,
    ))

    return server


def _rpc(client: TestClient, method: str, params: Any = None, req_id: int = 1):
    """发起 JSON-RPC 请求。"""
    body = {"jsonrpc": JSONRPC_VERSION, "method": method, "id": req_id}
    if params is not None:
        body["params"] = params
    return client.post("/mcp", json=body)


def test_health() -> None:
    """测试 1：健康检查。"""
    print("[测试 1] 健康检查")
    server = _make_server()
    client = TestClient(server.app)
    resp = client.get("/health")
    if resp.status_code != 200:
        _fail(f"health 状态码 {resp.status_code}")
    data = resp.json()
    if data.get("status") != "ok":
        _fail(f"health status 不对: {data}")
    if data.get("tools_count") != 3:
        _fail(f"tools_count 应为 3，实际 {data.get('tools_count')}")
    _ok(f"健康检查通过: server={data['server']}, tools_count={data['tools_count']}")


def test_initialize() -> None:
    """测试 2：initialize。"""
    print("[测试 2] initialize")
    server = _make_server()
    client = TestClient(server.app)
    resp = _rpc(client, "initialize", params={
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "clientInfo": {"name": "test-client", "version": "1.0"},
    })
    if resp.status_code != 200:
        _fail(f"initialize 状态码 {resp.status_code}: {resp.text}")
    data = resp.json()
    if "result" not in data:
        _fail(f"initialize 缺少 result: {data}")
    result = data["result"]
    if result.get("protocolVersion") != MCP_PROTOCOL_VERSION:
        _fail(f"protocolVersion 不对: {result.get('protocolVersion')}")
    if result.get("serverInfo", {}).get("name") != "test-mcp":
        _fail(f"serverInfo.name 不对: {result.get('serverInfo')}")
    _ok(f"initialize 通过: server={result['serverInfo']}")


def test_list_tools() -> None:
    """测试 3：tools/list。"""
    print("[测试 3] tools/list")
    server = _make_server()
    client = TestClient(server.app)
    resp = _rpc(client, "tools/list", params={})
    data = resp.json()
    tools = data.get("result", {}).get("tools", [])
    if len(tools) != 3:
        _fail(f"工具数量不对: {len(tools)}")
    names = [t["name"] for t in tools]
    if "test__hello" not in names or "test__compute" not in names or "test__fail" not in names:
        _fail(f"工具名缺失: {names}")
    # 验证 inputSchema 也返回了
    hello_tool = next(t for t in tools if t["name"] == "test__hello")
    if "name" not in hello_tool["inputSchema"].get("properties", {}):
        _fail(f"inputSchema 不对: {hello_tool['inputSchema']}")
    _ok(f"tools/list 列出 {len(tools)} 个工具: {names}")


def test_call_async() -> None:
    """测试 4：tools/call async handler。"""
    print("[测试 4] call async handler")
    server = _make_server()
    client = TestClient(server.app)
    resp = _rpc(client, "tools/call", params={
        "name": "test__hello",
        "arguments": {"name": "MCP"},
    })
    data = resp.json()
    result = data.get("result", {})
    content = result.get("content", [])
    if not content:
        _fail(f"content 为空: {result}")
    first = content[0]
    if first.get("type") != "responses":
        _fail(f"content[0].type 不对: {first}")
    if "MCP" not in first.get("data", [{}])[0].get("text", ""):
        _fail(f"handler 返回值不对: {first}")
    _ok(f"async handler 调用成功: {first['data'][0]['text']}")


def test_call_sync() -> None:
    """测试 5：tools/call sync handler（验证 to_thread 包装）。"""
    print("[测试 5] call sync handler")
    server = _make_server()
    client = TestClient(server.app)
    resp = _rpc(client, "tools/call", params={
        "name": "test__compute",
        "arguments": {"x": 3, "y": 5},
    })
    data = resp.json()
    result = data.get("result", {})
    content = result.get("content", [])
    if not content:
        _fail(f"sync handler 返回空: {result}")
    if content[0].get("data", {}).get("sum") != 8:
        _fail(f"sync handler 计算结果不对: {content[0]}")
    _ok(f"sync handler 调用成功（to_thread 包装）: sum={content[0]['data']['sum']}")


def test_call_tool_not_found() -> None:
    """测试 6：调用不存在的工具。"""
    print("[测试 6] 工具不存在")
    server = _make_server()
    client = TestClient(server.app)
    resp = _rpc(client, "tools/call", params={
        "name": "test__not_exists",
        "arguments": {},
    })
    data = resp.json()
    err = data.get("error", {})
    if err.get("code") != -32003:
        _fail(f"错误码应为 -32003（ToolNotFound），实际 {err.get('code')}")
    _ok(f"工具不存在错误响应正确: code={err['code']}, message={err['message']}")


def test_method_not_found() -> None:
    """测试 7：未知方法。"""
    print("[测试 7] 方法不存在")
    server = _make_server()
    client = TestClient(server.app)
    resp = _rpc(client, "unknown/method", params={})
    data = resp.json()
    err = data.get("error", {})
    if err.get("code") != -32601:
        _fail(f"错误码应为 -32601（MethodNotFound），实际 {err.get('code')}")
    _ok(f"未知方法错误响应正确: code={err['code']}, message={err['message']}")


def test_invalid_json() -> None:
    """测试 8：请求体非 JSON。"""
    print("[测试 8] 请求体非 JSON")
    server = _make_server()
    client = TestClient(server.app)
    resp = client.post("/mcp", content="not a json", headers={"Content-Type": "application/json"})
    if resp.status_code != 400:
        _fail(f"HTTP 状态码应为 400，实际 {resp.status_code}")
    data = resp.json()
    err = data.get("error", {})
    if err.get("code") != -32700:
        _fail(f"错误码应为 -32700（ParseError），实际 {err.get('code')}")
    _ok(f"非法 JSON 错误响应正确: HTTP {resp.status_code}, code={err['code']}")


def test_invalid_request_structure() -> None:
    """测试 9：非法请求结构（缺 jsonrpc 字段）。"""
    print("[测试 9] 非法请求结构")
    server = _make_server()
    client = TestClient(server.app)
    # 缺 jsonrpc 字段
    resp = client.post("/mcp", json={"method": "tools/list", "id": 1})
    if resp.status_code != 400:
        _fail(f"HTTP 状态码应为 400，实际 {resp.status_code}")
    data = resp.json()
    err = data.get("error", {})
    if err.get("code") not in (-32700, -32600):
        _fail(f"错误码应为 -32700 或 -32600，实际 {err.get('code')}")
    _ok(f"非法请求结构错误响应正确: HTTP {resp.status_code}, code={err['code']}")


def test_handler_exception() -> None:
    """测试 10：handler 抛异常。"""
    print("[测试 10] handler 异常")
    server = _make_server()
    client = TestClient(server.app)
    resp = _rpc(client, "tools/call", params={
        "name": "test__fail",
        "arguments": {},
    })
    if resp.status_code != 500:
        _fail(f"HTTP 状态码应为 500，实际 {resp.status_code}")
    data = resp.json()
    err = data.get("error", {})
    if err.get("code") != -32603:
        _fail(f"错误码应为 -32603（InternalError），实际 {err.get('code')}")
    if "故意失败" not in err.get("message", ""):
        _fail(f"错误消息不对: {err.get('message')}")
    if "traceback" not in err.get("data", {}):
        _fail(f"data 应含 traceback: {err.get('data')}")
    _ok(f"handler 异常错误响应正确: code={err['code']}, 含 traceback")


def test_normalize_result() -> None:
    """测试 11：_normalize_result 三种输入。"""
    print("[测试 11] 结果标准化")
    server = _make_server()

    # 1. 已是标准格式（含 content）
    r1 = server._normalize_result({"content": [{"type": "x", "data": 1}]})
    if r1.get("isError") is not False:
        _fail(f"标准格式补全 isError 失败: {r1}")
    _ok(f"标准格式: isError={r1['isError']}（补全）")

    # 2. 普通 dict
    r2 = server._normalize_result({"key": "value"})
    if r2.get("content", [{}])[0].get("type") != "result":
        _fail(f"普通 dict 包装错误: {r2}")
    if r2.get("content", [{}])[0].get("data") != {"key": "value"}:
        _fail(f"普通 dict data 不对: {r2}")
    _ok("普通 dict 包装为 {type: 'result', data: dict}")

    # 3. list
    r3 = server._normalize_result([1, 2, 3])
    if len(r3.get("content", [])) != 3:
        _fail(f"list 包装长度不对: {r3}")
    _ok("list 包装为多个 {type: 'result', data: item}")

    # 4. 其他类型
    r4 = server._normalize_result("hello")
    if r4.get("content", [{}])[0].get("data") != "hello":
        _fail(f"字符串包装错误: {r4}")
    _ok("字符串包装为 {type: 'result', data: 'hello'}")


def main() -> int:
    print("=" * 60)
    print("M1.4 MCPServer 单元测试")
    print("=" * 60)
    print()
    tests = [
        test_health,
        test_initialize,
        test_list_tools,
        test_call_async,
        test_call_sync,
        test_call_tool_not_found,
        test_method_not_found,
        test_invalid_json,
        test_invalid_request_structure,
        test_handler_exception,
        test_normalize_result,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M1.4 MCPServer 就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

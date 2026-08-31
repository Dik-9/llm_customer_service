# -*- coding: utf-8 -*-
"""
M1.2 验证脚本：JSON-RPC 2.0 消息封装单元测试

验证项：
1. Request/Response/Error/Notification 的 to_dict/from_dict 往返
2. 可选字段省略时不输出（params=None 不输出 params）
3. 判定函数 is_request/notification/response/error_response
4. parse_message 自动识别 4 种消息类型
5. parse_message 对非法结构抛 MCPParseError
6. JsonRpcErrorResponse.to_exception 转换为对应 MCP 异常
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atguigu_ai.mcp import (
    JSONRPC_VERSION,
    MCP_PROTOCOL_VERSION,
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
    MCPParseError,
    MCPMethodNotFound,
)


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


def test_request_roundtrip() -> None:
    """测试 1：Request 往返序列化。"""
    print("[测试 1] Request 往返")
    req = JsonRpcRequest(
        method="tools/call",
        params={"name": "ecommerce__query_order", "arguments": {"user_id": "1001"}},
        id=42,
    )
    d = req.to_dict()
    # 验证 dict 结构符合 JSON-RPC 2.0 规范
    expected_keys = {"jsonrpc", "method", "params", "id"}
    if set(d.keys()) != expected_keys:
        _fail(f"Request dict 字段不对: {set(d.keys())}，期望 {expected_keys}")
    if d["jsonrpc"] != JSONRPC_VERSION:
        _fail(f"jsonrpc 字段值不对: {d['jsonrpc']}")
    _ok(f"to_dict 字段完整: {sorted(d.keys())}")

    # from_dict 反序列化
    req2 = JsonRpcRequest.from_dict(d)
    if req2.method != req.method or req2.params != req.params or req2.id != req.id:
        _fail("from_dict 后字段值不一致")
    _ok(f"from_dict 往返一致: method={req2.method}, id={req2.id}")


def test_optional_omit() -> None:
    """测试 2：可选字段省略时不输出。"""
    print("[测试 2] 可选字段省略")
    req = JsonRpcRequest(method="tools/list", id=1)  # params=None
    d = req.to_dict()
    if "params" in d:
        _fail(f"params=None 时不应输出 params 字段，实际 dict: {d}")
    _ok(f"params=None 时省略输出: {sorted(d.keys())}")

    # id=None 时也应省略（虽然 JSON-RPC 不推荐 id=null，但本期省略更干净）
    req_no_id = JsonRpcRequest(method="initialized")
    d2 = req_no_id.to_dict()
    if "id" in d2:
        _fail(f"id=None 时不应输出 id 字段，实际 dict: {d2}")
    _ok(f"id=None 时省略输出: {sorted(d2.keys())}")


def test_response_roundtrip() -> None:
    """测试 3：Response 往返。"""
    print("[测试 3] Response 往返")
    resp = JsonRpcResponse(
        id=42,
        result={"content": [{"type": "responses", "data": [{"text": "请选择订单"}]}]},
    )
    d = resp.to_dict()
    if set(d.keys()) != {"jsonrpc", "id", "result"}:
        _fail(f"Response dict 字段不对: {set(d.keys())}")
    resp2 = JsonRpcResponse.from_dict(d)
    if resp2.id != resp.id or resp2.result != resp.result:
        _fail("Response from_dict 后字段不一致")
    _ok(f"Response 往返一致: id={resp2.id}, result keys={list(resp2.result.keys())}")


def test_error_response() -> None:
    """测试 4：ErrorResponse 含嵌套 error 对象。"""
    print("[测试 4] ErrorResponse 嵌套")
    err = JsonRpcError(code=-32601, message="方法不存在", data={"method": "tools/xxx"})
    err_resp = JsonRpcErrorResponse(id=42, error=err)
    d = err_resp.to_dict()
    if set(d.keys()) != {"jsonrpc", "id", "error"}:
        _fail(f"ErrorResponse dict 字段不对: {set(d.keys())}")
    if not isinstance(d["error"], dict):
        _fail("error 字段应该是 dict")
    if d["error"].get("code") != -32601 or d["error"].get("data") != {"method": "tools/xxx"}:
        _fail(f"嵌套 error 对象内容不对: {d['error']}")
    _ok(f"ErrorResponse 嵌套 error 对象正确: code={d['error']['code']}")

    err_resp2 = JsonRpcErrorResponse.from_dict(d)
    if err_resp2.error.code != err.code or err_resp2.error.data != err.data:
        _fail("ErrorResponse from_dict 后嵌套 error 不一致")
    _ok("ErrorResponse from_dict 往返一致")


def test_notification_no_id() -> None:
    """测试 5：Notification 无 id。"""
    print("[测试 5] Notification 无 id")
    notif = JsonRpcNotification(method="progress/update", params={"progress": 0.5})
    d = notif.to_dict()
    if "id" in d:
        _fail(f"Notification 不应有 id 字段: {d}")
    if set(d.keys()) != {"jsonrpc", "method", "params"}:
        _fail(f"Notification dict 字段不对: {set(d.keys())}")
    notif2 = JsonRpcNotification.from_dict(d)
    if notif2.method != notif.method:
        _fail("Notification from_dict 不一致")
    _ok(f"Notification 无 id 字段，method={notif2.method}")


def test_judgment_functions() -> None:
    """测试 6：4 个判定函数。"""
    print("[测试 6] 判定函数")
    req_d = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
    notif_d = {"jsonrpc": "2.0", "method": "progress"}  # 无 id
    resp_d = {"jsonrpc": "2.0", "id": 1, "result": {}}
    err_d = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "x"}}

    if not is_request(req_d):
        _fail("is_request 未识别合法 Request")
    if is_notification(req_d):
        _fail("is_notification 不应识别 Request（带 id）")
    if not is_notification(notif_d):
        _fail("is_notification 未识别合法 Notification（无 id）")
    if not is_response(resp_d):
        _fail("is_response 未识别合法 Response")
    if not is_error_response(err_d):
        _fail("is_error_response 未识别合法 ErrorResponse")
    # 边界：null id 的请求
    if is_request({"jsonrpc": "2.0", "method": "x"}):
        _fail("is_request 不应识别无 id 的消息为 Request")
    _ok("4 个判定函数全部正确（含边界）")


def test_parse_message_auto() -> None:
    """测试 7：parse_message 自动识别 4 种类型。"""
    print("[测试 7] parse_message 自动识别")
    cases = [
        ({"jsonrpc": "2.0", "method": "tools/list", "id": 1}, JsonRpcRequest),
        ({"jsonrpc": "2.0", "method": "progress", "params": {}}, JsonRpcNotification),
        ({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}, JsonRpcResponse),
        ({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "x"}}, JsonRpcErrorResponse),
    ]
    for d, expected_cls in cases:
        msg = parse_message(d)
        if not isinstance(msg, expected_cls):
            _fail(f"parse_message 识别错误: 期望 {expected_cls.__name__}，实际 {type(msg).__name__}")
    _ok("4 种消息类型自动识别全部正确")


def test_parse_message_invalid() -> None:
    """测试 8：parse_message 对非法结构抛 MCPParseError。"""
    print("[测试 8] 非法结构处理")
    invalid_cases = [
        "not a dict",                                              # 非 dict
        {"method": "tools/list", "id": 1},                          # 缺 jsonrpc
        {"jsonrpc": "2.0", "id": 1, "result": {}, "error": {}},     # 同时有 result 和 error
        {"jsonrpc": "2.0", "id": 1},                                # 既无 method 也无 result/error
    ]
    for d in invalid_cases:
        try:
            parse_message(d)
            _fail(f"应抛 MCPParseError，但解析成功: {d}")
        except MCPParseError:
            pass  # 预期行为
    _ok(f"{len(invalid_cases)} 种非法结构全部抛 MCPParseError")


def test_error_to_exception() -> None:
    """测试 9：ErrorResponse.to_exception 转换为对应异常。"""
    print("[测试 9] ErrorResponse 转异常")
    err = JsonRpcError(code=-32601, message="方法不存在: tools/xxx")
    err_resp = JsonRpcErrorResponse(id=42, error=err)
    exc = err_resp.to_exception()
    if not isinstance(exc, MCPMethodNotFound):
        _fail(f"to_exception 返回 {type(exc).__name__}，期望 MCPMethodNotFound")
    if exc.code != -32601 or exc.message != "方法不存在: tools/xxx":
        _fail(f"to_exception 后异常字段不对: code={exc.code}, message={exc.message}")
    _ok(f"ErrorResponse 转 MCPMethodNotFound 异常正确: code={exc.code}")


def main() -> int:
    print("=" * 60)
    print("M1.2 JSON-RPC 2.0 消息封装单元测试")
    print("=" * 60)
    print()
    tests = [
        test_request_roundtrip,
        test_optional_omit,
        test_response_roundtrip,
        test_error_response,
        test_notification_no_id,
        test_judgment_functions,
        test_parse_message_auto,
        test_parse_message_invalid,
        test_error_to_exception,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M1.2 消息封装就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

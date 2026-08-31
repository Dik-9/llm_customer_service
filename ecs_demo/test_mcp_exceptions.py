# -*- coding: utf-8 -*-
"""
M1.1 验证脚本：异常体系单元测试

验证项：
1. 异常层级正确（所有子类 isinstance MCPError）
2. 错误码符合 JSON-RPC 2.0 规范
3. to_dict 序列化符合 error 对象格式
4. error_from_code 反向映射正确
5. FallbackTriggered 不属于 MCPError（控制流信号隔离）
6. except 捕获行为符合预期
"""
from __future__ import annotations

import sys
from pathlib import Path

# 把项目根目录加入 sys.path，使 atguigu_ai 可被 import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atguigu_ai.mcp import (
    MCPError,
    MCPConnectionError,
    MCPTimeoutError,
    MCPParseError,
    MCPInvalidRequest,
    MCPMethodNotFound,
    MCPInvalidParams,
    MCPInternalError,
    MCPToolNotFoundError,
    FallbackTriggered,
    error_from_code,
)


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


def test_hierarchy() -> None:
    """测试 1：异常层级正确。"""
    print("[测试 1] 异常层级")
    cases = [
        (MCPConnectionError(), MCPError),
        (MCPTimeoutError(), MCPError),
        (MCPParseError(), MCPError),
        (MCPInvalidRequest(), MCPError),
        (MCPMethodNotFound(), MCPError),
        (MCPInvalidParams(), MCPError),
        (MCPInternalError(), MCPError),
        (MCPToolNotFoundError(), MCPError),
    ]
    for exc, base in cases:
        if not isinstance(exc, base):
            _fail(f"{type(exc).__name__} 不是 {base.__name__} 的子类")
    # 协议层子类也应同时是 MCPProtocolError 的子类
    from atguigu_ai.mcp.exceptions import MCPProtocolError
    for exc_cls in [MCPParseError, MCPInvalidRequest, MCPMethodNotFound, MCPInvalidParams, MCPInternalError]:
        if not issubclass(exc_cls, MCPProtocolError):
            _fail(f"{exc_cls.__name__} 不是 MCPProtocolError 的子类")
    _ok("所有异常继承关系正确")


def test_error_codes() -> None:
    """测试 2：错误码符合 JSON-RPC 2.0 规范。"""
    print("[测试 2] 错误码规范")
    expected = {
        MCPParseError: -32700,
        MCPInvalidRequest: -32600,
        MCPMethodNotFound: -32601,
        MCPInvalidParams: -32602,
        MCPInternalError: -32603,
        MCPConnectionError: -32001,
        MCPTimeoutError: -32002,
        MCPToolNotFoundError: -32003,
    }
    for cls, code in expected.items():
        exc = cls()
        if exc.code != code:
            _fail(f"{cls.__name__}.code = {exc.code}，期望 {code}")
    _ok(f"8 个异常类的错误码全部正确（-32700 ~ -32003）")


def test_str_and_to_dict() -> None:
    """测试 3：__str__ 和 to_dict 序列化。"""
    print("[测试 3] 序列化")
    exc = MCPToolNotFoundError("ecommerce__query_order")
    s = str(exc)
    if "ecommerce__query_order" not in s or "-32003" not in s:
        _fail(f"__str__ 输出异常: {s}")
    _ok(f"__str__ 输出正确: {s}")

    d = exc.to_dict()
    if d.get("code") != -32003 or d.get("message") != "工具不存在: ecommerce__query_order":
        _fail(f"to_dict 输出异常: {d}")
    _ok(f"to_dict 输出正确: code={d['code']}, message={d['message']}")

    # 带 data 的序列化
    exc2 = MCPInternalError("工具执行失败", data={"traceback": "ZeroDivisionError"})
    d2 = exc2.to_dict()
    if d2.get("data") != {"traceback": "ZeroDivisionError"}:
        _fail(f"to_dict 没有保留 data: {d2}")
    _ok(f"to_dict 保留 data 字段: {d2['data']}")


def test_error_from_code() -> None:
    """测试 4：error_from_code 反向映射。"""
    print("[测试 4] error_from_code 反向映射")
    cases = [
        (-32700, MCPParseError),
        (-32600, MCPInvalidRequest),
        (-32601, MCPMethodNotFound),
        (-32602, MCPInvalidParams),
        (-32603, MCPInternalError),
        (-32001, MCPConnectionError),
        (-32002, MCPTimeoutError),
        (-32003, MCPToolNotFoundError),
        (-99999, MCPInternalError),  # 未知码 fallback
    ]
    for code, expected_cls in cases:
        exc = error_from_code(code, message=f"code={code}")
        if not isinstance(exc, expected_cls):
            _fail(f"error_from_code({code}) 返回 {type(exc).__name__}，期望 {expected_cls.__name__}")
        if exc.code != code if code != -99999 else exc.code != -32603:
            _fail(f"error_from_code({code}) 返回实例的 code 不对: {exc.code}")
    _ok(f"9 种错误码（含未知码 fallback）映射全部正确")


def test_fallback_isolation() -> None:
    """测试 5：FallbackTriggered 不属于 MCPError。"""
    print("[测试 5] 降级信号隔离")
    fb = FallbackTriggered(reason="MCP 不可达", action_name="action_ask_order_id")
    if isinstance(fb, MCPError):
        _fail("FallbackTriggered 不应是 MCPError 的子类（控制流信号需隔离）")
    _ok("FallbackTriggered 不是 MCPError 子类，控制流信号隔离正确")

    s = str(fb)
    if "action_ask_order_id" not in s or "MCP 不可达" not in s:
        _fail(f"__str__ 输出异常: {s}")
    _ok(f"__str__ 输出正确: {s}")

    if fb.cause is not None:
        _fail("cause 默认应为 None")
    cause = RuntimeError("connect ECONNREFUSED")
    fb2 = FallbackTriggered(reason="超时", action_name="x", cause=cause)
    if fb2.cause is not cause:
        _fail("cause 字段保留原始异常失败")
    _ok("cause 字段保留原始异常正确")


def test_except_capture() -> None:
    """测试 6：except 捕获行为。"""
    print("[测试 6] except 捕获行为")

    # MCPConnectionError 应被 except MCPError 捕获
    caught = False
    try:
        raise MCPConnectionError("服务未启动")
    except MCPError as e:
        caught = True
        assert isinstance(e, MCPConnectionError)
    if not caught:
        _fail("MCPConnectionError 未被 except MCPError 捕获")
    _ok("MCPError 可统一捕获所有协议层异常")

    # FallbackTriggered 不应被 except MCPError 捕获（这是设计意图）
    caught_as_mcp = False
    actual_caught = False
    try:
        raise FallbackTriggered(reason="超时降级", action_name="x")
    except MCPError:
        caught_as_mcp = True
    except FallbackTriggered:
        actual_caught = True
    if caught_as_mcp:
        _fail("FallbackTriggered 不应被 except MCPError 捕获")
    if not actual_caught:
        _fail("FallbackTriggered 未被自身 except 捕获")
    _ok("FallbackTriggered 不被 MCPError 捕获，独立走降级路径")


def main() -> int:
    print("=" * 60)
    print("M1.1 异常体系单元测试")
    print("=" * 60)
    print()
    tests = [
        test_hierarchy,
        test_error_codes,
        test_str_and_to_dict,
        test_error_from_code,
        test_fallback_isolation,
        test_except_capture,
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
    print("\u2713 全部 6 个测试通过，M1.1 异常体系就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

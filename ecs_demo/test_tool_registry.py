# -*- coding: utf-8 -*-
"""
M1.6 验证脚本：ToolRegistry 单元测试

确定性验证（不连真实 MCP server，用 stub client + mock tracker）：
1. LocalExecutable 包装 get_action 本地直调（用内置 action_send_text）
2. ToolRegistry 路由：在映射表 → MCPExecutable；不在 → LocalExecutable
3. MCP 关闭（mcp_client=None）全部走 LocalExecutable
4. MCPExecutable 成功：把 MCP 响应转回 ActionResult + apply 副作用到 tracker
5. MCPExecutable 失败（MCPError）→ 降级 fallback
6. mcp_result_to_action_result 单元测试（responses/slot_sets/reject_action_listen/events/异常格式）
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atguigu_ai.agent.actions import ActionResult
from atguigu_ai.mcp.exceptions import MCPConnectionError, MCPTimeoutError
from atguigu_ai.mcp.tool_registry import (
    Executable,
    LocalExecutable,
    MCPExecutable,
    ToolRegistry,
    mcp_result_to_action_result,
)


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- 测试辅助 ----------

class _MockTracker:
    """轻量 tracker：记录 slot 读写，供验证 apply 副作用。"""
    def __init__(self, slots: Optional[Dict[str, Any]] = None) -> None:
        self._slots: Dict[str, Any] = dict(slots or {})

    def get_slot(self, name: str) -> Any:
        return self._slots.get(name)

    def set_slot(self, name: str, value: Any) -> None:
        self._slots[name] = value

    def get_all_slots(self) -> Dict[str, Any]:
        return dict(self._slots)


class _StubMCPClient:
    """Stub MCPClient：call_tool 返回预设 result 或抛预设异常。"""
    def __init__(self, result: Any = None, exc: Optional[Exception] = None) -> None:
        self._result = result
        self._exc = exc
        self.call_count = 0
        self.last_args: Optional[Dict[str, Any]] = None

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        self.call_count += 1
        self.last_args = arguments
        if self._exc is not None:
            raise self._exc
        return self._result

    async def close(self) -> None:
        pass


class _MockFallback(Executable):
    """降级 mock：记录是否被调用，返回固定结果。"""
    def __init__(self, result: ActionResult) -> None:
        self._result = result
        self.called = False

    async def run(self, tracker: Any, domain: Optional[Any] = None, **kwargs: Any) -> ActionResult:
        self.called = True
        return self._result


# ---------- 测试用例 ----------

def test_local_executable() -> None:
    """测试 1：LocalExecutable 本地直调（action_send_text）。"""
    print("[测试 1] LocalExecutable 本地直调")
    exe = LocalExecutable("action_send_text")
    tracker = _MockTracker()
    result = asyncio.run(exe.run(tracker, domain=None, text="hello world"))
    if not result.responses:
        _fail(f"应返回 1 个 response，实际 {result.responses}")
    if result.responses[0].get("text") != "hello world":
        _fail(f"text 不对: {result.responses[0]}")
    _ok(f"LocalExecutable 调 action_send_text 返回: {result.responses[0]['text']}")

    # 找不到的 action 返回失败结果（不抛异常）
    exe2 = LocalExecutable("action_not_exists_xyz")
    result2 = asyncio.run(exe2.run(tracker))
    if result2.success:
        _fail("找不到 action 时 success 应为 False")
    _ok("找不到 action 时返回失败结果（不抛异常）")


def test_registry_route() -> None:
    """测试 2：ToolRegistry 路由优先级。"""
    print("[测试 2] 路由优先级")
    stub = _StubMCPClient()
    registry = ToolRegistry(
        mcp_client=stub,  # type: ignore
        mcp_mapping={"action_ask_order_id": "ecommerce__query_order"},
    )
    # 在映射表 → MCPExecutable
    exe = registry.get("action_ask_order_id")
    if not isinstance(exe, MCPExecutable):
        _fail(f"映射表内的 action 应返回 MCPExecutable，实际 {type(exe).__name__}")
    # 不在映射表 → LocalExecutable
    exe2 = registry.get("action_send_text")
    if not isinstance(exe2, LocalExecutable):
        _fail(f"映射表外的 action 应返回 LocalExecutable，实际 {type(exe2).__name__}")
    _ok("路由：映射表内→MCPExecutable，表外→LocalExecutable")


def test_registry_mcp_disabled() -> None:
    """测试 3：MCP 关闭全部走本地。"""
    print("[测试 3] MCP 关闭")
    registry = ToolRegistry(mcp_client=None, mcp_mapping={"action_ask_order_id": "ecommerce__query_order"})
    exe = registry.get("action_ask_order_id")
    if not isinstance(exe, LocalExecutable):
        _fail(f"MCP 关闭时应返回 LocalExecutable，实际 {type(exe).__name__}")
    if registry.mcp_enabled:
        _fail("mcp_enabled 应为 False")
    _ok("MCP 关闭：映射表内 action 也走 LocalExecutable（等价基线）")


def test_mcp_success_apply_side_effects() -> None:
    """测试 4：MCPExecutable 成功，副作用 apply 到 tracker。"""
    print("[测试 4] MCP 成功 apply 副作用")
    mcp_result = {
        "content": [
            {"type": "responses", "data": [{"text": "请选择订单", "buttons": [{"title": "O001"}]}]},
            {"type": "slot_sets", "data": {"order_id": "false"}},
            {"type": "reject_action_listen", "data": True},
            {"type": "events", "data": [{"event": "slot_set", "name": "order_id", "value": "false"}]},
        ],
        "isError": False,
    }
    stub = _StubMCPClient(result=mcp_result)
    fallback = _MockFallback(ActionResult())
    exe = MCPExecutable(
        action_name="action_ask_order_id",
        mcp_tool_name="ecommerce__query_order",
        mcp_client=stub,  # type: ignore
        fallback=fallback,
    )
    tracker = _MockTracker(slots={"user_id": "1001", "goto": "action_ask_order_id_before_delivered"})
    result = asyncio.run(exe.run(tracker))

    # 降级不应触发
    if fallback.called:
        _fail("MCP 成功时不应触发降级")
    # responses 正确
    if not result.responses or result.responses[0].get("text") != "请选择订单":
        _fail(f"responses 不对: {result.responses}")
    if result.responses[0].get("buttons") != [{"title": "O001"}]:
        _fail(f"buttons 透传失败: {result.responses[0]}")
    # slot_sets apply 到 tracker
    if tracker.get_slot("order_id") != "false":
        _fail(f"slot_sets 未 apply 到 tracker: order_id={tracker.get_slot('order_id')}")
    # reject_action_listen
    if not getattr(result, "reject_action_listen", False):
        _fail("reject_action_listen 未设置")
    # events
    if not result.events or result.events[0].get("name") != "order_id":
        _fail(f"events 不对: {result.events}")
    # isError=False → success=True
    if not result.success:
        _fail("isError=False 时 success 应为 True")
    # args 从 tracker.get_all_slots 取
    if stub.last_args != {"user_id": "1001", "goto": "action_ask_order_id_before_delivered"}:
        _fail(f"args 未从 tracker slots 取: {stub.last_args}")
    _ok("MCP 成功：responses/buttons/slot_sets/reject_action_listen/events 全部 apply，args 取自 tracker slots")


def test_mcp_fallback_on_error() -> None:
    """测试 5：MCP 失败（MCPError）降级 fallback。"""
    print("[测试 5] MCP 失败降级")
    # 超时
    stub = _StubMCPClient(exc=MCPTimeoutError("timeout 3"))
    fallback_result = ActionResult()
    fallback_result.add_response("本地兜底结果")
    fallback = _MockFallback(fallback_result)
    exe = MCPExecutable(
        action_name="action_ask_order_id",
        mcp_tool_name="ecommerce__query_order",
        mcp_client=stub,  # type: ignore
        fallback=fallback,
    )
    tracker = _MockTracker(slots={"user_id": "1001"})
    result = asyncio.run(exe.run(tracker))
    if not fallback.called:
        _fail("MCP 失败应触发 fallback")
    if not result.responses or result.responses[0].get("text") != "本地兜底结果":
        _fail(f"降级结果不对: {result.responses}")
    if stub.call_count != 1:
        _fail(f"应只调一次 MCP（降级不重试 MCP），实际 {stub.call_count}")
    _ok("MCP 超时 → 降级 fallback，返回本地结果")

    # 连接失败也降级
    stub2 = _StubMCPClient(exc=MCPConnectionError("refused"))
    fallback2 = _MockFallback(ActionResult())
    exe2 = MCPExecutable("a", "t", stub2, fallback2)  # type: ignore
    asyncio.run(exe2.run(_MockTracker()))
    if not fallback2.called:
        _fail("连接失败也应降级")
    _ok("MCP 连接失败 → 降级 fallback")


def test_mcp_result_to_action_result_unit() -> None:
    """测试 6：mcp_result_to_action_result 单元测试。"""
    print("[测试 6] 转换函数单元")
    tracker = _MockTracker()

    # 6.1 空内容
    r1 = mcp_result_to_action_result({"content": [], "isError": False}, tracker)
    if r1.responses or not r1.success:
        _fail(f"空内容应无 responses 且 success=True: {r1.responses}")
    _ok("空 content → 无 responses，success=True")

    # 6.2 isError=True
    r2 = mcp_result_to_action_result({"content": [{"type": "responses", "data": [{"text": "出错了"}]}], "isError": True}, tracker)
    if r2.success:
        _fail("isError=True 时 success 应为 False")
    _ok("isError=True → success=False")

    # 6.3 非 dict 返回（异常格式）
    r3 = mcp_result_to_action_result("not a dict", tracker)
    if r3.success:
        _fail("非 dict 返回应 success=False")
    _ok("非 dict 返回 → success=False + 错误提示")

    # 6.4 responses 多条
    r4 = mcp_result_to_action_result(
        {"content": [{"type": "responses", "data": [{"text": "A"}, {"text": "B", "buttons": [{"title": "x"}]}]}],
         "isError": False}, tracker)
    if len(r4.responses) != 2 or r4.responses[1].get("buttons") != [{"title": "x"}]:
        _fail(f"多条 responses 不对: {r4.responses}")
    _ok("多条 responses + buttons 透传正确")


def main() -> int:
    print("=" * 60)
    print("M1.6 ToolRegistry 单元测试")
    print("=" * 60)
    print()
    tests = [
        test_local_executable,
        test_registry_route,
        test_registry_mcp_disabled,
        test_mcp_success_apply_side_effects,
        test_mcp_fallback_on_error,
        test_mcp_result_to_action_result_unit,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M1.6 ToolRegistry 就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

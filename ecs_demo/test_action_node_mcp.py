# -*- coding: utf-8 -*-
"""
M1.8 验证脚本：action_node 集成 ToolRegistry 统一执行入口

确定性验证（不连真实 MCP server，不调 LLM，用 stub client + mock tracker）：
1. _execute_action：MCP 路由存在 → MCPExecutable（stub 返回结果）
2. _execute_action：无路由 / MCP 关闭 → 本地 get_action 直调（action_send_text）
3. _execute_action：本地未找到 → 返回 None（基线 not-found 信号）
4. action_node：MCP 路由动作 → MCP 结果累积 + slot_sets apply + bot 消息 + latest_action_name
5. action_node：MCP 关闭（无 _tool_registry）→ 本地直调（基线等价，SPEC §6.4）
6. action_node：未找到动作 → 基线 not-found 失败（无响应累积）
7. action_node：utter 无响应 → fallback 走统一执行入口
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atguigu_ai.agent.actions import Action, ActionResult, register_action
from atguigu_ai.agent.graph.nodes.action import _execute_action, action_node
from atguigu_ai.mcp.tool_registry import ToolRegistry


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- 测试辅助 ----------

class _MockTracker:
    """轻量 tracker：记录 slot 读写 + bot 消息 + latest_action_name。"""
    def __init__(self, slots: Optional[Dict[str, Any]] = None) -> None:
        self._slots: Dict[str, Any] = dict(slots or {})
        self.latest_action_name: Optional[str] = None
        self.bot_messages: list = []

    def get_slot(self, name: str) -> Any:
        return self._slots.get(name)

    def set_slot(self, name: str, value: Any) -> None:
        self._slots[name] = value

    def get_all_slots(self) -> Dict[str, Any]:
        return dict(self._slots)

    def add_bot_message(self, message: Any) -> None:
        self.bot_messages.append(message)


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


class _EmptyUtterAction(Action):
    """测试用 utter 动作：返回空 responses，用于触发 fallback 分支。"""
    @property
    def name(self) -> str:
        return "utter_empty_test"

    async def run(self, tracker: Any, domain: Optional[Any] = None, **kwargs: Any) -> ActionResult:
        return ActionResult()  # 无 responses


def _make_state(
    action_name: str,
    tracker: _MockTracker,
    metadata: Optional[Dict[str, Any]] = None,
    prediction_metadata: Optional[Dict[str, Any]] = None,
    tool_registry: Any = None,
) -> Dict[str, Any]:
    """构造 action_node 所需的最小 state。"""
    return {
        "tracker": tracker,
        "domain": None,
        "metadata": metadata or {},
        "current_prediction": SimpleNamespace(
            action=action_name,
            metadata=prediction_metadata or {},
        ),
        "final_responses": [],
        "action_count": 0,
        "node_history": [],
        "_command_generator": None,
        "_tool_registry": tool_registry,
    }


# ---------- _execute_action 单元测试 ----------

def test_execute_action_mcp_route() -> None:
    """测试 1：_execute_action MCP 路由存在 → MCPExecutable。"""
    print("[测试 1] _execute_action MCP 路由")
    mcp_result = {
        "content": [{"type": "responses", "data": [{"text": "MCP 结果"}]}],
        "isError": False,
    }
    stub = _StubMCPClient(result=mcp_result)
    registry = ToolRegistry(
        mcp_client=stub,  # type: ignore
        mcp_mapping={"action_ask_order_id": "ecommerce__query_order"},
    )
    tracker = _MockTracker(slots={"user_id": "1001"})
    result = asyncio.run(_execute_action("action_ask_order_id", tracker, None, {}, registry))
    if result is None:
        _fail("MCP 路径应返回 ActionResult，不应为 None")
    if not result.responses or result.responses[0].get("text") != "MCP 结果":
        _fail(f"MCP 结果不对: {result.responses if result else None}")
    if stub.call_count != 1:
        _fail(f"应调一次 MCP call_tool，实际 {stub.call_count}")
    if stub.last_args != {"user_id": "1001"}:
        _fail(f"args 应取自 tracker slots: {stub.last_args}")
    _ok("MCP 路由 → MCPExecutable.call_tool，args 取自 tracker slots")


def test_execute_action_local() -> None:
    """测试 2：_execute_action 无路由 / MCP 关闭 → 本地 get_action 直调。"""
    print("[测试 2] _execute_action 本地直调")
    tracker = _MockTracker()
    # tool_registry=None（MCP 关闭）+ action_send_text（内置，不在映射表）
    result = asyncio.run(_execute_action("action_send_text", tracker, None, {"text": "你好"}, None))
    if result is None:
        _fail("本地路径应返回 ActionResult")
    if not result.responses or result.responses[0].get("text") != "你好":
        _fail(f"本地结果不对: {result.responses if result else None}")
    _ok("无路由 / MCP 关闭 → get_action 本地直调（基线等价）")

    # 有 tool_registry 但 action 不在映射表 → 仍走本地
    registry = ToolRegistry(mcp_client=_StubMCPClient(), mcp_mapping={"action_ask_order_id": "ecommerce__query_order"})
    result2 = asyncio.run(_execute_action("action_send_text", tracker, None, {"text": "hi"}, registry))
    if result2 is None or not result2.responses:
        _fail("不在映射表的 action 应走本地")
    _ok("映射表外的 action 即使 MCP 启用也走本地直调")


def test_execute_action_not_found() -> None:
    """测试 3：_execute_action 本地未找到 → None。"""
    print("[测试 3] _execute_action 未找到返回 None")
    tracker = _MockTracker()
    result = asyncio.run(_execute_action("action_not_exists_xyz", tracker, None, {}, None))
    if result is not None:
        _fail(f"未找到动作应返回 None，实际 {result}")
    _ok("本地未找到 → 返回 None（由调用方按基线 not-found 处理）")


# ---------- action_node 集成测试 ----------

def test_action_node_mcp_routed() -> None:
    """测试 4：action_node MCP 路由动作 → MCP 结果 + slot_sets apply + bot 消息。"""
    print("[测试 4] action_node MCP 路由动作")
    mcp_result = {
        "content": [
            {"type": "responses", "data": [{"text": "请选择订单"}]},
            {"type": "slot_sets", "data": {"order_id": "false"}},
        ],
        "isError": False,
    }
    stub = _StubMCPClient(result=mcp_result)
    registry = ToolRegistry(
        mcp_client=stub,  # type: ignore
        mcp_mapping={"action_ask_order_id": "ecommerce__query_order"},
    )
    tracker = _MockTracker(slots={"user_id": "1001"})
    state = _make_state("action_ask_order_id", tracker, tool_registry=registry)
    update = asyncio.run(action_node(state))

    # 响应累积
    final = update.get("final_responses", [])
    if not final or final[0].get("text") != "请选择订单":
        _fail(f"final_responses 不对: {final}")
    # slot_sets apply 到 tracker
    if tracker.get_slot("order_id") != "false":
        _fail(f"slot_sets 未 apply: order_id={tracker.get_slot('order_id')}")
    # bot 消息
    if not tracker.bot_messages:
        _fail("应添加 bot 消息到 tracker")
    # latest_action_name
    if tracker.latest_action_name != "action_ask_order_id":
        _fail(f"latest_action_name 不对: {tracker.latest_action_name}")
    # action_count +1
    if update.get("action_count") != 1:
        _fail(f"action_count 应为 1: {update.get('action_count')}")
    # MCP 仅调一次
    if stub.call_count != 1:
        _fail(f"MCP 应调一次: {stub.call_count}")
    _ok("MCP 路由动作：responses/slot_sets/bot 消息/latest_action_name 全部正确")


def test_action_node_mcp_off_baseline() -> None:
    """测试 5：action_node MCP 关闭（无 _tool_registry）→ 本地直调（基线等价）。"""
    print("[测试 5] action_node MCP 关闭基线")
    tracker = _MockTracker()
    state = _make_state(
        "action_send_text",
        tracker,
        metadata={"text": "基线你好"},
        tool_registry=None,  # MCP 关闭
    )
    update = asyncio.run(action_node(state))
    final = update.get("final_responses", [])
    if not final or final[0].get("text") != "基线你好":
        _fail(f"基线响应不对: {final}")
    if tracker.latest_action_name != "action_send_text":
        _fail(f"latest_action_name 不对: {tracker.latest_action_name}")
    _ok("MCP 关闭 → 本地直调 action_send_text（与基线行为等价）")


def test_action_node_not_found() -> None:
    """测试 6：action_node 未找到动作 → 基线 not-found 失败（无响应累积）。"""
    print("[测试 6] action_node 未找到动作")
    tracker = _MockTracker()
    state = _make_state("action_not_exists_xyz", tracker, tool_registry=None)
    update = asyncio.run(action_node(state))

    if update.get("action_count") != 1:
        _fail(f"action_count 应 +1: {update.get('action_count')}")
    # 不应累积响应
    if update.get("final_responses"):
        _fail(f"未找到动作不应累积响应: {update.get('final_responses')}")
    # current_action_result.success=False
    car = update.get("current_action_result")
    if car is None or car.success:
        _fail(f"未找到应 success=False: {car}")
    # 不应添加 bot 消息
    if tracker.bot_messages:
        _fail("未找到动作不应添加 bot 消息")
    _ok("未找到动作 → 基线 not-found 失败（无响应 / 无 bot 消息 / success=False）")


def test_action_node_utter_fallback() -> None:
    """测试 7：utter 无响应 → fallback 走统一执行入口。"""
    print("[测试 7] action_node utter fallback")
    # 注册测试用空响应 utter 动作
    register_action(_EmptyUtterAction())
    tracker = _MockTracker()
    state = _make_state(
        "utter_empty_test",
        tracker,
        metadata={"text": "fallback你好"},
        prediction_metadata={"fallback_action": "action_send_text"},
        tool_registry=None,
    )
    update = asyncio.run(action_node(state))
    final = update.get("final_responses", [])
    # fallback 的响应应被累积
    if not final or final[0].get("text") != "fallback你好":
        _fail(f"fallback 响应不对: {final}")
    # action_name 应更新为 fallback_action
    if tracker.latest_action_name != "action_send_text":
        _fail(f"latest_action_name 应为 fallback action: {tracker.latest_action_name}")
    _ok("utter 无响应 → fallback 走 _execute_action(action_send_text)，响应累积 + action_name 更新")


def main() -> int:
    print("=" * 60)
    print("M1.8 action_node 集成 ToolRegistry 统一执行入口 单元测试")
    print("=" * 60)
    print()
    tests = [
        test_execute_action_mcp_route,
        test_execute_action_local,
        test_execute_action_not_found,
        test_action_node_mcp_routed,
        test_action_node_mcp_off_baseline,
        test_action_node_not_found,
        test_action_node_utter_fallback,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M1.8 action_node 集成就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

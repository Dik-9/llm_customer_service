# -*- coding: utf-8 -*-
"""
M2.8 验证脚本：节点 hook 注入 + prompt memory_context 渲染

确定性验证（注入 fake memory_hooks + monkeypatch 节点依赖）：
1. understand_node 调用 before_understand hook → tracker.memory_context 被写入
2. understand_node hooks=None → no-op（memory_context 不被设置）
3. understand_node hook 抛错 → 降级不阻断主链路
4. action_node 调用 after_each_action hook（monkeypatch _execute_action）
5. action_node hooks=None → no-op
6. handle_message 调用 before_save_tracker hook（mock graph.ainvoke）
7. handle_message hooks=None → no-op
8. prompt 渲染 memory_context（画像/提及/摘要/消歧）
9. prompt 无 memory_context → 占位"无长期记忆"
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- Fakes ----------

class _FakeMsg:
    def __init__(self, text: str = "") -> None:
        self.text = text


class _FakeStack:
    def top(self):
        return None


class _FakeTracker:
    """轻量 tracker，兼容 understand_node / action_node / hook 访问。"""

    def __init__(self, sender_id: str = "u1", latest: str = "你好") -> None:
        self.sender_id = sender_id
        self.latest_message = _FakeMsg(latest)
        self.latest_action_name: Optional[str] = None
        self.dialogue_stack = _FakeStack()
        self.active_flow = None
        self._slots: Dict[str, Any] = {"user_id": sender_id}
        self.memory_context: Optional[Dict[str, Any]] = None
        self._bot_messages: List[Any] = []

    def update_with_message(self, msg: Any) -> None:
        # 模拟真实 tracker：更新 latest_message
        self.latest_message = _FakeMsg(getattr(msg, "text", ""))

    def get_slot(self, name: str) -> Any:
        return self._slots.get(name)

    def set_slot(self, name: str, value: Any) -> None:
        self._slots[name] = value

    def add_bot_message(self, msg: Any) -> None:
        self._bot_messages.append(msg)

    def get_messages_for_llm(self, max_turns: int = 10) -> List[Dict[str, str]]:
        return []


class _FakeHooks:
    """记录 hook 调用，用于验证注入。"""

    def __init__(self, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None) -> None:
        self.enabled = True
        self._ctx = ctx or {
            "profile_text": "偏好：快递=顺丰",
            "mentions_text": "最近提及订单：O001",
            "session_summary": "用户查询了订单O001",
            "resolved_order_id": "O001",
        }
        self._exc = exc
        self.before_understand_called = False
        self.after_each_action_called = False
        self.before_save_tracker_called = False

    async def before_understand(self, tracker: Any) -> Dict[str, Any]:
        self.before_understand_called = True
        if self._exc:
            raise self._exc
        tracker.memory_context = dict(self._ctx)
        return dict(self._ctx)

    async def after_each_action(self, tracker: Any, action_result: Any = None) -> List[Any]:
        self.after_each_action_called = True
        return []

    async def before_save_tracker(self, tracker: Any) -> Dict[str, Any]:
        self.before_save_tracker_called = True
        return {"compression": None, "end_of_session_facts": [], "session_ended": False}


class _FakeTrackerStore:
    def __init__(self) -> None:
        self.saved: List[Any] = []
        self._tracker = _FakeTracker()

    def set_domain(self, domain: Any) -> None:
        pass

    async def get_or_create_tracker(self, sender_id: str) -> Any:
        self._tracker.sender_id = sender_id
        return self._tracker

    async def save(self, tracker: Any) -> None:
        self.saved.append(tracker)


# ---------- 测试 ----------

def test_understand_node_calls_before_understand() -> None:
    print("[测试 1] understand_node 调用 before_understand hook → memory_context 被写入")
    from atguigu_ai.agent.graph.nodes.understand import understand_node

    tracker = _FakeTracker(latest="改下上次那个订单的地址")
    hooks = _FakeHooks()
    state = {
        "tracker": tracker,
        "input_message": "改下上次那个订单的地址",
        "metadata": {},
        "domain": None,
        "flows": None,
        "_command_generator": None,  # 避免 LLM 调用
        "_command_processor": None,
        "_memory_hooks": hooks,
        "node_history": [],
    }
    asyncio.run(understand_node(state))
    if not hooks.before_understand_called:
        _fail("before_understand 未被调用")
    if tracker.memory_context is None:
        _fail("tracker.memory_context 未被写入")
    if tracker.memory_context.get("resolved_order_id") != "O001":
        _fail(f"memory_context.resolved_order_id 应为 O001，实际 {tracker.memory_context.get('resolved_order_id')}")
    _ok("understand_node 调用 before_understand，memory_context 写入成功")


def test_understand_node_hooks_none_noop() -> None:
    print("[测试 2] understand_node hooks=None → no-op（基线等价）")
    from atguigu_ai.agent.graph.nodes.understand import understand_node

    tracker = _FakeTracker(latest="你好")
    state = {
        "tracker": tracker,
        "input_message": "你好",
        "metadata": {},
        "domain": None,
        "flows": None,
        "_command_generator": None,
        "_command_processor": None,
        "_memory_hooks": None,  # 基线
        "node_history": [],
    }
    asyncio.run(understand_node(state))
    if tracker.memory_context is not None:
        _fail(f"hooks=None 时 memory_context 应保持 None，实际 {tracker.memory_context}")
    _ok("hooks=None → memory_context 不被设置（基线等价）")


def test_understand_node_hook_exception_degrades() -> None:
    print("[测试 3] understand_node hook 抛错 → 降级不阻断主链路")
    from atguigu_ai.agent.graph.nodes.understand import understand_node

    tracker = _FakeTracker(latest="你好")
    hooks = _FakeHooks(exc=RuntimeError("neo4j down"))
    state = {
        "tracker": tracker,
        "input_message": "你好",
        "metadata": {},
        "domain": None,
        "flows": None,
        "_command_generator": None,
        "_command_processor": None,
        "_memory_hooks": hooks,
        "node_history": [],
    }
    # 不应抛错
    try:
        result = asyncio.run(understand_node(state))
    except Exception as e:
        _fail(f"hook 异常不应抛出，实际抛出: {e}")
        return
    if "error" in result and result["error"]:
        _fail(f"hook 异常不应产生 error，实际 error={result['error']}")
    _ok("hook 抛错 → 降级，understand_node 正常返回")


def test_action_node_calls_after_each_action() -> None:
    print("[测试 4] action_node 调用 after_each_action hook")
    from atguigu_ai.agent.graph.nodes import action as action_mod
    from atguigu_ai.agent.actions import ActionResult
    from atguigu_ai.policies.base_policy import PolicyPrediction

    tracker = _FakeTracker(latest="记住我默认用顺丰")
    hooks = _FakeHooks()
    state = {
        "tracker": tracker,
        "domain": None,
        "metadata": {},
        "current_prediction": PolicyPrediction(action="action_listen"),
        "final_responses": [],
        "action_count": 0,
        "_command_generator": None,
        "_tool_registry": None,
        "_memory_hooks": hooks,
        "node_history": [],
    }
    fake_result = ActionResult(responses=[{"text": "好的"}], success=True)
    with patch.object(action_mod, "_execute_action", new=AsyncMock(return_value=fake_result)):
        asyncio.run(action_mod.action_node(state))
    if not hooks.after_each_action_called:
        _fail("after_each_action 未被调用")
    _ok("action_node 调用 after_each_action hook")


def test_action_node_hooks_none_noop() -> None:
    print("[测试 5] action_node hooks=None → no-op")
    from atguigu_ai.agent.graph.nodes import action as action_mod
    from atguigu_ai.agent.actions import ActionResult
    from atguigu_ai.policies.base_policy import PolicyPrediction

    tracker = _FakeTracker(latest="你好")
    hooks = _FakeHooks()
    state = {
        "tracker": tracker,
        "domain": None,
        "metadata": {},
        "current_prediction": PolicyPrediction(action="action_listen"),
        "final_responses": [],
        "action_count": 0,
        "_command_generator": None,
        "_tool_registry": None,
        "_memory_hooks": None,  # 基线
        "node_history": [],
    }
    fake_result = ActionResult(responses=[{"text": "好的"}], success=True)
    with patch.object(action_mod, "_execute_action", new=AsyncMock(return_value=fake_result)):
        asyncio.run(action_mod.action_node(state))
    if hooks.after_each_action_called:
        _fail("hooks=None 时 after_each_action 不应被调用")
    _ok("hooks=None → after_each_action 不调用（基线等价）")


def test_handle_message_calls_before_save_tracker() -> None:
    print("[测试 6] handle_message 调用 before_save_tracker hook")
    from atguigu_ai.agent.agent import Agent
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.flow import FlowsList

    tracker = _FakeTracker(latest="再见")
    hooks = _FakeHooks()
    store = _FakeTrackerStore()
    agent = Agent(
        domain=Domain(),
        flows=FlowsList(),
        tracker_store=store,
        memory_hooks=hooks,
    )
    final_state = {
        "tracker": tracker,
        "final_responses": [{"text": "再见"}],
        "node_history": ["understand", "policy", "action", "response"],
        "error": None,
    }
    with patch.object(agent, "graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=final_state)
        asyncio.run(agent.handle_message("再见", sender_id="u1"))
    if not hooks.before_save_tracker_called:
        _fail("before_save_tracker 未被调用")
    if not store.saved:
        _fail("tracker 未被 save")
    _ok("handle_message 调用 before_save_tracker hook，且 tracker 正常 save")


def test_handle_message_hooks_none_noop() -> None:
    print("[测试 7] handle_message hooks=None → no-op（基线等价）")
    from atguigu_ai.agent.agent import Agent
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.flow import FlowsList

    tracker = _FakeTracker(latest="你好")
    store = _FakeTrackerStore()
    agent = Agent(
        domain=Domain(),
        flows=FlowsList(),
        tracker_store=store,
        memory_hooks=None,  # 基线
    )
    final_state = {
        "tracker": tracker,
        "final_responses": [{"text": "你好"}],
        "node_history": ["understand", "policy", "action", "response"],
        "error": None,
    }
    with patch.object(agent, "graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=final_state)
        try:
            asyncio.run(agent.handle_message("你好", sender_id="u1"))
        except Exception as e:
            _fail(f"hooks=None 时 handle_message 不应抛错: {e}")
            return
    if not store.saved:
        _fail("hooks=None 时 tracker 仍应被 save")
    _ok("hooks=None → before_save_tracker no-op，tracker 正常 save（基线等价）")


def test_prompt_renders_memory_context() -> None:
    print("[测试 8] prompt 渲染 memory_context（画像/提及/摘要/消歧）")
    from atguigu_ai.dialogue_understanding.generator.prompt_builder import PromptBuilder

    tracker = _FakeTracker(latest="改下那个订单")
    tracker.memory_context = {
        "profile_text": "偏好：快递=顺丰",
        "mentions_text": "最近提及订单：O001",
        "session_summary": "用户查询了订单O001",
        "resolved_order_id": "O001",
    }
    builder = PromptBuilder()
    prompt = builder.build_prompt(tracker, None, None)
    if "偏好：快递=顺丰" not in prompt:
        _fail("prompt 缺少画像文本")
    if "最近提及订单：O001" not in prompt:
        _fail("prompt 缺少提及文本")
    if "用户查询了订单O001" not in prompt:
        _fail("prompt 缺少摘要文本")
    if "指代消歧" not in prompt or "O001" not in prompt:
        _fail("prompt 缺少指代消歧")
    _ok("prompt 渲染画像/提及/摘要/消歧全部命中")


def test_prompt_no_memory_context_placeholder() -> None:
    print("[测试 9] prompt 无 memory_context → 占位'无长期记忆'")
    from atguigu_ai.dialogue_understanding.generator.prompt_builder import PromptBuilder

    tracker = _FakeTracker(latest="你好")
    # memory_context 保持 None
    builder = PromptBuilder()
    prompt = builder.build_prompt(tracker, None, None)
    if "无长期记忆" not in prompt:
        _fail("prompt 缺少'无长期记忆'占位")
    if "用户画像" in prompt:
        _fail("无 memory_context 时不应渲染画像块")
    _ok("无 memory_context → 占位'无长期记忆'（基线等价）")


# ---------- 运行器 ----------

def main() -> None:
    tests = [
        test_understand_node_calls_before_understand,
        test_understand_node_hooks_none_noop,
        test_understand_node_hook_exception_degrades,
        test_action_node_calls_after_each_action,
        test_action_node_hooks_none_noop,
        test_handle_message_calls_before_save_tracker,
        test_handle_message_hooks_none_noop,
        test_prompt_renders_memory_context,
        test_prompt_no_memory_context_placeholder,
    ]
    print(f"\n=== M2.8 节点 hook 注入 + prompt 渲染测试（共 {len(tests)} 项）===\n")
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print()
        except Exception:
            failed += 1
            import traceback
            traceback.print_exc()
            print()
    print(f"=== 结果: {passed} 通过 / {failed} 失败 / 共 {len(tests)} ===")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

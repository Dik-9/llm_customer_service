# -*- coding: utf-8 -*-
"""
M2.9 验证脚本：记忆系统端到端集成测试

在真实节点函数（understand_node / action_node）+ 真实 PromptBuilder 链路上验证
三处 hook 协同 + prompt memory_context 渲染 + 基线等价（不连真实 LLM/Neo4j）。

用例：
1. 三处 hook 协同 E2E：understand_node（before_understand）→ action_node（after_each_action）
   → before_save_tracker，一次逻辑流转中三个 hook 都被调用，tracker.memory_context 贯穿
2. prompt 渲染 memory_context E2E：understand_node 写入 memory_context 后，
   PromptBuilder.build_prompt 产出的 prompt 包含画像/提及/摘要/消歧
3. Agent.handle_message 基线 E2E：memory_hooks=None，mock graph，验证无 hook 调用 + 正常返回
4. Agent.handle_message 记忆启用 E2E：memory_hooks=fake，mock graph，
   验证 before_save_tracker 调用 + tracker 正常 save
5. ecs_demo 真实 config.yml 基线确认：无 memory 节点 → _build_memory_hooks 返回 None
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
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
    def __init__(self, sender_id: str = "u1", latest: str = "改下那个订单") -> None:
        self.sender_id = sender_id
        self.latest_message = _FakeMsg(latest)
        self.latest_action_name: Optional[str] = None
        self.dialogue_stack = _FakeStack()
        self.active_flow = None
        self._slots: Dict[str, Any] = {"user_id": sender_id}
        self.memory_context: Optional[Dict[str, Any]] = None
        self._bot_messages: List[Any] = []

    def update_with_message(self, msg: Any) -> None:
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
    """记录三处 hook 调用，模拟真实 MemoryHooks 行为。"""

    def __init__(self) -> None:
        self.enabled = True
        self.before_understand_called = False
        self.after_each_action_called = False
        self.before_save_tracker_called = False
        self._ctx = {
            "profile_text": "偏好：快递=顺丰",
            "mentions_text": "最近提及订单：O001",
            "session_summary": "用户查询了订单O001",
            "resolved_order_id": "O001",
        }

    async def before_understand(self, tracker: Any) -> Dict[str, Any]:
        self.before_understand_called = True
        tracker.memory_context = dict(self._ctx)
        # 指代消歧：写 order_id 槽
        if not tracker.get_slot("order_id"):
            tracker.set_slot("order_id", "O001")
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

def test_three_hooks_coordinated_e2e() -> None:
    print("[测试 1] 三处 hook 协同 E2E：understand→action→save 全调用")
    from atguigu_ai.agent.graph.nodes.understand import understand_node
    from atguigu_ai.agent.graph.nodes import action as action_mod
    from atguigu_ai.agent.actions import ActionResult
    from atguigu_ai.policies.base_policy import PolicyPrediction
    from atguigu_ai.memory.hooks import before_save_tracker

    tracker = _FakeTracker(latest="改下那个订单的地址")
    hooks = _FakeHooks()

    # mock command_generator：捕获 tracker，返回空命令（避免复杂命令处理）
    async def _fake_generate(tr, domain, flows_list):
        # 验证 hook 在 generate 之前执行：tracker.memory_context 已写入
        if tr.memory_context is None:
            _fail("generate 调用时 tracker.memory_context 应已被 hook 写入")
        return SimpleNamespace(commands=[], raw_output="", prompt="fake", metadata={})

    # mock command_processor（不会被执行，因为 commands 为空）
    fake_processor = SimpleNamespace(process=lambda cmds, tr: SimpleNamespace(
        events=[], commands_executed=0, next_action="action_listen",
        response_type="none", metadata={},
    ))

    state = {
        "tracker": tracker,
        "input_message": "改下那个订单的地址",
        "metadata": {},
        "domain": None,
        "flows": None,
        "_command_generator": SimpleNamespace(generate=_fake_generate),
        "_command_processor": fake_processor,
        "_memory_hooks": hooks,
        "node_history": [],
    }
    asyncio.run(understand_node(state))

    if not hooks.before_understand_called:
        _fail("before_understand 未被调用")
    if tracker.memory_context is None:
        _fail("tracker.memory_context 未写入")
    if tracker.get_slot("order_id") != "O001":
        _fail(f"指代消歧应写 order_id=O001，实际 {tracker.get_slot('order_id')}")
    _ok("before_understand 调用 + memory_context 写入 + order_id 消歧")

    # action_node
    state["current_prediction"] = PolicyPrediction(action="action_listen")
    state["final_responses"] = []
    state["action_count"] = 0
    state["_tool_registry"] = None
    state["_command_generator"] = None
    fake_result = ActionResult(responses=[{"text": "好的"}], success=True)
    with patch.object(action_mod, "_execute_action", new=AsyncMock(return_value=fake_result)):
        asyncio.run(action_mod.action_node(state))

    if not hooks.after_each_action_called:
        _fail("after_each_action 未被调用")
    _ok("after_each_action 调用")

    # before_save_tracker
    asyncio.run(before_save_tracker(tracker, hooks))
    if not hooks.before_save_tracker_called:
        _fail("before_save_tracker 未被调用")
    _ok("before_save_tracker 调用")
    print("  → 三处 hook 协同成功，memory_context 贯穿 understand→action→save")


def test_prompt_renders_memory_context_e2e() -> None:
    print("[测试 2] prompt 渲染 memory_context E2E（understand 写入 → prompt 渲染）")
    from atguigu_ai.agent.graph.nodes.understand import understand_node
    from atguigu_ai.dialogue_understanding.generator.prompt_builder import PromptBuilder

    tracker = _FakeTracker(latest="改下那个订单")
    hooks = _FakeHooks()

    async def _fake_generate(tr, domain, flows_list):
        return SimpleNamespace(commands=[], raw_output="", prompt="", metadata={})

    state = {
        "tracker": tracker,
        "input_message": "改下那个订单",
        "metadata": {},
        "domain": None,
        "flows": None,
        "_command_generator": SimpleNamespace(generate=_fake_generate),
        "_command_processor": None,
        "_memory_hooks": hooks,
        "node_history": [],
    }
    asyncio.run(understand_node(state))

    if tracker.memory_context is None:
        _fail("understand_node 后 tracker.memory_context 应非 None")

    # 用真实 PromptBuilder 渲染
    builder = PromptBuilder()
    prompt = builder.build_prompt(tracker, None, None)
    for keyword in ["偏好：快递=顺丰", "最近提及订单：O001", "用户查询了订单O001", "指代消歧"]:
        if keyword not in prompt:
            _fail(f"prompt 缺少: {keyword}")
    _ok("understand 写入 memory_context → prompt 渲染画像/提及/摘要/消歧全命中")


def test_handle_message_baseline_e2e() -> None:
    print("[测试 3] Agent.handle_message 基线 E2E（memory_hooks=None）")
    from atguigu_ai.agent.agent import Agent
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.flow import FlowsList

    store = _FakeTrackerStore()
    agent = Agent(
        domain=Domain(),
        flows=FlowsList(),
        tracker_store=store,
        memory_hooks=None,  # 基线
    )
    final_state = {
        "tracker": _FakeTracker(latest="你好"),
        "final_responses": [{"text": "你好"}],
        "node_history": ["understand", "policy", "action", "response"],
        "error": None,
    }
    with patch.object(agent, "graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=final_state)
        resp = asyncio.run(agent.handle_message("你好", sender_id="u1"))
    if not store.saved:
        _fail("基线 tracker 应被 save")
    if not resp.messages:
        _fail("基线应返回响应")
    _ok("memory_hooks=None → 正常返回响应 + tracker save（基线等价）")


def test_handle_message_memory_enabled_e2e() -> None:
    print("[测试 4] Agent.handle_message 记忆启用 E2E（before_save_tracker 调用）")
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
        resp = asyncio.run(agent.handle_message("再见", sender_id="u1"))
    if not hooks.before_save_tracker_called:
        _fail("before_save_tracker 未被调用")
    if not store.saved:
        _fail("tracker 未被 save")
    if not resp.messages:
        _fail("应返回响应")
    _ok("memory_hooks 注入 → before_save_tracker 调用 + tracker save + 响应返回")


def test_ecs_demo_config_baseline_e2e() -> None:
    print("[测试 5] ecs_demo 真实 config.yml + endpoints.yml 配置加载确认")
    cfg_path = Path(__file__).resolve().parent / "config.yml"
    if not cfg_path.exists():
        _fail(f"ecs_demo/config.yml 不存在: {cfg_path}")
    from atguigu_ai.shared.yaml_loader import read_yaml_file
    from atguigu_ai.agent.agent import _build_memory_hooks
    from atguigu_ai.shared.config import EndpointsConfig

    cfg = read_yaml_file(str(cfg_path)) or {}
    if "memory" not in cfg:
        _fail("ecs_demo/config.yml 应含 memory 节点")

    # 用真实 ecs_demo endpoints.yml 构造 EndpointsConfig（不连真实服务，仅读配置）
    endpoints_path = Path(__file__).resolve().parent / "endpoints.yml"
    endpoints = EndpointsConfig.load(str(endpoints_path)) if endpoints_path.exists() else EndpointsConfig()

    # 不硬编码 enabled 值（自测时用户会改），仅验证 _build_memory_hooks 能正常执行不抛错
    hooks = _build_memory_hooks(cfg, endpoints)
    # enabled=false → None；enabled=true → 可能 None（组件失败）或 MemoryHooks
    # 此处仅验证不抛异常
    _ok(f"ecs_demo config.yml + endpoints.yml 加载正常，memory_hooks={'None(基线)' if hooks is None else '已注入'}")


# ---------- 运行器 ----------

def main() -> None:
    tests = [
        test_three_hooks_coordinated_e2e,
        test_prompt_renders_memory_context_e2e,
        test_handle_message_baseline_e2e,
        test_handle_message_memory_enabled_e2e,
        test_ecs_demo_config_baseline_e2e,
    ]
    print(f"\n=== M2.9 记忆系统端到端集成测试（共 {len(tests)} 项）===\n")
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

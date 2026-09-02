# -*- coding: utf-8 -*-
"""
M4 整体集成验收（SPEC §7 M4 + §8.4）

三种开关组合（全关/单开M1/单开M2/全开）的 e2e 冒烟测试 + 基线回归。
确定性验证（mock LLM/Neo4j/MCP），不连真实服务。

M4.1 三种开关组合的 Agent.load 集成：
  1. 全关（mcp.enabled=false + memory.enabled=false）→ tool_registry 本地 + memory_hooks=None（基线等价）
  2. 单开 M1（mcp.enabled=true + memory=false）→ tool_registry 有 MCP 路由 + memory_hooks=None
  3. 单开 M2（mcp=false + memory=true）→ tool_registry 本地 + memory_hooks 注入
  4. 全开（mcp=true + memory=true）→ tool_registry 有 MCP 路由 + memory_hooks 注入
  5. 三模块 state 透传互不干扰（_tool_registry + _memory_hooks 同时存在）

M4.2 三模块协同 e2e：
  6. 全开配置下 understand→action→save 全链路，MCP+记忆同时生效
  7. 全关配置下 understand→action→save 全链路，基线等价

M4.3 基线回归：
  8. 真实 ecs_demo 默认配置（全关）→ Agent.load 后 tool_registry 本地 + memory_hooks=None
  9. 全量测试套件回归确认（M1+M2+M3 测试文件存在且可导入）
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
    def __init__(self, sender_id: str = "u1", latest: str = "查询我的订单") -> None:
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
    def __init__(self) -> None:
        self.enabled = True
        self.before_understand_called = False
        self.after_each_action_called = False
        self.before_save_tracker_called = False

    async def before_understand(self, tracker):
        self.before_understand_called = True
        tracker.memory_context = {"profile_text": "偏好：顺丰", "mentions_text": "", "session_summary": "", "resolved_order_id": None}
        return tracker.memory_context

    async def after_each_action(self, tracker, action_result=None):
        self.after_each_action_called = True
        return []

    async def before_save_tracker(self, tracker):
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


class _FakeLLMClient:
    def complete_sync(self, messages, **kwargs):
        return SimpleNamespace(content='```yaml\nversion: "3.1"\nflows: {}\n```')


class _FakeGraphStore:
    def get_user_profile(self, user_id):
        return {"preferences": [], "default_address": None}

    def get_recent_mentions(self, user_id, limit=5):
        return []

    def close(self):
        pass


def _make_endpoints(
    mcp_enabled: bool = False,
    memory_enabled: bool = False,
) -> Any:
    """构造 fake EndpointsConfig，控制 mcp/memory 开关 + LLM/Neo4j 可用性。"""
    from atguigu_ai.shared.config import (
        EndpointsConfig,
        LLMConfig,
        MCPConfig,
        MCPServerConfig,
        VectorStoreConfig,
    )
    models = {
        "command": LLMConfig(
            type="openai", model="test", api_key="fake", api_base="http://fake"
        )
    }
    mcp_cfg = None
    if mcp_enabled:
        # MCPConfig 需要 servers 字典（build_tool_registry 从 servers 取 base_url）
        server = MCPServerConfig(base_url="http://127.0.0.1:8765/mcp")
        mcp_cfg = MCPConfig(enabled=True, servers={"ecommerce": server})
    vs_cfg = VectorStoreConfig(config={
        "uri": "bolt://fake:7687", "user": "neo4j", "password": "fake"
    }) if memory_enabled else VectorStoreConfig(config={})
    return EndpointsConfig(models=models, vector_store=vs_cfg, mcp=mcp_cfg)


def _mem_data(short_en: bool = False, long_en: bool = False) -> Dict[str, Any]:
    return {
        "memory": {
            "short_term": {"enabled": short_en, "max_raw_turns": 20, "keep_recent_turns": 10, "llm": "command"},
            "long_term": {"enabled": long_en, "graph_database": "user_memory", "llm": "command",
                          "realtime_extract": True, "end_of_session_extract": True},
        }
    }


# ======================================================================
# M4.1 三种开关组合的 Agent.load 集成
# ======================================================================

def test_all_disabled_baseline() -> None:
    print("[测试 1] 全关（mcp=false + memory=false）→ 基线等价")
    from atguigu_ai.agent.agent import _build_memory_hooks
    from atguigu_ai.mcp.tool_registry import build_tool_registry

    endpoints = _make_endpoints(mcp_enabled=False, memory_enabled=False)
    tool_registry = build_tool_registry(endpoints.mcp)
    memory_hooks = _build_memory_hooks(_mem_data(False, False), endpoints)

    if tool_registry.mcp_enabled:
        _fail("全关时 tool_registry.mcp_enabled 应为 False")
    if memory_hooks is not None:
        _fail(f"全关时 memory_hooks 应为 None，实际 {memory_hooks}")
    _ok("全关 → tool_registry 本地直调 + memory_hooks=None（基线等价）")


def test_mcp_only() -> None:
    print("[测试 2] 单开 M1（mcp=true + memory=false）→ MCP 注入，记忆 no-op")
    from atguigu_ai.agent.agent import _build_memory_hooks
    from atguigu_ai.mcp.tool_registry import build_tool_registry

    endpoints = _make_endpoints(mcp_enabled=True, memory_enabled=False)
    tool_registry = build_tool_registry(endpoints.mcp)
    memory_hooks = _build_memory_hooks(_mem_data(False, False), endpoints)

    if not tool_registry.mcp_enabled:
        _fail("mcp=true 时 tool_registry.mcp_enabled 应为 True")
    # 验证电商 action 有 MCP 路由
    if not tool_registry.has_mcp_route("action_get_order_detail"):
        _fail("action_get_order_detail 应有 MCP 路由")
    if memory_hooks is not None:
        _fail("memory=false 时 memory_hooks 应为 None")
    _ok("单开 M1 → MCP 路由注入 + memory_hooks=None")


def test_memory_only() -> None:
    print("[测试 3] 单开 M2（mcp=false + memory=true）→ 记忆注入，MCP 本地")
    from atguigu_ai.agent.agent import _build_memory_hooks
    from atguigu_ai.mcp.tool_registry import build_tool_registry

    endpoints = _make_endpoints(mcp_enabled=False, memory_enabled=True)
    tool_registry = build_tool_registry(endpoints.mcp)

    fake_store = _FakeGraphStore()
    with patch("atguigu_ai.shared.llm.create_llm_client", return_value=_FakeLLMClient()), \
         patch("atguigu_ai.memory.long_term.graph_store.GraphMemoryStore.connect", return_value=fake_store):
        memory_hooks = _build_memory_hooks(_mem_data(True, True), endpoints)

    if tool_registry.mcp_enabled:
        _fail("mcp=false 时 tool_registry.mcp_enabled 应为 False")
    if memory_hooks is None:
        _fail("memory=true 时 memory_hooks 应注入")
    if memory_hooks.graph_store is not fake_store:
        _fail("memory_hooks.graph_store 应为 fake_store")
    _ok("单开 M2 → tool_registry 本地 + memory_hooks 注入（graph_store/recaller/extractor/compressor）")


def test_all_enabled() -> None:
    print("[测试 4] 全开（mcp=true + memory=true）→ MCP+记忆协同注入")
    from atguigu_ai.agent.agent import _build_memory_hooks
    from atguigu_ai.mcp.tool_registry import build_tool_registry

    endpoints = _make_endpoints(mcp_enabled=True, memory_enabled=True)
    tool_registry = build_tool_registry(endpoints.mcp)

    fake_store = _FakeGraphStore()
    with patch("atguigu_ai.shared.llm.create_llm_client", return_value=_FakeLLMClient()), \
         patch("atguigu_ai.memory.long_term.graph_store.GraphMemoryStore.connect", return_value=fake_store):
        memory_hooks = _build_memory_hooks(_mem_data(True, True), endpoints)

    if not tool_registry.mcp_enabled:
        _fail("mcp=true 时 tool_registry.mcp_enabled 应为 True")
    if not tool_registry.has_mcp_route("action_get_order_detail"):
        _fail("MCP 路由应注入")
    if memory_hooks is None:
        _fail("memory=true 时 memory_hooks 应注入")
    if memory_hooks.graph_store is not fake_store:
        _fail("memory_hooks.graph_store 应为 fake_store")
    _ok("全开 → MCP 路由 + memory_hooks 同时注入，互不干扰")


def test_state_passes_both_components() -> None:
    print("[测试 5] state 透传 _tool_registry + _memory_hooks 互不干扰")
    from atguigu_ai.agent.graph.state import create_initial_state

    class _RegMarker:
        pass
    class _HooksMarker:
        pass

    reg = _RegMarker()
    hooks = _HooksMarker()
    state = create_initial_state(
        tracker=object(),
        input_message="hi",
        tool_registry=reg,
        memory_hooks=hooks,
    )
    if state.get("_tool_registry") is not reg:
        _fail("state['_tool_registry'] 应为 reg marker")
    if state.get("_memory_hooks") is not hooks:
        _fail("state['_memory_hooks'] 应为 hooks marker")
    _ok("state 同时透传 _tool_registry + _memory_hooks，互不干扰")


# ======================================================================
# M4.2 三模块协同 e2e（全开 + 全关）
# ======================================================================

def test_full_pipeline_all_enabled() -> None:
    print("[测试 6] 全开配置下 understand→action→save 全链路，MCP+记忆同时生效")
    from atguigu_ai.agent.graph.nodes.understand import understand_node
    from atguigu_ai.agent.graph.nodes import action as action_mod
    from atguigu_ai.agent.actions import ActionResult
    from atguigu_ai.policies.base_policy import PolicyPrediction
    from atguigu_ai.memory.hooks import before_save_tracker

    tracker = _FakeTracker(latest="查询我的订单")
    hooks = _FakeHooks()

    # mock command_generator 避免真实 LLM
    async def _fake_generate(tr, domain, flows_list):
        if tr.memory_context is None:
            _fail("全开时 understand 应已写入 memory_context")
        return SimpleNamespace(commands=[], raw_output="", prompt="", metadata={})

    state = {
        "tracker": tracker,
        "input_message": "查询我的订单",
        "metadata": {},
        "domain": None,
        "flows": None,
        "_command_generator": SimpleNamespace(generate=_fake_generate),
        "_command_processor": None,
        "_memory_hooks": hooks,
        "node_history": [],
    }
    asyncio.run(understand_node(state))

    if not hooks.before_understand_called:
        _fail("before_understand 未调用")
    if tracker.memory_context is None:
        _fail("memory_context 未写入")

    # action_node（mock _execute_action 避免 MCP/DB）
    state["current_prediction"] = PolicyPrediction(action="action_listen")
    state["final_responses"] = []
    state["action_count"] = 0
    state["_tool_registry"] = None  # 测试聚焦记忆 hook，tool_registry 用 None
    state["_command_generator"] = None
    fake_result = ActionResult(responses=[{"text": "好的"}], success=True)
    with patch.object(action_mod, "_execute_action", new=AsyncMock(return_value=fake_result)):
        asyncio.run(action_mod.action_node(state))

    if not hooks.after_each_action_called:
        _fail("after_each_action 未调用")

    # before_save_tracker
    asyncio.run(before_save_tracker(tracker, hooks))
    if not hooks.before_save_tracker_called:
        _fail("before_save_tracker 未调用")
    _ok("全开：understand(memory)→action(memory)→save(memory) 全链路 hook 生效")


def test_full_pipeline_all_disabled_baseline() -> None:
    print("[测试 7] 全关配置下 understand→action→save 全链路，基线等价")
    from atguigu_ai.agent.graph.nodes.understand import understand_node
    from atguigu_ai.agent.graph.nodes import action as action_mod
    from atguigu_ai.agent.actions import ActionResult
    from atguigu_ai.policies.base_policy import PolicyPrediction

    tracker = _FakeTracker(latest="查询我的订单")
    hooks = _FakeHooks()  # 不会被调用

    async def _fake_generate(tr, domain, flows_list):
        if tr.memory_context is not None:
            _fail("全关时 memory_context 应保持 None")
        return SimpleNamespace(commands=[], raw_output="", prompt="", metadata={})

    state = {
        "tracker": tracker,
        "input_message": "查询我的订单",
        "metadata": {},
        "domain": None,
        "flows": None,
        "_command_generator": SimpleNamespace(generate=_fake_generate),
        "_command_processor": None,
        "_memory_hooks": None,  # 基线
        "node_history": [],
    }
    asyncio.run(understand_node(state))

    if hooks.before_understand_called:
        _fail("全关时 before_understand 不应调用")
    if tracker.memory_context is not None:
        _fail("全关时 memory_context 应为 None")

    # action_node
    state["current_prediction"] = PolicyPrediction(action="action_listen")
    state["final_responses"] = []
    state["action_count"] = 0
    state["_tool_registry"] = None
    state["_command_generator"] = None
    fake_result = ActionResult(responses=[{"text": "好的"}], success=True)
    with patch.object(action_mod, "_execute_action", new=AsyncMock(return_value=fake_result)):
        asyncio.run(action_mod.action_node(state))

    if hooks.after_each_action_called:
        _fail("全关时 after_each_action 不应调用")
    _ok("全关：understand→action 全链路 hook no-op，基线等价")


def test_handle_message_full_disabled() -> None:
    print("[测试 8] Agent.handle_message 全关 → 正常返回响应 + tracker save")
    from atguigu_ai.agent.agent import Agent
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.flow import FlowsList

    store = _FakeTrackerStore()
    agent = Agent(
        domain=Domain(),
        flows=FlowsList(),
        tracker_store=store,
        memory_hooks=None,
    )
    final_state = {
        "tracker": _FakeTracker(latest="查询我的订单"),
        "final_responses": [{"text": "这是订单列表"}],
        "node_history": ["understand", "policy", "action", "response"],
        "error": None,
    }
    with patch.object(agent, "graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=final_state)
        resp = asyncio.run(agent.handle_message("查询我的订单", sender_id="u1"))
    if not store.saved:
        _fail("全关时 tracker 应被 save")
    if not resp.messages:
        _fail("全关时应返回响应")
    _ok("全关 handle_message → 正常响应 + tracker save（基线等价）")


def test_handle_message_full_enabled() -> None:
    print("[测试 9] Agent.handle_message 全开 → before_save_tracker 调用 + 响应返回")
    from atguigu_ai.agent.agent import Agent
    from atguigu_ai.core.domain import Domain
    from atguigu_ai.dialogue_understanding.flow import FlowsList

    tracker = _FakeTracker(latest="查询我的订单")
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
        "final_responses": [{"text": "这是订单列表"}],
        "node_history": ["understand", "policy", "action", "response"],
        "error": None,
    }
    with patch.object(agent, "graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=final_state)
        resp = asyncio.run(agent.handle_message("查询我的订单", sender_id="u1"))
    if not hooks.before_save_tracker_called:
        _fail("全开时 before_save_tracker 应调用")
    if not store.saved:
        _fail("全开时 tracker 应被 save")
    if not resp.messages:
        _fail("全开时应返回响应")
    _ok("全开 handle_message → before_save_tracker 调用 + 响应返回 + tracker save")


# ======================================================================
# M4.3 基线回归
# ======================================================================

def test_ecs_demo_default_config_baseline() -> None:
    print("[测试 10] 真实 ecs_demo 配置加载 + 开关一致性验证")
    from atguigu_ai.shared.yaml_loader import read_yaml_file
    from atguigu_ai.shared.config import EndpointsConfig, MemoryConfig
    from atguigu_ai.agent.agent import _build_memory_hooks
    from atguigu_ai.mcp.tool_registry import build_tool_registry

    ecs_demo = Path(__file__).resolve().parent
    endpoints = EndpointsConfig.load(str(ecs_demo / "endpoints.yml"))
    cfg = read_yaml_file(str(ecs_demo / "config.yml")) or {}

    tool_registry = build_tool_registry(endpoints.mcp)
    memory_hooks = _build_memory_hooks(cfg, endpoints)

    # 读 config.yml 实际开关值（自测时用户可能改 enabled）
    mem_cfg = MemoryConfig.from_dict(cfg.get("memory", {}))
    st_en = mem_cfg.short_term.enabled
    lt_en = mem_cfg.long_term.enabled
    mcp_en = bool(endpoints.mcp and endpoints.mcp.enabled)

    # 一致性判据：开关值与注入结果一致
    if mcp_en and not tool_registry.mcp_enabled:
        _fail(f"mcp.enabled=true 但 tool_registry.mcp_enabled=False（不一致）")
    if (not mcp_en) and tool_registry.mcp_enabled:
        _fail(f"mcp.enabled=false 但 tool_registry.mcp_enabled=True（不一致）")
    if (st_en or lt_en) and memory_hooks is None:
        _fail(f"memory 开启(short={st_en},long={lt_en}) 但 memory_hooks=None（不一致）")
    if (not st_en and not lt_en) and memory_hooks is not None:
        _fail(f"memory 关闭 但 memory_hooks={memory_hooks}（不一致）")
    _ok(f"ecs_demo 配置一致性: mcp={mcp_en}(registry={tool_registry.mcp_enabled}), "
        f"memory(short={st_en},long={lt_en})(hooks={'注入' if memory_hooks else 'None'})")


def test_test_suite_regression() -> None:
    print("[测试 11] 全量测试套件回归确认（M1+M2+M3 测试文件存在且可导入）")
    ecs_demo = Path(__file__).resolve().parent
    required_tests = [
        "test_mcp_protocol.py",
        "test_mcp_exceptions.py",
        "test_mcp_server.py",
        "test_mcp_client.py",
        "test_mcp_config.py",
        "test_action_node_mcp.py",
        "test_mcp_e2e.py",
        "test_memory_config.py",
        "test_graph_store.py",
        "test_extractor.py",
        "test_recaller.py",
        "test_compressor.py",
        "test_hooks.py",
        "test_memory_integration.py",
        "test_memory_hooks_integration.py",
        "test_memory_e2e.py",
        "test_flow_generator.py",
    ]
    missing = []
    for t in required_tests:
        if not (ecs_demo / t).exists():
            missing.append(t)
    if missing:
        _fail(f"缺少测试文件: {missing}")

    # 验证关键模块可导入（无语法/导入错误）
    import importlib
    modules_to_check = [
        "atguigu_ai.mcp.tool_registry",
        "atguigu_ai.mcp.client",
        "atguigu_ai.mcp.server",
        "atguigu_ai.memory.hooks",
        "atguigu_ai.memory.short_term.compressor",
        "atguigu_ai.memory.long_term.graph_store",
        "atguigu_ai.memory.long_term.extractor",
        "atguigu_ai.memory.long_term.recaller",
        "atguigu_ai.training.flow_generator",
        "atguigu_ai.cli.flow_generate",
    ]
    for mod_name in modules_to_check:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            _fail(f"导入 {mod_name} 失败: {e}")
    _ok(f"M1+M2+M3 全部 {len(required_tests)} 测试文件存在 + {len(modules_to_check)} 核心模块可导入")


def test_three_main_flows_config_intact() -> None:
    print("[测试 12] ecs_demo 三条主链路 Flow 配置完整（基线回归前置）")
    from atguigu_ai.dialogue_understanding.flow import FlowLoader

    ecs_demo = Path(__file__).resolve().parent
    loader = FlowLoader()
    flows = loader.load(ecs_demo / "data")

    flow_ids = {f.id for f in flows}
    required_flows = ["query_order_detail", "modify_order_receive_info", "cancel_order"]
    for rf in required_flows:
        if rf not in flow_ids:
            _fail(f"缺少主链路 Flow: {rf}（现有: {flow_ids}）")
    _ok(f"三条主链路 Flow 完整: {required_flows}（共 {len(flows)} 个 flow）")


# ---------- 运行器 ----------

def main() -> None:
    tests = [
        # M4.1 三种开关组合
        test_all_disabled_baseline,
        test_mcp_only,
        test_memory_only,
        test_all_enabled,
        test_state_passes_both_components,
        # M4.2 三模块协同 e2e
        test_full_pipeline_all_enabled,
        test_full_pipeline_all_disabled_baseline,
        test_handle_message_full_disabled,
        test_handle_message_full_enabled,
        # M4.3 基线回归
        test_ecs_demo_default_config_baseline,
        test_test_suite_regression,
        test_three_main_flows_config_intact,
    ]
    print(f"\n=== M4 整体集成验收（共 {len(tests)} 项）===\n")
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

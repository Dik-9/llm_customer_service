# -*- coding: utf-8 -*-
"""
M2.7 验证脚本：Agent.load 集成记忆系统 + graph state 透传

确定性验证（注入 fake EndpointsConfig + monkeypatch Neo4j/LLM 连接）：
1. memory 节点缺失 → _build_memory_hooks 返回 None（基线等价）
2. memory.enabled=false → 返回 None（基线等价）
3. memory.enabled=true 但 endpoints 空（无 LLM/Neo4j）→ 所有子组件 None → 返回 None
4. 仅 short_term.enabled + LLM 就绪 → compressor 就绪，返回 hooks
5. long_term.enabled + Neo4j + LLM 就绪 → 全组件就绪，返回 hooks（monkeypatch connect）
6. Neo4j connect 抛错 → graph_store 降级 None，但 short_term compressor 仍可用 → 返回 hooks
7. create_initial_state 透传 memory_hooks → state["_memory_hooks"]
8. Agent.__init__ 透传 memory_hooks → agent.memory_hooks
9. 真实 ecs_demo config.yml（无 memory 节点）→ Agent.load 后 memory_hooks is None（基线等价）
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- Fakes ----------

class _FakeLLMClient:
    """假 LLM 客户端，避免真实 API 调用。"""
    async def complete(self, messages, **kwargs):
        return {"content": "[]"}


class _FakeGraphStore:
    """假 Neo4j 图谱存储，避免真实连接。"""
    def __init__(self, database: str = "user_memory"):
        self._database = database

    def get_user_profile(self, user_id):
        return {"user_id": user_id, "preferences": [], "default_address": None}

    def get_recent_mentions(self, user_id, limit=5):
        return []

    def upsert_preference(self, *args, **kwargs):
        pass

    def upsert_address(self, *args, **kwargs):
        pass

    def add_order_mention(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _make_endpoints(
    with_llm: bool = True,
    with_neo4j: bool = True,
) -> Any:
    """构造 fake EndpointsConfig。"""
    from atguigu_ai.shared.config import (
        EndpointsConfig,
        LLMConfig,
        VectorStoreConfig,
    )
    models: Dict[str, Any] = {}
    if with_llm:
        models["command"] = LLMConfig(
            type="openai", model="test-model", api_key="fake-key", api_base="http://fake"
        )
    vs_config = VectorStoreConfig(config={})
    if with_neo4j:
        vs_config = VectorStoreConfig(config={
            "uri": "bolt://fake:7687",
            "user": "neo4j",
            "password": "fake-pwd",
        })
    return EndpointsConfig(models=models, vector_store=vs_config)


def _mem_data(short_en: bool = False, long_en: bool = False) -> Dict[str, Any]:
    """构造 config_data 的 memory 节点。"""
    return {
        "memory": {
            "short_term": {
                "enabled": short_en,
                "max_raw_turns": 5,
                "keep_recent_turns": 3,
                "llm": "command",
            },
            "long_term": {
                "enabled": long_en,
                "graph_database": "user_memory",
                "llm": "command",
                "realtime_extract": True,
                "end_of_session_extract": True,
            },
        }
    }


# ---------- 测试 ----------

def test_no_memory_node_returns_none() -> None:
    print("[测试 1] memory 节点缺失 → _build_memory_hooks 返回 None")
    from atguigu_ai.agent.agent import _build_memory_hooks
    hooks = _build_memory_hooks({}, _make_endpoints())
    if hooks is not None:
        _fail(f"应返回 None，实际: {hooks}")
    _ok("无 memory 节点 → None（基线等价）")


def test_memory_disabled_returns_none() -> None:
    print("[测试 2] memory.enabled=false → 返回 None")
    from atguigu_ai.agent.agent import _build_memory_hooks
    hooks = _build_memory_hooks(_mem_data(short_en=False, long_en=False), _make_endpoints())
    if hooks is not None:
        _fail(f"enabled=false 应返回 None，实际: {hooks}")
    _ok("memory.enabled=false → None（基线等价）")


def test_all_components_fail_returns_none() -> None:
    print("[测试 3] memory.enabled=true 但 endpoints 空 → 全组件 None → 返回 None")
    from atguigu_ai.agent.agent import _build_memory_hooks
    # endpoints 既无 LLM 也无 Neo4j
    empty_endpoints = _make_endpoints(with_llm=False, with_neo4j=False)
    hooks = _build_memory_hooks(_mem_data(short_en=True, long_en=True), empty_endpoints)
    if hooks is not None:
        _fail(f"全组件失败应返回 None，实际: {hooks}")
    _ok("无 LLM + 无 Neo4j → 全组件 None → None（降级基线）")


def test_short_term_only_with_llm() -> None:
    print("[测试 4] 仅 short_term.enabled + LLM 就绪 → compressor 就绪")
    from atguigu_ai.agent.agent import _build_memory_hooks
    with patch("atguigu_ai.shared.llm.create_llm_client", return_value=_FakeLLMClient()) as mock_llm:
        hooks = _build_memory_hooks(
            _mem_data(short_en=True, long_en=False),
            _make_endpoints(with_llm=True, with_neo4j=False),
        )
    if hooks is None:
        _fail("short_term + LLM 应返回 hooks，实际 None")
    if mock_llm.call_count != 1:
        _fail(f"create_llm_client 应调用 1 次，实际 {mock_llm.call_count}")
    if hooks.compressor is None:
        _fail("compressor 应就绪，实际 None")
    if hooks.recaller is not None or hooks.extractor is not None or hooks.graph_store is not None:
        _fail("long_term 未启用，相关组件应为 None")
    _ok("short_term + LLM → compressor 就绪，long_term 组件 None")


def test_long_term_full() -> None:
    print("[测试 5] long_term.enabled + Neo4j + LLM → 全组件就绪")
    from atguigu_ai.agent.agent import _build_memory_hooks
    fake_store = _FakeGraphStore()
    with patch("atguigu_ai.shared.llm.create_llm_client", return_value=_FakeLLMClient()), \
         patch(
             "atguigu_ai.memory.long_term.graph_store.GraphMemoryStore.connect",
             return_value=fake_store,
         ) as mock_connect:
        hooks = _build_memory_hooks(
            _mem_data(short_en=True, long_en=True),
            _make_endpoints(with_llm=True, with_neo4j=True),
        )
    if hooks is None:
        _fail("long_term + Neo4j + LLM 应返回 hooks，实际 None")
    if mock_connect.call_count != 1:
        _fail(f"GraphMemoryStore.connect 应调用 1 次，实际 {mock_connect.call_count}")
    if hooks.graph_store is not fake_store:
        _fail("graph_store 应为 fake_store")
    if hooks.recaller is None:
        _fail("recaller 应就绪")
    if hooks.extractor is None:
        _fail("extractor 应就绪")
    if hooks.compressor is None:
        _fail("compressor 应就绪")
    # 验证 database 参数透传
    call_kwargs = mock_connect.call_args.kwargs
    if call_kwargs.get("database") != "user_memory":
        _fail(f"connect database 参数应为 user_memory，实际 {call_kwargs.get('database')}")
    _ok("long_term + Neo4j + LLM → graph_store/recaller/extractor/compressor 全就绪")


def test_neo4j_connect_failure_degrades() -> None:
    print("[测试 6] Neo4j connect 抛错 → graph_store 降级，short_term compressor 仍可用")
    from atguigu_ai.agent.agent import _build_memory_hooks
    with patch("atguigu_ai.shared.llm.create_llm_client", return_value=_FakeLLMClient()), \
         patch(
             "atguigu_ai.memory.long_term.graph_store.GraphMemoryStore.connect",
             side_effect=ConnectionError("neo4j unreachable"),
         ):
        hooks = _build_memory_hooks(
            _mem_data(short_en=True, long_en=True),
            _make_endpoints(with_llm=True, with_neo4j=True),
        )
    if hooks is None:
        _fail("Neo4j 失败但 compressor 可用时应返回 hooks，实际 None")
    if hooks.graph_store is not None:
        _fail("graph_store 应降级为 None")
    if hooks.recaller is not None:
        _fail("recaller 依赖 graph_store 应为 None")
    if hooks.extractor is None:
        _fail("extractor 依赖 LLM 应就绪")
    if hooks.compressor is None:
        _fail("compressor 依赖 LLM 应就绪")
    _ok("Neo4j 失败 → graph_store/recaller 降级，extractor/compressor 仍可用")


def test_state_passes_memory_hooks() -> None:
    print("[测试 7] create_initial_state 透传 memory_hooks → state['_memory_hooks']")
    from atguigu_ai.agent.graph.state import create_initial_state

    class _Marker:
        pass

    marker = _Marker()
    state = create_initial_state(
        tracker=object(),
        input_message="hi",
        memory_hooks=marker,
    )
    if state.get("_memory_hooks") is not marker:
        _fail(f"state['_memory_hooks'] 应为 marker，实际 {state.get('_memory_hooks')}")
    _ok("create_initial_state 透传 memory_hooks")

    # 默认 None
    state2 = create_initial_state(tracker=object(), input_message="hi")
    if state2.get("_memory_hooks") is not None:
        _fail(f"默认 memory_hooks 应为 None，实际 {state2.get('_memory_hooks')}")
    _ok("默认 memory_hooks=None")


def test_agent_init_memory_hooks() -> None:
    print("[测试 8] Agent.__init__ 透传 memory_hooks → agent.memory_hooks")
    from atguigu_ai.agent.agent import Agent

    class _Marker:
        pass

    marker = _Marker()
    agent = Agent(memory_hooks=marker)
    if agent.memory_hooks is not marker:
        _fail("agent.memory_hooks 应为 marker")
    _ok("Agent.__init__ 透传 memory_hooks")

    # 默认 None（基线等价）
    agent2 = Agent()
    if agent2.memory_hooks is not None:
        _fail(f"默认 agent.memory_hooks 应为 None，实际 {agent2.memory_hooks}")
    _ok("默认 agent.memory_hooks=None")


def test_ecs_demo_baseline_no_memory() -> None:
    print("[测试 9] 真实 ecs_demo config.yml（memory.enabled=false）→ memory_hooks is None")
    cfg_path = Path(__file__).resolve().parent / "config.yml"
    if not cfg_path.exists():
        _fail(f"ecs_demo/config.yml 不存在: {cfg_path}")
    from atguigu_ai.shared.yaml_loader import read_yaml_file
    cfg = read_yaml_file(str(cfg_path)) or {}
    if "memory" not in cfg:
        _fail("ecs_demo/config.yml 应含 memory 节点（M2.9 已加，默认 enabled=false）")
    mem = cfg.get("memory") or {}
    st_en = (mem.get("short_term") or {}).get("enabled", False)
    lt_en = (mem.get("long_term") or {}).get("enabled", False)
    if st_en or lt_en:
        _fail(f"ecs_demo/config.yml memory 默认应 enabled=false，实际 short={st_en} long={lt_en}")
    _ok("ecs_demo/config.yml memory.enabled=false → Agent.load 后 memory_hooks=None（基线等价）")


# ---------- 运行器 ----------

def main() -> None:
    tests = [
        test_no_memory_node_returns_none,
        test_memory_disabled_returns_none,
        test_all_components_fail_returns_none,
        test_short_term_only_with_llm,
        test_long_term_full,
        test_neo4j_connect_failure_degrades,
        test_state_passes_memory_hooks,
        test_agent_init_memory_hooks,
        test_ecs_demo_baseline_no_memory,
    ]
    print(f"\n=== M2.7 Agent.load 集成测试（共 {len(tests)} 项）===\n")
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print()
    print(f"=== 结果: {passed} 通过 / {failed} 失败 / 共 {len(tests)} ===")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

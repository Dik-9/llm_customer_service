# -*- coding: utf-8 -*-
"""
M1.7 验证脚本：配置扩展 + Agent.load 加载 MCP 客户端

确定性验证（不连真实 MCP server，不调 LLM）：
1. MCPServerConfig.from_dict：circuit_breaker 子节扁平化为顶层字段 + 默认值
2. MCPConfig.from_dict：完整解析 enabled/servers + get_server 查询
3. EndpointsConfig.load：从 ecs_demo/endpoints.yml 加载，mcp 节正确解析
4. EndpointsConfig.from_dict：无 mcp 节时 mcp 为 None（向后兼容）
5. build_tool_registry(None)：返回本地 ToolRegistry（基线）
6. build_tool_registry(enabled=False)：返回本地 ToolRegistry（基线）
7. build_tool_registry(enabled=True + ecommerce server)：注入 MCPClient + 电商映射，路由生效
8. build_tool_registry(enabled=True 但无 servers)：优雅回退本地（容错）
9. DEFAULT_ECOMMERCE_MAPPING：14 个映射，key 全 action_ 前缀，value 全 ecommerce__ 前缀
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atguigu_ai.shared.config import (
    EndpointsConfig,
    MCPConfig,
    MCPServerConfig,
)
from atguigu_ai.mcp.tool_registry import (
    DEFAULT_ECOMMERCE_MAPPING,
    LocalExecutable,
    MCPExecutable,
    ToolRegistry,
    build_tool_registry,
)


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- 测试用例 ----------

def test_mcp_server_config_from_dict() -> None:
    """测试 1：MCPServerConfig 解析（circuit_breaker 扁平化 + 默认值）。"""
    print("[测试 1] MCPServerConfig.from_dict")
    cfg = MCPServerConfig.from_dict({
        "base_url": "http://127.0.0.1:8765/mcp",
        "timeout": 8,
        "retry": 3,
        "circuit_breaker": {"failure_threshold": 7, "reset_timeout": 60},
    })
    if cfg.base_url != "http://127.0.0.1:8765/mcp":
        _fail(f"base_url 不对: {cfg.base_url}")
    if cfg.timeout != 8 or cfg.retry != 3:
        _fail(f"timeout/retry 不对: {cfg.timeout}/{cfg.retry}")
    # circuit_breaker 子节扁平化到顶层字段
    if cfg.failure_threshold != 7 or cfg.reset_timeout != 60:
        _fail(f"circuit_breaker 未扁平化: {cfg.failure_threshold}/{cfg.reset_timeout}")
    _ok("circuit_breaker 子节扁平化到 failure_threshold/reset_timeout")

    # 默认值
    cfg2 = MCPServerConfig.from_dict({})
    if cfg2.base_url or cfg2.timeout != 10.0 or cfg2.retry != 2:
        _fail(f"默认值不对: {cfg2}")
    if cfg2.failure_threshold != 5 or cfg2.reset_timeout != 30.0:
        _fail(f"circuit_breaker 默认值不对: {cfg2.failure_threshold}/{cfg2.reset_timeout}")
    _ok("空配置 → 默认值（timeout=10 retry=2 failure=5 reset=30）")

    # circuit_breaker 子节缺失但顶层有其他字段
    cfg3 = MCPServerConfig.from_dict({"base_url": "http://x/mcp"})
    if cfg3.failure_threshold != 5:
        _fail(f"无 circuit_breaker 时应用默认值: {cfg3.failure_threshold}")
    _ok("无 circuit_breaker 子节 → 熔断参数取默认值")


def test_mcp_config_from_dict() -> None:
    """测试 2：MCPConfig 解析 + get_server。"""
    print("[测试 2] MCPConfig.from_dict")
    mcp = MCPConfig.from_dict({
        "enabled": True,
        "servers": {
            "ecommerce": {
                "base_url": "http://127.0.0.1:8765/mcp",
                "circuit_breaker": {"failure_threshold": 5, "reset_timeout": 30},
            },
            "extra": {"base_url": "http://127.0.0.1:8766/mcp"},
        },
    })
    if not mcp.enabled:
        _fail("enabled 应为 True")
    if len(mcp.servers) != 2:
        _fail(f"servers 数量不对: {len(mcp.servers)}")
    srv = mcp.get_server("ecommerce")
    if not isinstance(srv, MCPServerConfig):
        _fail(f"get_server 应返回 MCPServerConfig: {type(srv)}")
    if srv.base_url != "http://127.0.0.1:8765/mcp":
        _fail(f"ecommerce base_url 不对: {srv.base_url}")
    if mcp.get_server("not_exists") is not None:
        _fail("不存在的 server 应返回 None")
    _ok("解析 enabled/servers + get_server 命中/未命中")

    # 默认（空配置）
    mcp2 = MCPConfig.from_dict({})
    if mcp2.enabled or mcp2.servers:
        _fail(f"空配置应 enabled=False 且无 servers: {mcp2}")
    _ok("空配置 → enabled=False, servers={}")


def test_endpoints_config_load_mcp() -> None:
    """测试 3：从 ecs_demo/endpoints.yml 加载，mcp 节正确解析。"""
    print("[测试 3] EndpointsConfig.load 真实 endpoints.yml")
    yml_path = Path(__file__).resolve().parent / "endpoints.yml"
    if not yml_path.exists():
        _fail(f"endpoints.yml 不存在: {yml_path}")
    endpoints = EndpointsConfig.load(yml_path)

    # mcp 节存在
    if endpoints.mcp is None:
        _fail("endpoints.yml 有 mcp 节，endpoints.mcp 不应为 None")
    if endpoints.mcp.enabled:
        _fail("endpoints.yml mcp.enabled 应为 false（默认关闭）")
    srv = endpoints.mcp.get_server("ecommerce")
    if srv is None:
        _fail("endpoints.yml 应配置 ecommerce server")
    if srv.base_url != "http://127.0.0.1:8765/mcp":
        _fail(f"ecommerce base_url 不对: {srv.base_url}")
    if srv.timeout != 10 or srv.retry != 2:
        _fail(f"ecommerce timeout/retry 不对: {srv.timeout}/{srv.retry}")
    if srv.failure_threshold != 5 or srv.reset_timeout != 30:
        _fail(f"circuit_breaker 不对: {srv.failure_threshold}/{srv.reset_timeout}")
    _ok("endpoints.yml mcp 节：enabled=false + ecommerce server + circuit_breaker 全部解析")


def test_endpoints_config_no_mcp_section() -> None:
    """测试 4：无 mcp 节时 mcp 为 None（向后兼容旧配置）。"""
    print("[测试 4] 无 mcp 节向后兼容")
    endpoints = EndpointsConfig.from_dict({
        "models": {"default": {"type": "openai", "model": "x", "api_key": "k"}},
    })
    if endpoints.mcp is not None:
        _fail(f"无 mcp 节时 mcp 应为 None: {endpoints.mcp}")
    _ok("无 mcp 节 → endpoints.mcp=None（旧配置不受影响）")


def test_build_tool_registry_none() -> None:
    """测试 5：build_tool_registry(None) 返回本地 ToolRegistry。"""
    print("[测试 5] build_tool_registry(None)")
    registry = build_tool_registry(None)
    if not isinstance(registry, ToolRegistry):
        _fail(f"应返回 ToolRegistry: {type(registry)}")
    if registry.mcp_enabled:
        _fail("None 入参时 mcp_enabled 应为 False")
    exe = registry.get("action_ask_order_id")
    if not isinstance(exe, LocalExecutable):
        _fail(f"None 入参时全部走 LocalExecutable: {type(exe)}")
    _ok("None 入参 → 本地 ToolRegistry，全走 LocalExecutable（基线）")


def test_build_tool_registry_disabled() -> None:
    """测试 6：build_tool_registry(enabled=False) 返回本地 ToolRegistry。"""
    print("[测试 6] build_tool_registry(enabled=False)")
    mcp_config = MCPConfig.from_dict({
        "enabled": False,
        "servers": {"ecommerce": {"base_url": "http://127.0.0.1:8765/mcp"}},
    })
    registry = build_tool_registry(mcp_config)
    if registry.mcp_enabled:
        _fail("enabled=False 时 mcp_enabled 应为 False")
    if not isinstance(registry.get("action_ask_order_id"), LocalExecutable):
        _fail("enabled=False 时应走 LocalExecutable")
    _ok("enabled=False → 本地 ToolRegistry（配置了 server 也不启用）")


def test_build_tool_registry_enabled() -> None:
    """测试 7：build_tool_registry(enabled=True) 注入 MCPClient + 电商映射，路由生效。"""
    print("[测试 7] build_tool_registry(enabled=True)")
    mcp_config = MCPConfig.from_dict({
        "enabled": True,
        "servers": {
            "ecommerce": {
                "base_url": "http://127.0.0.1:8765/mcp",
                "timeout": 10,
                "retry": 2,
                "circuit_breaker": {"failure_threshold": 5, "reset_timeout": 30},
            }
        },
    })
    registry = build_tool_registry(mcp_config)
    if not registry.mcp_enabled:
        _fail("enabled=True 时 mcp_enabled 应为 True")
    # 映射表内的 action → MCPExecutable
    exe = registry.get("action_ask_order_id")
    if not isinstance(exe, MCPExecutable):
        _fail(f"映射表内应返回 MCPExecutable: {type(exe)}")
    if not registry.has_mcp_route("action_ask_order_id"):
        _fail("has_mcp_route(action_ask_order_id) 应为 True")
    # 映射表外的 action → LocalExecutable
    exe2 = registry.get("action_send_text")
    if not isinstance(exe2, LocalExecutable):
        _fail(f"映射表外应返回 LocalExecutable: {type(exe2)}")
    if registry.has_mcp_route("action_send_text"):
        _fail("has_mcp_route(action_send_text) 应为 False")
    _ok("enabled=True → MCPClient 注入 + 电商映射路由（表内→MCP / 表外→本地）")


def test_build_tool_registry_enabled_no_servers() -> None:
    """测试 8：build_tool_registry(enabled=True 但无 servers) 优雅回退本地。"""
    print("[测试 8] build_tool_registry(enabled=True 无 servers)")
    mcp_config = MCPConfig.from_dict({"enabled": True, "servers": {}})
    registry = build_tool_registry(mcp_config)
    if registry.mcp_enabled:
        _fail("无 servers 时应回退本地，mcp_enabled=False")
    if not isinstance(registry.get("action_ask_order_id"), LocalExecutable):
        _fail("无 servers 时应走 LocalExecutable")
    _ok("enabled=True 但无 servers → 优雅回退本地 ToolRegistry（容错）")


def test_default_mapping() -> None:
    """测试 9：DEFAULT_ECOMMERCE_MAPPING 14 个映射的命名规范。"""
    print("[测试 9] DEFAULT_ECOMMERCE_MAPPING")
    if len(DEFAULT_ECOMMERCE_MAPPING) != 14:
        _fail(f"应有 14 个映射，实际 {len(DEFAULT_ECOMMERCE_MAPPING)}")
    for action_name, tool_name in DEFAULT_ECOMMERCE_MAPPING.items():
        if not action_name.startswith("action_"):
            _fail(f"key 应以 action_ 开头: {action_name}")
        if not tool_name.startswith("ecommerce__"):
            _fail(f"value 应以 ecommerce__ 开头: {tool_name}")
    # 关键映射抽检
    if DEFAULT_ECOMMERCE_MAPPING.get("action_ask_order_id") != "ecommerce__query_order":
        _fail(f"action_ask_order_id 映射不对: {DEFAULT_ECOMMERCE_MAPPING.get('action_ask_order_id')}")
    if DEFAULT_ECOMMERCE_MAPPING.get("action_apply_postsale") != "ecommerce__apply_postsale":
        _fail(f"action_apply_postsale 映射不对: {DEFAULT_ECOMMERCE_MAPPING.get('action_apply_postsale')}")
    # 与 server 端工具名一致性抽检（覆盖 订单/物流/售后 三类）
    expected = {
        "action_ask_order_id": "ecommerce__query_order",
        "action_get_logistics_info": "ecommerce__get_logistics_info",
        "action_apply_postsale": "ecommerce__apply_postsale",
    }
    for k, v in expected.items():
        if DEFAULT_ECOMMERCE_MAPPING.get(k) != v:
            _fail(f"{k} 应映射到 {v}，实际 {DEFAULT_ECOMMERCE_MAPPING.get(k)}")
    _ok(f"14 个映射全部 action_→ecommerce__ 命名规范，覆盖订单/物流/售后三类")


def main() -> int:
    print("=" * 60)
    print("M1.7 配置扩展 + Agent.load 加载 MCP 客户端 单元测试")
    print("=" * 60)
    print()
    tests = [
        test_mcp_server_config_from_dict,
        test_mcp_config_from_dict,
        test_endpoints_config_load_mcp,
        test_endpoints_config_no_mcp_section,
        test_build_tool_registry_none,
        test_build_tool_registry_disabled,
        test_build_tool_registry_enabled,
        test_build_tool_registry_enabled_no_servers,
        test_default_mapping,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M1.7 配置扩展就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

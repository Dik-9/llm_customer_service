# -*- coding: utf-8 -*-
"""
M1.9 验证脚本：MCP 端到端集成测试

在真实 httpx + FastAPI/ASGI 链路上验证全链路打通（不连真实 TCP、不调 LLM、不连 DB）：
  Agent.load 配置层 → ToolRegistry → MCPClient（httpx）→ MCPServer（FastAPI）
  → ProxyTracker → Action.run → content 扁平化 → 响应回传 → mcp_result_to_action_result
  → 副作用 apply 到真实 tracker

用 httpx.ASGITransport 把 MCPClient 的 HTTP 调用直接路由到进程内 MCPServer.app，
既走完整 JSON-RPC 序列化/路由/解析栈，又保持确定性、无需起 uvicorn。

复用 ecommerce_mcp_server.wrap_action + ProxyTracker（服务端适配器），
但 Action 用 mock 实现（避免 DB 依赖），验证的是【协议链路 + 适配器】，不是【电商业务】。

用例：
1. initialize + tools/list：真实 ASGI 链路拉取工具
2. _execute_action 端到端：MCPClient → MCPServer → ProxyTracker → Action → 副作用 apply
3. action_node 端到端：节点级全链路（responses + bot 消息 + latest_action_name）
4. MCP 连接失败 → 降级本地直调（MockTransport 触发 ConnectError）
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# ecommerce_mcp_server 顶部会把 ECS_DEMO 加入 sys.path
from ecommerce_mcp_server import wrap_action  # noqa: E402

from atguigu_ai.agent.actions import Action, ActionResult, register_action  # noqa: E402
from atguigu_ai.agent.graph.nodes.action import _execute_action, action_node  # noqa: E402
from atguigu_ai.mcp.client import MCPClient, MCPClientConfig  # noqa: E402
from atguigu_ai.mcp.server import MCPServer  # noqa: E402
from atguigu_ai.mcp.tool_registry import DEFAULT_ECOMMERCE_MAPPING, ToolRegistry  # noqa: E402


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- mock 组件 ----------

class _MockTracker:
    """客户端真实 tracker 替身：记录 slot 读写 + bot 消息 + latest_action_name。"""
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


class _ServerQueryOrderAction(Action):
    """服务端 mock Action：返回 SERVER 标记响应 + set_slot 副作用（被 ProxyTracker 捕获）。"""
    @property
    def name(self) -> str:
        return "action_ask_order_id"

    async def run(self, tracker: Any, domain: Optional[Any] = None, **kwargs: Any) -> ActionResult:
        user_id = tracker.get_slot("user_id")
        result = ActionResult()
        result.add_response(f"SERVER: 查询用户 {user_id} 的订单", buttons=[{"title": "O001"}])
        # 模拟 Action 内部 set_slot 副作用（ProxyTracker 会捕获为 slot_sets 回传）
        tracker.set_slot("goto", "action_ask_order_id_before_delivered")
        return result


class _LocalQueryOrderAction(Action):
    """本地降级 mock Action：返回 LOCAL 标记响应，用于验证降级路径命中本地。"""
    @property
    def name(self) -> str:
        return "action_ask_order_id"

    async def run(self, tracker: Any, domain: Optional[Any] = None, **kwargs: Any) -> ActionResult:
        result = ActionResult()
        result.add_response("LOCAL: 本地兜底查询")
        return result


# ---------- 工厂 ----------

def _make_server() -> MCPServer:
    """构造进程内 MCPServer，注册一个 mock 包装工具。"""
    server = MCPServer(name="ecommerce-mcp-test", version="1.0.0")
    tool = wrap_action(
        _ServerQueryOrderAction,
        "ecommerce__query_order",
        "查询用户订单（E2E mock，不连 DB）",
        {"type": "object",
         "properties": {"user_id": {"type": "string"}, "goto": {"type": "string"}},
         "required": ["user_id"]},
    )
    server.register_tool(tool)
    return server


def _make_client(server: MCPServer) -> MCPClient:
    """构造指向进程内 server.app 的 MCPClient（ASGITransport，不走真实 TCP）。"""
    transport = httpx.ASGITransport(app=server.app)
    return MCPClient(MCPClientConfig(
        base_url="http://testserver/mcp",
        timeout=5.0,
        retry=0,
        transport=transport,
    ))


def _make_state(action_name: str, tracker: _MockTracker, tool_registry: Any) -> Dict[str, Any]:
    return {
        "tracker": tracker,
        "domain": None,
        "metadata": {},
        "current_prediction": SimpleNamespace(action=action_name, metadata={}),
        "final_responses": [],
        "action_count": 0,
        "node_history": [],
        "_command_generator": None,
        "_tool_registry": tool_registry,
    }


# ---------- 测试用例 ----------

def test_initialize_and_list_tools() -> None:
    """测试 1：initialize + tools/list 真实 ASGI 链路。"""
    print("[测试 1] initialize + tools/list")
    server = _make_server()
    client = _make_client(server)

    async def run() -> list:
        await client.initialize()
        tools = await client.list_tools()
        await client.close()
        return tools

    tools = asyncio.run(run())
    if not client.is_initialized:
        _fail("initialize 后 is_initialized 应为 True")
    names = [t.get("name") for t in tools]
    if "ecommerce__query_order" not in names:
        _fail(f"应返回 ecommerce__query_order，实际 {names}")
    _ok(f"initialize + tools/list 返回 {len(tools)} 个工具: {names}（真实 ASGI 链路）")


def test_execute_action_e2e() -> None:
    """测试 2：_execute_action 端到端 MCP 链路（含 ProxyTracker 副作用回传）。"""
    print("[测试 2] _execute_action 端到端")
    server = _make_server()
    client = _make_client(server)
    registry = ToolRegistry(mcp_client=client, mcp_mapping=DEFAULT_ECOMMERCE_MAPPING)
    tracker = _MockTracker(slots={"user_id": "1001", "goto": "initial"})

    async def run() -> Any:
        r = await _execute_action("action_ask_order_id", tracker, None, {}, registry)
        await client.close()
        return r

    result = asyncio.run(run())
    if result is None:
        _fail("MCP 链路应返回 ActionResult")
    if not result.responses or not result.responses[0].get("text", "").startswith("SERVER:"):
        _fail(f"响应应来自服务端: {result.responses if result else None}")
    if result.responses[0].get("buttons") != [{"title": "O001"}]:
        _fail(f"buttons 透传失败: {result.responses[0]}")
    # 服务端 Action 的 set_slot 副作用经 ProxyTracker → slot_sets → apply 到真实 tracker
    if tracker.get_slot("goto") != "action_ask_order_id_before_delivered":
        _fail(f"服务端 set_slot 未 apply 到真实 tracker: goto={tracker.get_slot('goto')}")
    _ok("MCP 全链路：服务端 responses + buttons + set_slot 副作用 apply 到真实 tracker")


def test_action_node_e2e() -> None:
    """测试 3：action_node 节点级端到端 MCP 链路。"""
    print("[测试 3] action_node 端到端")
    server = _make_server()
    client = _make_client(server)
    registry = ToolRegistry(mcp_client=client, mcp_mapping=DEFAULT_ECOMMERCE_MAPPING)
    tracker = _MockTracker(slots={"user_id": "1001"})
    state = _make_state("action_ask_order_id", tracker, tool_registry=registry)

    async def run() -> Dict[str, Any]:
        u = await action_node(state)
        await client.close()
        return u

    update = asyncio.run(run())
    final = update.get("final_responses", [])
    if not final or not final[0].get("text", "").startswith("SERVER:"):
        _fail(f"action_node 应经 MCP 返回服务端响应: {final}")
    if tracker.latest_action_name != "action_ask_order_id":
        _fail(f"latest_action_name 不对: {tracker.latest_action_name}")
    if not tracker.bot_messages:
        _fail("应添加 bot 消息到 tracker")
    if update.get("action_count") != 1:
        _fail(f"action_count 应为 1: {update.get('action_count')}")
    if tracker.get_slot("goto") != "action_ask_order_id_before_delivered":
        _fail(f"服务端 set_slot 未 apply: goto={tracker.get_slot('goto')}")
    _ok("action_node → ToolRegistry → MCPClient → MCPServer → ProxyTracker → Action 全链路打通")


def test_degradation_to_local() -> None:
    """测试 4：MCP 连接失败 → 降级本地直调。"""
    print("[测试 4] MCP 失败降级本地")
    register_action(_LocalQueryOrderAction())  # 注册本地兜底 Action

    def _fail_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("mock connection refused")

    transport = httpx.MockTransport(_fail_handler)
    client = MCPClient(MCPClientConfig(
        base_url="http://testserver/mcp",
        timeout=2.0,
        retry=0,
        transport=transport,
    ))
    registry = ToolRegistry(mcp_client=client, mcp_mapping=DEFAULT_ECOMMERCE_MAPPING)
    tracker = _MockTracker(slots={"user_id": "1001"})

    async def run() -> Any:
        r = await _execute_action("action_ask_order_id", tracker, None, {}, registry)
        await client.close()
        return r

    result = asyncio.run(run())
    if result is None:
        _fail("降级应返回本地 ActionResult")
    if not result.responses or not result.responses[0].get("text", "").startswith("LOCAL:"):
        _fail(f"应降级到本地 _LocalQueryOrderAction: {result.responses if result else None}")
    _ok("MCP 连接失败 → MCPExecutable 降级 LocalExecutable → 本地 Action 兜底")


def main() -> int:
    print("=" * 60)
    print("M1.9 MCP 端到端集成测试（真实 httpx + FastAPI/ASGI 链路）")
    print("=" * 60)
    print()
    tests = [
        test_initialize_and_list_tools,
        test_execute_action_e2e,
        test_action_node_e2e,
        test_degradation_to_local,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M1.9 端到端链路打通")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

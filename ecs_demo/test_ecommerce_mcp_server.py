# -*- coding: utf-8 -*-
"""
M1.5 验证脚本：电商 MCP 服务入口单元测试

确定性验证包装层逻辑（不依赖数据库）：
1. build_tools 注册 14 个工具（订单 8 + 物流 2 + 售后 4）
2. tools/list 返回的工具名都是 ecommerce__ 前缀，SPEC §3.4 的 8 个订单工具齐备
3. action_result_to_content 扁平化正确（responses/slot_sets/reject_action_listen/events/空兜底）
4. wrap_action 用 mock Action 端到端验证：handler 把 Action.run 结果转成标准 MCP content

连库冒烟留到 M1.9 端到端验证。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
ECS_DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ECS_DEMO))

from fastapi.testclient import TestClient

from atguigu_ai.agent.actions import Action, ActionResult
from atguigu_ai.mcp import JSONRPC_VERSION, Tool

from ecommerce_mcp_server import (
    ProxyTracker,
    action_result_to_content,
    wrap_action,
    build_tools,
    create_server,
)


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# SPEC §3.4 规定的 8 个订单工具名
_ORDER_TOOL_NAMES = [
    "ecommerce__query_order",
    "ecommerce__get_order_detail",
    "ecommerce__query_receive_info",
    "ecommerce__list_provinces",
    "ecommerce__list_cities",
    "ecommerce__list_districts",
    "ecommerce__update_receive_info",
    "ecommerce__cancel_order",
]


def test_tools_count() -> None:
    """测试 1：注册 14 个工具。"""
    print("[测试 1] 工具数量")
    tools = build_tools()
    if len(tools) != 14:
        _fail(f"应注册 14 个工具，实际 {len(tools)}")
    server = create_server()
    if len(server.tools) != 14:
        _fail(f"server.tools 应为 14，实际 {len(server.tools)}")
    _ok(f"注册 14 个工具（订单 8 + 物流 2 + 售后 4）")


def test_tool_names() -> None:
    """测试 2：工具名都是 ecommerce__ 前缀，SPEC §3.4 订单工具齐备。"""
    print("[测试 2] 工具命名空间")
    server = create_server()
    names = list(server.tools.keys())
    # 全部 ecommerce__ 前缀
    bad = [n for n in names if not n.startswith("ecommerce__")]
    if bad:
        _fail(f"非 ecommerce__ 前缀的工具: {bad}")
    # SPEC §3.4 的 8 个订单工具都在
    missing = [n for n in _ORDER_TOOL_NAMES if n not in names]
    if missing:
        _fail(f"缺少 SPEC §3.4 规定的工具: {missing}")
    # 物流/售后工具也在
    for n in ["ecommerce__list_logistics_companys", "ecommerce__get_logistics_info",
              "ecommerce__query_postsale_orders", "ecommerce__check_postsale_eligible",
              "ecommerce__ask_postsale_reason", "ecommerce__apply_postsale"]:
        if n not in names:
            _fail(f"缺少工具: {n}")
    _ok(f"14 个工具均为 ecommerce__ 前缀，SPEC §3.4 订单工具齐备")


def test_action_result_to_content() -> None:
    """测试 3：action_result_to_content 扁平化。"""
    print("[测试 3] content 扁平化")

    # 3.1 完整：responses + slot_sets + reject_action_listen + events
    tracker = ProxyTracker(slots={"user_id": "1001"})
    tracker.set_slot("order_id", "false")  # 模拟 Action 内部 set_slot
    result = ActionResult()
    result.add_response("暂无订单")
    result.reject_action_listen = True
    result.add_event("slot_set", name="order_id", value="false")
    content = action_result_to_content(result, tracker)
    types = [c["type"] for c in content]
    if types != ["responses", "slot_sets", "reject_action_listen", "events"]:
        _fail(f"content 条目顺序不对: {types}")
    if content[0]["data"][0]["text"] != "暂无订单":
        _fail(f"responses data 不对: {content[0]['data']}")
    if content[1]["data"] != {"order_id": "false"}:
        _fail(f"slot_sets data 不对: {content[1]['data']}")
    if content[2]["data"] is not True:
        _fail(f"reject_action_listen data 应为 True，实际 {content[2]['data']}")
    _ok("完整结果扁平化顺序正确（responses→slot_sets→reject_action_listen→events）")

    # 3.2 空结果 → handler 兜底空 responses（这里函数本身返回空 list）
    tracker2 = ProxyTracker()
    result2 = ActionResult()
    content2 = action_result_to_content(result2, tracker2)
    if content2 != []:
        _fail(f"空结果应返回空 list，实际 {content2}")
    _ok("空结果返回空 list（handler 层兜底为空 responses）")

    # 3.3 只有 responses（如 action_cancel_order）
    tracker3 = ProxyTracker()
    result3 = ActionResult()
    result3.add_response("订单已取消")
    content3 = action_result_to_content(result3, tracker3)
    if len(content3) != 1 or content3[0]["type"] != "responses":
        _fail(f"只有 responses 时应单条，实际 {content3}")
    _ok("只有 responses 的结果正确（无 slot_sets/reject_action_listen）")


def test_wrap_action_with_mock() -> None:
    """测试 4：wrap_action 用 mock Action 端到端验证包装层。"""
    print("[测试 4] wrap_action 端到端")

    # mock Action：读 user_id，写 order_id，返回 response + reject + event
    class _MockAction(Action):
        @property
        def name(self) -> str:
            return "mock_action"

        async def run(
            self,
            tracker: Any,
            domain: Optional[Any] = None,
            **kwargs: Any,
        ) -> ActionResult:
            result = ActionResult()
            result.add_response(f"hello {tracker.get_slot('user_id')}")
            tracker.set_slot("order_id", "false")
            result.reject_action_listen = True
            result.add_event("mock_event", key="val")
            return result

    tool = wrap_action(
        _MockAction,
        "mock__hello",
        "测试用 mock 工具",
        input_schema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    )

    # 用 TestClient 起服务调一次
    from atguigu_ai.mcp import MCPServer
    server = MCPServer(name="mock-server", version="0.1.0")
    server.register_tool(tool)
    client = TestClient(server.app)

    resp = client.post("/mcp", json={
        "jsonrpc": JSONRPC_VERSION, "method": "tools/call", "id": 1,
        "params": {"name": "mock__hello", "arguments": {"user_id": "1001"}},
    })
    if resp.status_code != 200:
        _fail(f"HTTP 状态码应为 200，实际 {resp.status_code}: {resp.text}")
    data = resp.json()
    if "result" not in data:
        _fail(f"缺少 result: {data}")
    result = data["result"]
    content = result.get("content", [])
    types = [c["type"] for c in content]
    if types != ["responses", "slot_sets", "reject_action_listen", "events"]:
        _fail(f"content 条目类型/顺序不对: {types}")
    # responses 含 mock 返回的 text
    if "hello 1001" not in content[0]["data"][0]["text"]:
        _fail(f"responses text 不对: {content[0]['data']}")
    # slot_sets 捕获了 Action 内部 set_slot 副作用
    if content[1]["data"] != {"order_id": "false"}:
        _fail(f"slot_sets 未捕获 set_slot 副作用: {content[1]['data']}")
    # reject_action_listen 回传
    if content[2]["data"] is not True:
        _fail(f"reject_action_listen 未回传: {content[2]}")
    # isError 为 False（success 默认 True）
    if result.get("isError") is not False:
        _fail(f"isError 应为 False，实际 {result.get('isError')}")
    _ok("wrap_action 端到端：responses/slot_sets/reject_action_listen/events 全部正确回传")


def test_list_tools_via_http() -> None:
    """测试 5：HTTP tools/list 返回 inputSchema。"""
    print("[测试 5] tools/list 经 HTTP 返回")
    server = create_server()
    client = TestClient(server.app)
    resp = client.post("/mcp", json={
        "jsonrpc": JSONRPC_VERSION, "method": "tools/list", "id": 1, "params": {},
    })
    data = resp.json()
    tools = data.get("result", {}).get("tools", [])
    if len(tools) != 14:
        _fail(f"tools/list 应返回 14 个，实际 {len(tools)}")
    # query_order 的 inputSchema 含 goto enum
    qo = next((t for t in tools if t["name"] == "ecommerce__query_order"), None)
    if qo is None:
        _fail("tools/list 缺少 ecommerce__query_order")
    goto = qo["inputSchema"]["properties"].get("goto", {})
    if "enum" not in goto or len(goto["enum"]) != 6:
        _fail(f"query_order 的 goto 应有 6 个 enum，实际 {goto}")
    if "required" not in qo["inputSchema"] or "user_id" not in qo["inputSchema"]["required"]:
        _fail(f"query_order required 应含 user_id: {qo['inputSchema']}")
    _ok("tools/list 经 HTTP 返回 14 个工具，query_order 的 goto enum + required 正确")


def main() -> int:
    print("=" * 60)
    print("M1.5 电商 MCP 服务入口单元测试")
    print("=" * 60)
    print()
    tests = [
        test_tools_count,
        test_tool_names,
        test_action_result_to_content,
        test_wrap_action_with_mock,
        test_list_tools_via_http,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M1.5 电商 MCP 服务入口就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

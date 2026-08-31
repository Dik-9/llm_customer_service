# -*- coding: utf-8 -*-
"""
M1.5 电商 MCP 服务入口

包装现有电商 Action（订单/物流/售后）为 MCP 工具，通过 MCPServer 暴露。

核心适配问题：
    电商 Action 签名是 run(tracker, domain, **kwargs) -> ActionResult，
    依赖有状态的 tracker（get_slot / set_slot）；而 MCP 是无状态 RPC。
解决：
    ProxyTracker 把 MCP tools/call 的 arguments 当作初始槽位喂给 Action.run，
    并把 Action 内部 set_slot 的副作用、responses、reject_action_listen 标志
    扁平化为 MCP content 条目（SPEC §3.2.3）：
        - {type: "responses",         data: [...]}      要发送的回复
        - {type: "slot_sets",        data: {slot: v}}  Action 内部 set_slot 副作用
        - {type: "reject_action_listen", data: true}   打断 action_listen 的标志
        - {type: "events",           data: [...]}      Flow 事件
    客户端拿到后 apply 到真实 tracker，即可与本地直调等价。

Action.run 内部逻辑零改动，仅在本层做序列化/反序列化。

启动：
    python ecommerce_mcp_server.py --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from atguigu_ai.agent.actions import Action, ActionResult
from atguigu_ai.mcp import MCPServer, Tool

# 让 `from actions...` 和 `from atguigu_ai...` 都能解析
ECS_DEMO = Path(__file__).resolve().parent
ROOT = ECS_DEMO.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ECS_DEMO))


# =============================================================================
# ProxyTracker：有状态 Action ↔ 无状态 MCP 的适配器
# =============================================================================

class ProxyTracker:
    """轻量 tracker 适配器。

    把 MCP tools/call 的 arguments 当作初始槽位喂给 Action.run，
    捕获 Action 内部 set_slot 的副作用（记录到 slot_sets），
    供包装层回传给 MCP 客户端，客户端再 apply 到真实 tracker。

    只实现电商 Action 实际用到的 tracker 接口（get_slot / set_slot /
    get_all_slots / latest_message），不引入完整 DialogueStateTracker。
    """

    def __init__(self, slots: Optional[Dict[str, Any]] = None) -> None:
        self._slots: Dict[str, Any] = dict(slots or {})
        # 记录 Action.run 内部 set_slot 的副作用，供包装层回传
        self.slot_sets: Dict[str, Any] = {}
        # 部分动作可能读取 latest_message（如 action_extract_slots），这里置空
        self.latest_message = None

    def get_slot(self, name: str) -> Any:
        return self._slots.get(name)

    def set_slot(self, name: str, value: Any) -> None:
        self._slots[name] = value
        self.slot_sets[name] = value

    def get_all_slots(self) -> Dict[str, Any]:
        return dict(self._slots)


def action_result_to_content(
    result: ActionResult,
    tracker: ProxyTracker,
) -> List[Dict[str, Any]]:
    """把 ActionResult + ProxyTracker 副作用扁平化为 MCP content 条目。

    顺序：responses → slot_sets → reject_action_listen → events
    （responses 在前，保证客户端优先拿到要发送的回复）。
    """
    content: List[Dict[str, Any]] = []

    # 1. responses（要发送的回复，含 buttons）
    if result.responses:
        content.append({"type": "responses", "data": result.responses})

    # 2. slot_sets（Action 内部 set_slot 的副作用，客户端需 apply 到真实 tracker）
    if tracker.slot_sets:
        content.append({"type": "slot_sets", "data": dict(tracker.slot_sets)})

    # 3. reject_action_listen（ActionAskOrderId 无订单时打断流程的标志）
    if getattr(result, "reject_action_listen", False):
        content.append({"type": "reject_action_listen", "data": True})

    # 4. events（Flow 事件等，预留）
    if result.events:
        content.append({"type": "events", "data": result.events})

    return content


# =============================================================================
# Action → MCP Tool 包装
# =============================================================================

def wrap_action(
    action_cls: Type[Action],
    tool_name: str,
    description: str,
    input_schema: Dict[str, Any],
) -> Tool:
    """把电商 Action 类包装成 MCP Tool。

    handler 逻辑：
        1. 用 args 构造 ProxyTracker
        2. 调 action.run(tracker, domain=None)
        3. 把 result + tracker 副作用扁平化为 content
        4. 返回标准 MCP result {content, isError}
    """
    action = action_cls()  # Action 实例（电商 Action 无状态，可复用）

    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        tracker = ProxyTracker(slots=args)
        # 电商 Action 不依赖 domain（不调 domain.get_response），domain=None 即可
        result = await action.run(tracker, domain=None)
        content = action_result_to_content(result, tracker)
        return {
            "content": content or [{"type": "responses", "data": []}],
            "isError": not result.success,
        }

    return Tool(
        name=tool_name,
        description=description,
        input_schema=input_schema,
        handler=handler,
    )


# =============================================================================
# 工具映射表（SPEC §3.4 + 物流/售后同理）
# =============================================================================

# goto 的 6 个枚举值（SPEC §3.2.2）
_GOTO_ENUM = [
    "action_ask_order_id_before_completed_3_days",
    "action_ask_order_id_before_delivered",
    "action_ask_order_id_before_shipped",
    "action_ask_order_id_shipped",
    "action_ask_order_id_shipped_delivered",
    "action_ask_order_id_after_delivered",
]


def _build_tool_specs() -> List[Tuple[Type[Action], str, str, Dict[str, Any]]]:
    """构建全部电商 MCP 工具映射表 (Action类, 工具名, 描述, inputSchema)。"""
    # 延迟 import：actions 包依赖 ecs_demo 在 sys.path（本文件顶部已设置）
    from actions import (
        ActionAskOrderId,
        ActionGetOrderDetail,
        ActionAskReceiveId,
        ActionAskReceiveProvince,
        ActionAskReceiveCity,
        ActionAskReceiveDistrict,
        ActionAskSetReceiveInfo,
        ActionCancelOrder,
        ActionGetLogisticsCompanys,
        ActionGetLogisticsInfo,
        ActionAskOrderIdAfterDelivered,
        ActionCheckPostsaleEligible,
        ActionAskPostsaleReason,
        ActionApplyPostsale,
    )

    str_prop = lambda desc: {"type": "string", "description": desc}

    return [
        # ---------- 订单 ----------
        (ActionAskOrderId, "ecommerce__query_order",
         "按条件查询用户的订单列表，返回可选择的订单按钮数据",
         {"type": "object",
          "properties": {"user_id": str_prop("用户ID"),
                         "goto": {"type": "string", "enum": _GOTO_ENUM,
                                  "description": "订单状态过滤条件"}},
          "required": ["user_id", "goto"]}),
        (ActionGetOrderDetail, "ecommerce__get_order_detail",
         "获取订单详情（含明细、收货信息、最近物流、售后）",
         {"type": "object",
          "properties": {"order_id": str_prop("订单ID")},
          "required": ["order_id"]}),
        (ActionAskReceiveId, "ecommerce__query_receive_info",
         "展示用户现有收货地址列表供选择，含当前订单的收货信息回填",
         {"type": "object",
          "properties": {"user_id": str_prop("用户ID"),
                         "order_id": str_prop("订单ID（用于回填当前收货信息）")},
          "required": ["user_id"]}),
        (ActionAskReceiveProvince, "ecommerce__list_provinces",
         "列出可选省份",
         {"type": "object", "properties": {}}),
        (ActionAskReceiveCity, "ecommerce__list_cities",
         "按省份列出可选城市",
         {"type": "object",
          "properties": {"receive_province": str_prop("省份")},
          "required": ["receive_province"]}),
        (ActionAskReceiveDistrict, "ecommerce__list_districts",
         "按城市列出可选区县",
         {"type": "object",
          "properties": {"receive_city": str_prop("城市")},
          "required": ["receive_city"]}),
        (ActionAskSetReceiveInfo, "ecommerce__update_receive_info",
         "设置/修改订单收货信息（首次展示并询问确认，确认后入库）",
         {"type": "object",
          "properties": {
              "receive_id": str_prop("收货信息ID（modify/modified 表示新建）"),
              "order_id": str_prop("订单ID"),
              "user_id": str_prop("用户ID"),
              "set_receive_info": {"type": "boolean", "description": "是否确认修改"},
              "receiver_name": str_prop("收货人姓名"),
              "receiver_phone": str_prop("联系电话"),
              "receive_province": str_prop("省份"),
              "receive_city": str_prop("城市"),
              "receive_district": str_prop("区县"),
              "receive_street_address": str_prop("详细地址")},
          "required": ["receive_id", "order_id"]}),
        (ActionCancelOrder, "ecommerce__cancel_order",
         "取消订单（状态改为已取消）",
         {"type": "object",
          "properties": {"order_id": str_prop("订单ID")},
          "required": ["order_id"]}),
        # ---------- 物流 ----------
        (ActionGetLogisticsCompanys, "ecommerce__list_logistics_companys",
         "列出支持的快递公司",
         {"type": "object", "properties": {}}),
        (ActionGetLogisticsInfo, "ecommerce__get_logistics_info",
         "查询订单物流轨迹",
         {"type": "object",
          "properties": {"order_id": str_prop("订单ID")},
          "required": ["order_id"]}),
        # ---------- 售后 ----------
        (ActionAskOrderIdAfterDelivered, "ecommerce__query_postsale_orders",
         "查询用户可申请售后的订单列表",
         {"type": "object",
          "properties": {"user_id": str_prop("用户ID")},
          "required": ["user_id"]}),
        (ActionCheckPostsaleEligible, "ecommerce__check_postsale_eligible",
         "检查订单是否具备售后申请资格（需已签收且在7天售后期内）",
         {"type": "object",
          "properties": {"order_id": str_prop("订单ID")},
          "required": ["order_id"]}),
        (ActionAskPostsaleReason, "ecommerce__ask_postsale_reason",
         "询问/校验售后原因",
         {"type": "object",
          "properties": {"order_id": str_prop("订单ID")},
          "required": ["order_id"]}),
        (ActionApplyPostsale, "ecommerce__apply_postsale",
         "提交售后申请（退款/退货/换货）",
         {"type": "object",
          "properties": {
              "order_id": str_prop("订单ID"),
              "postsale_type": {"type": "string", "enum": ["退款", "退货", "换货"],
                                "description": "售后类型"},
              "postsale_reason": str_prop("售后原因"),
              "user_id": str_prop("用户ID")},
          "required": ["order_id", "postsale_type", "postsale_reason", "user_id"]}),
    ]


def build_tools() -> List[Tool]:
    """构建全部电商 MCP 工具。"""
    return [
        wrap_action(cls, name, desc, schema)
        for cls, name, desc, schema in _build_tool_specs()
    ]


def create_server(name: str = "ecommerce-mcp", version: str = "1.0.0") -> MCPServer:
    """构建电商 MCP Server（注册全部工具）。供测试与入口复用。"""
    server = MCPServer(name=name, version=version)
    for tool in build_tools():
        server.register_tool(tool)
    return server


# =============================================================================
# 入口
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="电商 MCP Server")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    args = parser.parse_args()

    server = create_server()
    print(f"电商 MCP Server 启动: http://{args.host}:{args.port}/mcp")
    print(f"已注册 {len(server.tools)} 个工具:")
    for t in server.tools.values():
        props = list(t.input_schema.get("properties", {}).keys())
        print(f"  - {t.name:40s} 参数: {props or '无'}")
    server.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())

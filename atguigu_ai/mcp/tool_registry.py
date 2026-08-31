# -*- coding: utf-8 -*-
"""
ToolRegistry：统一执行入口 + 命名空间路由 + MCP 失败降级

action_node 原本用 get_action(action_name) 直调本地 Action。
本模块用 ToolRegistry.get(action_name) -> Executable 替换该工厂，
Executable 统一 run(tracker, domain, **kwargs) -> ActionResult 接口：

- LocalExecutable：包装 get_action，本地直调（等价基线行为）
- MCPExecutable：调 MCPClient.call_tool，把 MCP 响应转回 ActionResult，
                  并把 slot_sets / reject_action_listen 等副作用 apply 到真实 tracker；
                  MCP 失败（MCPError）自动降级 LocalExecutable

路由优先级（SPEC §3.3）：
    1. action_name 在 mcp_mapping 且 MCP 客户端可用 → MCPExecutable
    2. 否则 → LocalExecutable

MCP 关闭（mcp_client=None）时全部走 LocalExecutable，运行时等价基线（SPEC §6.4）。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from atguigu_ai.agent.actions import ActionResult, get_action
from atguigu_ai.mcp.client import MCPClient
from atguigu_ai.mcp.exceptions import MCPError

logger = logging.getLogger(__name__)


# =============================================================================
# 统一执行接口
# =============================================================================

class Executable(ABC):
    """统一执行接口，让 action_node 无需感知本地/MCP 差异。"""

    @abstractmethod
    async def run(
        self,
        tracker: Any,
        domain: Optional[Any] = None,
        **kwargs: Any,
    ) -> ActionResult:
        """执行动作，返回 ActionResult。"""
        raise NotImplementedError()


class LocalExecutable(Executable):
    """本地 Action 直调（基线行为）。

    包装 get_action(action_name)，找不到时返回失败结果而非抛异常，
    保持与原 action_node 的容错语义一致。
    """

    def __init__(self, action_name: str) -> None:
        self._action_name = action_name

    @property
    def action_name(self) -> str:
        return self._action_name

    async def run(
        self,
        tracker: Any,
        domain: Optional[Any] = None,
        **kwargs: Any,
    ) -> ActionResult:
        action = get_action(self._action_name)
        if action is None:
            logger.warning(f"动作未找到: {self._action_name}")
            result = ActionResult(success=False)
            result.add_response(f"动作未找到: {self._action_name}")
            return result
        return await action.run(tracker, domain, **kwargs)


# =============================================================================
# MCP 响应 → ActionResult 转换
# =============================================================================

def mcp_result_to_action_result(mcp_result: Any, tracker: Any) -> ActionResult:
    """把 MCP call_tool 返回的 {content, isError} 转回 ActionResult。

    并把以下副作用 apply 到真实 tracker（让 MCP 调用等价本地直调）：
        - slot_sets 条目      → tracker.set_slot(name, value)
        - reject_action_listen → result.reject_action_listen = True
        - responses            → result.responses（供 action_node 发送）
        - events               → result.events
    """
    result = ActionResult()

    # 防御：非标准返回
    if not isinstance(mcp_result, dict):
        result.success = False
        result.add_response("MCP 返回结果格式异常")
        return result

    content = mcp_result.get("content", []) or []
    result.success = not mcp_result.get("isError", False)

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        data = item.get("data")

        if item_type == "responses":
            # data 是 responses 列表，每个 {text, buttons?, image?, custom?}
            for resp in (data or []):
                if not isinstance(resp, dict):
                    continue
                text = resp.get("text", "")
                # 其余字段（buttons/image/custom 等）作为 kwargs 透传
                extra = {k: v for k, v in resp.items() if k != "text"}
                result.add_response(text, **extra)

        elif item_type == "slot_sets":
            # data 是 {slot: value}，apply 到真实 tracker
            if isinstance(data, dict) and hasattr(tracker, "set_slot"):
                for slot_name, slot_value in data.items():
                    tracker.set_slot(slot_name, slot_value)

        elif item_type == "reject_action_listen":
            if data:
                result.reject_action_listen = True

        elif item_type == "events":
            for ev in (data or []):
                if isinstance(ev, dict):
                    ev_type = ev.get("event", "unknown")
                    ev_kwargs = {k: v for k, v in ev.items() if k != "event"}
                    result.add_event(ev_type, **ev_kwargs)

    return result


# =============================================================================
# MCPExecutable：走 MCP，失败降级本地
# =============================================================================

class MCPExecutable(Executable):
    """走 MCP 调用，失败自动降级本地直调。

    降级触发条件：MCPClient.call_tool 抛 MCPError（含超时/连接失败/熔断/协议错误）。
    降级路径：调用构造时注入的 fallback（通常是 LocalExecutable）。
    """

    def __init__(
        self,
        action_name: str,
        mcp_tool_name: str,
        mcp_client: MCPClient,
        fallback: Executable,
    ) -> None:
        self._action_name = action_name
        self._mcp_tool_name = mcp_tool_name
        self._mcp_client = mcp_client
        self._fallback = fallback

    async def run(
        self,
        tracker: Any,
        domain: Optional[Any] = None,
        **kwargs: Any,
    ) -> ActionResult:
        # args 从 tracker slots 取（MCP server 端 ProxyTracker 按需 get_slot）
        args: Dict[str, Any] = {}
        if hasattr(tracker, "get_all_slots"):
            args = tracker.get_all_slots() or {}

        try:
            mcp_result = await self._mcp_client.call_tool(self._mcp_tool_name, args)
        except MCPError as e:
            logger.warning(
                f"MCP 调用 {self._mcp_tool_name} 失败，降级本地直调 "
                f"{self._action_name}：{e}"
            )
            return await self._fallback.run(tracker, domain, **kwargs)

        return mcp_result_to_action_result(mcp_result, tracker)


# =============================================================================
# ToolRegistry
# =============================================================================

class ToolRegistry:
    """统一执行入口 + 命名空间路由。

    生命周期：
        registry = ToolRegistry(mcp_client=client, mcp_mapping={...})
        exe = registry.get(action_name)
        result = await exe.run(tracker, domain, **kwargs)

    MCP 关闭：mcp_client=None，registry.get 全部返回 LocalExecutable（基线）。
    """

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        mcp_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self._mcp_client = mcp_client
        self._mcp_mapping: Dict[str, str] = dict(mcp_mapping) if mcp_mapping else {}

    def get(self, action_name: str) -> Executable:
        """根据路由优先级返回 Executable。"""
        if self._mcp_client is not None and action_name in self._mcp_mapping:
            mcp_tool_name = self._mcp_mapping[action_name]
            return MCPExecutable(
                action_name=action_name,
                mcp_tool_name=mcp_tool_name,
                mcp_client=self._mcp_client,
                fallback=LocalExecutable(action_name),
            )
        return LocalExecutable(action_name)

    def has_mcp_route(self, action_name: str) -> bool:
        """该 action 是否走 MCP 路由（MCP 可用且在映射表内）。"""
        return self._mcp_client is not None and action_name in self._mcp_mapping

    @property
    def mcp_enabled(self) -> bool:
        """MCP 是否启用（客户端已注入）。"""
        return self._mcp_client is not None

    async def close(self) -> None:
        """关闭底层 MCP 客户端。"""
        if self._mcp_client is not None:
            await self._mcp_client.close()


# =============================================================================
# 默认映射表 + 工厂
# =============================================================================

# SPEC §3.4：电商 Action 名 → MCP 工具名（含命名空间）映射
# 与 ecs_demo/ecommerce_mcp_server.py 暴露的工具名保持一致
DEFAULT_ECOMMERCE_MAPPING: Dict[str, str] = {
    # 订单
    "action_ask_order_id": "ecommerce__query_order",
    "action_get_order_detail": "ecommerce__get_order_detail",
    "action_ask_receive_id": "ecommerce__query_receive_info",
    "action_ask_receive_province": "ecommerce__list_provinces",
    "action_ask_receive_city": "ecommerce__list_cities",
    "action_ask_receive_district": "ecommerce__list_districts",
    "action_ask_set_receive_info": "ecommerce__update_receive_info",
    "action_cancel_order": "ecommerce__cancel_order",
    # 物流
    "action_get_logistics_companys": "ecommerce__list_logistics_companys",
    "action_get_logistics_info": "ecommerce__get_logistics_info",
    # 售后
    "action_ask_order_id_after_delivered": "ecommerce__query_postsale_orders",
    "action_check_postsale_eligible": "ecommerce__check_postsale_eligible",
    "action_ask_postsale_reason": "ecommerce__ask_postsale_reason",
    "action_apply_postsale": "ecommerce__apply_postsale",
}


def build_tool_registry(mcp_config: Any) -> ToolRegistry:
    """从 MCPConfig 构造 ToolRegistry（Agent.load 调用）。

    - mcp_config 为 None 或 enabled=False → 返回本地 ToolRegistry（mcp_client=None，基线）
    - enabled=True 且配置了 server → 创建 MCPClient + 注入 DEFAULT_ECOMMERCE_MAPPING

    注：MCPClient 此处不调用 initialize（Agent.load 为同步方法）。
    server 端 tools/call 不强制 initialize 协商，首次 call_tool 直连即可；
    如需 initialize 协商，可在 Agent 首次 handle_message 时 lazy 调用。
    """
    enabled = getattr(mcp_config, "enabled", False) if mcp_config else False
    if not enabled:
        return ToolRegistry()

    servers = getattr(mcp_config, "servers", {}) or {}
    if not servers:
        logger.warning("MCP enabled 但未配置 servers，回退本地 ToolRegistry")
        return ToolRegistry()

    # 优先取 ecommerce server，否则取第一个
    server_cfg = servers.get("ecommerce") or next(iter(servers.values()))
    base_url = getattr(server_cfg, "base_url", "")
    if not base_url:
        logger.warning("MCP server base_url 为空，回退本地 ToolRegistry")
        return ToolRegistry()

    from atguigu_ai.mcp.client import (
        CircuitBreakerConfig,
        MCPClient,
        MCPClientConfig,
    )

    client_config = MCPClientConfig(
        base_url=base_url,
        timeout=getattr(server_cfg, "timeout", 10.0),
        retry=getattr(server_cfg, "retry", 2),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=getattr(server_cfg, "failure_threshold", 5),
            reset_timeout=getattr(server_cfg, "reset_timeout", 30.0),
        ),
    )
    mcp_client = MCPClient(client_config)
    logger.info(
        f"MCP 已启用，ToolRegistry 注入 MCPClient(base_url={base_url}) "
        f"+ {len(DEFAULT_ECOMMERCE_MAPPING)} 个电商工具映射"
    )
    return ToolRegistry(
        mcp_client=mcp_client,
        mcp_mapping=DEFAULT_ECOMMERCE_MAPPING,
    )


__all__ = [
    "Executable",
    "LocalExecutable",
    "MCPExecutable",
    "ToolRegistry",
    "mcp_result_to_action_result",
    "DEFAULT_ECOMMERCE_MAPPING",
    "build_tool_registry",
]

# -*- coding: utf-8 -*-
"""
长期记忆召回器（SPEC §4.3 记忆召回与注入 + 指代消歧）

在 understand_node 调用 command_generator.generate 之前，召回用户画像与最近提及实体：
1. 用户画像摘要（偏好 + 默认地址）→ 注入 command_generator 提示词
2. 最近提及的订单 → 注入提示词，并用于指代消歧
3. 指代消歧：用户说"上次那个订单/刚才那个" → 返回最近 order_id，由 hook 写入 tracker 槽

graph_store 已容错（异常返回空），recaller 直接消费空结果即可，无需额外 try/except。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from atguigu_ai.memory.long_term.graph_store import GraphMemoryStore

logger = logging.getLogger(__name__)


# 指代消歧信号（用户引用之前提过的实体）
_COREFERENCE_PATTERNS = [
    re.compile(r"上次(那个|说的|提的|的)"),
    re.compile(r"刚才(那个|说的|提的|的)"),
    re.compile(r"就是(那个|刚才|上次)"),
    re.compile(r"那个订单"),
    re.compile(r"这个订单"),
]


class MemoryRecaller:
    """画像召回 + 指代消歧。

    Args:
        graph_store: 长期记忆图谱存储（可注入 fake）
    """

    def __init__(self, graph_store: GraphMemoryStore) -> None:
        self._store = graph_store

    # ------------------------------------------------------------------
    # 画像召回
    # ------------------------------------------------------------------

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """返回用户画像原始数据（偏好列表 + 默认地址）。"""
        if not user_id:
            return {"user_id": user_id, "preferences": [], "default_address": None}
        return self._store.get_user_profile(user_id)

    def format_profile_text(self, user_id: str) -> str:
        """生成注入提示词的画像文本（无画像时返回空串）。"""
        profile = self.get_user_profile(user_id)
        prefs: List[Dict[str, Any]] = profile.get("preferences") or []
        addr: Optional[Dict[str, Any]] = profile.get("default_address")

        parts: List[str] = []
        if prefs:
            pref_str = "；".join(f"{p.get('type')}={p.get('value')}" for p in prefs)
            parts.append(f"用户长期偏好：{pref_str}")
        if addr:
            label = addr.get("label") or ""
            region = f"{addr.get('province', '')}{addr.get('city', '')}{addr.get('district', '')}"
            parts.append(f"默认收货地址{('标签=' + label) if label else ''}（{region}）")

        return "；".join(parts)

    # ------------------------------------------------------------------
    # 最近提及
    # ------------------------------------------------------------------

    def get_recent_mentions(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """返回用户最近提及的订单列表。"""
        if not user_id:
            return []
        return self._store.get_recent_mentions(user_id, limit=limit)

    def format_mentions_text(self, user_id: str, limit: int = 5) -> str:
        """生成注入提示词的提及文本（无提及时返回空串）。"""
        mentions = self.get_recent_mentions(user_id, limit=limit)
        if not mentions:
            return ""
        items = []
        for m in mentions:
            order_id = m.get("order_id")
            if not order_id:
                continue
            items.append(str(order_id))
        if not items:
            return ""
        return f"【参考记忆】用户最近提及订单：{'、'.join(items)}"

    # ------------------------------------------------------------------
    # 指代消歧
    # ------------------------------------------------------------------

    @staticmethod
    def has_coreference_signal(text: str) -> bool:
        """用户消息是否含指代消歧信号。"""
        if not text:
            return False
        return any(p.search(text) for p in _COREFERENCE_PATTERNS)

    def resolve_coreference(self, user_id: str, message: str, limit: int = 5) -> Optional[str]:
        """指代消歧：若消息含指代信号，返回最近提及的 order_id。

        Returns:
            order_id 字符串；无信号或无提及时返回 None。
        """
        if not user_id or not message:
            return None
        if not self.has_coreference_signal(message):
            return None
        mentions = self.get_recent_mentions(user_id, limit=limit)
        if not mentions:
            return None
        return mentions[0].get("order_id")

    # ------------------------------------------------------------------
    # 统一召回（hook 便捷入口）
    # ------------------------------------------------------------------

    def get_memory_context(self, user_id: str, message: str = "") -> Dict[str, Any]:
        """统一召回画像 + 提及 + 指代消歧，供 hook 注入。

        Returns:
            {
              "profile_text": str,      # 画像文本（可空）
              "mentions_text": str,     # 提及文本（可空）
              "resolved_order_id": Optional[str],  # 指代消歧结果
              "preferences": list,      # 原始偏好
              "default_address": dict,  # 原始默认地址
            }
        """
        profile = self.get_user_profile(user_id)
        mentions = self.get_recent_mentions(user_id, limit=5)
        resolved = self.resolve_coreference(user_id, message)

        return {
            "profile_text": self.format_profile_text(user_id) if profile.get("preferences") or profile.get("default_address") else "",
            "mentions_text": self.format_mentions_text(user_id) if mentions else "",
            "resolved_order_id": resolved,
            "preferences": profile.get("preferences") or [],
            "default_address": profile.get("default_address"),
            "recent_mentions": mentions,
        }


__all__ = ["MemoryRecaller"]

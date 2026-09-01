# -*- coding: utf-8 -*-
"""
记忆系统 hook 编排（SPEC §6.1）

三处挂接点，在 Agent.handle_message / understand_node / action_node 中显式调用：

1. before_understand(tracker)
   长期记忆召回：画像 + 提及 + 指代消歧 → 写入 tracker.memory_context 供 prompt 渲染；
   指代消歧命中时把 order_id 写入 tracker 槽位（SPEC §4.3）。
   同时把短期压缩摘要 session_summary 一并放进 memory_context。

2. after_each_action(tracker, action_result)
   长期记忆实时抽取：对用户最新消息做 LLM 结构化抽取（先粗筛信号），命中的事实写入图谱。
   仅用 latest_message + 最近3轮上下文，token 成本可控。

3. before_save_tracker(tracker)
   - 短期压缩：dialogue_turns 超阈值时 LLM 摘要 + 裁剪 + 写 session_summary 槽
   - 长期兜底：检测到会话结束信号（"再见/结束/拜拜/restart"）时，对完整会话跑一次抽取并入库

所有 hook 捕获自身异常仅记日志，不向主对话链路抛错（SPEC §1.2 多路径保障）。
memory.enabled=false 时 MemoryHooks 为 None，hook 直接 no-op（SPEC §6.4）。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from atguigu_ai.memory.long_term.extractor import ExtractedFact, MemoryExtractor
from atguigu_ai.memory.long_term.graph_store import GraphMemoryStore
from atguigu_ai.memory.long_term.recaller import MemoryRecaller
from atguigu_ai.memory.short_term.compressor import (
    SLOT_SESSION_SUMMARY,
    CompressionResult,
    ShortTermCompressor,
)
from atguigu_ai.shared.config import MemoryConfig

logger = logging.getLogger(__name__)


# 会话结束信号（显式）
_SESSION_END_PATTERNS = [
    re.compile(r"/restart", re.IGNORECASE),
    re.compile(r"再见"),
    re.compile(r"拜拜"),
    re.compile(r"^结束$"),
    re.compile(r"bye", re.IGNORECASE),
]


def _has_session_end_signal(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _SESSION_END_PATTERNS)


def _turns_to_text(turns: List[Any], limit: Optional[int] = None) -> List[Dict[str, str]]:
    """把 DialogueTurn 列表转为 [{user, bot}] 文本（供 extractor prompt）。"""
    items: List[Dict[str, str]] = []
    src = turns[-limit:] if limit else turns
    for turn in src:
        um = getattr(turn, "user_message", None)
        user_text = getattr(um, "text", "") if um else ""
        bot_msgs = getattr(turn, "bot_messages", []) or []
        bot_parts = [getattr(bm, "text", "") or "" for bm in bot_msgs if getattr(bm, "text", None)]
        items.append({"user": user_text, "bot": " / ".join(bot_parts)})
    return items


class MemoryHooks:
    """记忆 hook 编排器，聚合 recaller / extractor / compressor / graph_store。

    任一子模块关闭时对应方法 no-op。组件可全为 None（记忆系统关闭）。
    """

    def __init__(
        self,
        config: MemoryConfig,
        recaller: Optional[MemoryRecaller] = None,
        extractor: Optional[MemoryExtractor] = None,
        compressor: Optional[ShortTermCompressor] = None,
        graph_store: Optional[GraphMemoryStore] = None,
    ) -> None:
        self.config = config
        self.recaller = recaller
        self.extractor = extractor
        self.compressor = compressor
        self.graph_store = graph_store

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # ------------------------------------------------------------------
    # 1. understand 前：召回 + 指代消歧
    # ------------------------------------------------------------------

    async def before_understand(self, tracker: Any) -> Dict[str, Any]:
        """召回画像/提及/摘要 → 写入 tracker.memory_context；指代消歧写 order_id 槽。"""
        ctx: Dict[str, Any] = {
            "profile_text": "",
            "mentions_text": "",
            "session_summary": "",
            "resolved_order_id": None,
        }
        if not self.enabled:
            return ctx

        user_id = self._get_user_id(tracker)
        message = self._get_latest_message_text(tracker)

        # 短期压缩摘要
        if self.config.short_term.enabled:
            summary = self._get_slot(tracker, SLOT_SESSION_SUMMARY)
            if summary:
                ctx["session_summary"] = str(summary)

        # 长期召回
        if self.config.long_term.enabled and self.recaller is not None and user_id:
            try:
                mem_ctx = self.recaller.get_memory_context(user_id, message)
                ctx["profile_text"] = mem_ctx.get("profile_text", "")
                ctx["mentions_text"] = mem_ctx.get("mentions_text", "")
                ctx["resolved_order_id"] = mem_ctx.get("resolved_order_id")
                # 指代消歧命中 → 写 order_id 槽
                resolved = mem_ctx.get("resolved_order_id")
                if resolved and self._get_slot(tracker, "order_id") in (None, "", False):
                    self._set_slot(tracker, "order_id", resolved)
                    print(f"[MemoryHooks] 指代消歧 → order_id={resolved}")
            except Exception as e:
                logger.warning(f"[MemoryHooks] 长期召回失败，降级无记忆: {e}")

        # 写入 tracker 供 prompt_builder 渲染
        try:
            setattr(tracker, "memory_context", ctx)
        except Exception:
            pass
        return ctx

    # ------------------------------------------------------------------
    # 2. action 后：实时抽取
    # ------------------------------------------------------------------

    async def after_each_action(
        self,
        tracker: Any,
        action_result: Any = None,
    ) -> List[ExtractedFact]:
        """实时抽取用户最新消息中的记忆事实并写入图谱。"""
        if not self.enabled or not self.config.long_term.enabled:
            return []
        if not self.config.long_term.realtime_extract:
            return []
        if self.extractor is None or self.graph_store is None:
            return []

        message = self._get_latest_message_text(tracker)
        if not message:
            return []

        user_id = self._get_user_id(tracker)
        if not user_id:
            return []

        try:
            recent = _turns_to_text(getattr(tracker, "dialogue_turns", []) or [], limit=3)
            facts = await self.extractor.extract_realtime(message, recent)
            self._write_facts(user_id, facts, source="realtime")
            if facts:
                print(f"[MemoryHooks] 实时抽取 {len(facts)} 条事实")
            return facts
        except Exception as e:
            logger.warning(f"[MemoryHooks] 实时抽取失败，跳过: {e}")
            return []

    # ------------------------------------------------------------------
    # 3. save tracker 前：短期压缩 + 会话结束兜底
    # ------------------------------------------------------------------

    async def before_save_tracker(
        self,
        tracker: Any,
    ) -> Dict[str, Any]:
        """短期压缩 + 会话结束兜底抽取。

        Returns:
            {"compression": CompressionResult, "end_of_session_facts": List[ExtractedFact],
             "session_ended": bool}
        """
        result: Dict[str, Any] = {
            "compression": CompressionResult(compressed=False),
            "end_of_session_facts": [],
            "session_ended": False,
        }
        if not self.enabled:
            return result

        # 短期压缩
        if self.config.short_term.enabled and self.compressor is not None:
            try:
                if self.compressor.should_compress(tracker):
                    comp = await self.compressor.compress(tracker)
                    result["compression"] = comp
                    print(f"[MemoryHooks] 短期压缩触发，摘要长度={len(comp.summary)}")
            except Exception as e:
                logger.warning(f"[MemoryHooks] 短期压缩失败，跳过: {e}")

        # 会话结束兜底抽取
        message = self._get_latest_message_text(tracker)
        if (self.config.long_term.enabled
                and self.config.long_term.end_of_session_extract
                and self.extractor is not None
                and self.graph_store is not None
                and _has_session_end_signal(message)):
            result["session_ended"] = True
            user_id = self._get_user_id(tracker)
            if user_id:
                try:
                    all_turns = _turns_to_text(getattr(tracker, "dialogue_turns", []) or [])
                    facts = await self.extractor.extract_end_of_session(all_turns)
                    self._write_facts(user_id, facts, source="session_end")
                    result["end_of_session_facts"] = facts
                    print(f"[MemoryHooks] 会话结束兜底抽取 {len(facts)} 条事实")
                except Exception as e:
                    logger.warning(f"[MemoryHooks] 会话结束兜底抽取失败，跳过: {e}")

        return result

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _write_facts(self, user_id: str, facts: List[ExtractedFact], source: str) -> None:
        """把抽取的事实写入图谱。"""
        if not facts or self.graph_store is None:
            return
        for fact in facts:
            try:
                if fact.kind == "preference":
                    self.graph_store.upsert_preference(
                        user_id,
                        pref_type=fact.data.get("type", ""),
                        value=fact.data.get("value", ""),
                        confidence=fact.confidence,
                        source=source,
                    )
                elif fact.kind == "address":
                    self.graph_store.upsert_address(user_id, fact.data)
                elif fact.kind == "order_ref":
                    self.graph_store.add_order_mention(
                        user_id,
                        order_id=fact.data.get("order_id", ""),
                        context=fact.data.get("context", ""),
                    )
            except Exception as e:
                logger.warning(f"[MemoryHooks] 写入事实失败 {fact.kind}: {e}")

    @staticmethod
    def _get_user_id(tracker: Any) -> str:
        # 优先 user_id 槽，回退 sender_id
        if hasattr(tracker, "get_slot"):
            uid = tracker.get_slot("user_id")
            if uid:
                return str(uid)
        sid = getattr(tracker, "sender_id", None)
        return str(sid) if sid else ""

    @staticmethod
    def _get_latest_message_text(tracker: Any) -> str:
        lm = getattr(tracker, "latest_message", None)
        if lm is not None:
            return getattr(lm, "text", "") or ""
        return ""

    @staticmethod
    def _get_slot(tracker: Any, name: str) -> Any:
        if hasattr(tracker, "get_slot"):
            return tracker.get_slot(name)
        return None

    @staticmethod
    def _set_slot(tracker: Any, name: str, value: Any) -> None:
        if hasattr(tracker, "set_slot"):
            tracker.set_slot(name, value)

    def close(self) -> None:
        """关闭底层图谱连接。"""
        if self.graph_store is not None:
            self.graph_store.close()


# ======================================================================
# 模块级便捷函数（SPEC §6.1 签名兼容）
# ======================================================================

async def before_understand(
    tracker: Any,
    hooks: Optional[MemoryHooks],
) -> Dict[str, Any]:
    """understand 节点前 hook（hooks 为 None 时 no-op）。"""
    if hooks is None:
        return {"profile_text": "", "mentions_text": "", "session_summary": "", "resolved_order_id": None}
    return await hooks.before_understand(tracker)


async def after_each_action(
    tracker: Any,
    action_result: Any,
    hooks: Optional[MemoryHooks],
) -> List[ExtractedFact]:
    """action 节点后 hook（hooks 为 None 时 no-op）。"""
    if hooks is None:
        return []
    return await hooks.after_each_action(tracker, action_result)


async def before_save_tracker(
    tracker: Any,
    hooks: Optional[MemoryHooks],
) -> Dict[str, Any]:
    """save tracker 前 hook（hooks 为 None 时 no-op）。"""
    if hooks is None:
        return {"compression": CompressionResult(compressed=False), "end_of_session_facts": [], "session_ended": False}
    return await hooks.before_save_tracker(tracker)


__all__ = [
    "MemoryHooks",
    "before_understand",
    "after_each_action",
    "before_save_tracker",
]

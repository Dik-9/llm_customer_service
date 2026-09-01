# -*- coding: utf-8 -*-
"""
短期记忆压缩器（SPEC §4.2 历史压缩）

触发条件：tracker.dialogue_turns（已完成轮次）数量 > max_raw_turns 时，
保留最近 keep_recent_turns 轮完整记录，前面历史由 LLM 压缩为一段摘要。

数据流：
- 摘要写入 tracker 槽位 session_summary（string）
- 已压缩轮次计数写入槽位 summary_covered_turns（累计原始轮次，跨多次压缩累加）
- 物理裁剪 dialogue_turns：压缩后只保留最近 keep_recent_turns 轮
- 产出 memory_compressed 事件 {from_turn, to_turn, summary} 供观测

不变量（Approach C）：
- dialogue_turns 只持有未压缩的轮次；summary_covered_turns 为累计已压缩轮次数。
- 多次压缩时新摘要与旧摘要合并，summary_covered_turns 单调递增。

LLMClient 可注入；LLM 失败时不裁剪、不写槽位，返回 compressed=False（保障主链路）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from atguigu_ai.shared.llm.base_client import LLMClient

logger = logging.getLogger(__name__)


_PROMPTS_DIR = Path(__file__).parent / "prompts"

# 槽位名（写入 tracker）
SLOT_SESSION_SUMMARY = "session_summary"
SLOT_SUMMARY_COVERED_TURNS = "summary_covered_turns"


@dataclass
class CompressionResult:
    """压缩结果。"""
    compressed: bool
    summary: str = ""
    from_turn: int = 0  # 累计原始轮次起始
    to_turn: int = 0    # 累计原始轮次结束
    event: Optional[Dict[str, Any]] = None  # memory_compressed 事件


class ShortTermCompressor:
    """对话历史压缩器。

    Args:
        llm_client: LLM 客户端（可注入）
        max_raw_turns: 触发压缩的原始轮次阈值
        keep_recent_turns: 保留的最近完整轮次数
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_raw_turns: int = 20,
        keep_recent_turns: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm_client
        self._max_raw_turns = max_raw_turns
        self._keep_recent_turns = keep_recent_turns
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._env = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # 判定
    # ------------------------------------------------------------------

    def should_compress(self, tracker: Any) -> bool:
        """是否需要压缩：已完成轮次超过阈值。"""
        turns = self._get_completed_turns(tracker)
        return len(turns) > self._max_raw_turns

    # ------------------------------------------------------------------
    # 压缩
    # ------------------------------------------------------------------

    async def compress(self, tracker: Any) -> CompressionResult:
        """压缩对话历史。

        Returns:
            CompressionResult（compressed=False 表示无需压缩或 LLM 失败）。
        """
        turns = self._get_completed_turns(tracker)
        total = len(turns)
        if total <= self._max_raw_turns:
            return CompressionResult(compressed=False)

        keep = self._keep_recent_turns
        if total - keep <= 0:
            return CompressionResult(compressed=False)

        # 累计已压缩轮次（用于 from_turn/to_turn 观测）
        covered = self._get_covered_turns(tracker)
        # 待压缩：dialogue_turns 中前 (total - keep) 轮
        to_compress = turns[: total - keep]

        # 渲染 prompt
        previous_summary = self._get_previous_summary(tracker)
        template = self._env.get_template("compress.jinja2")
        prompt = template.render(
            previous_summary=previous_summary,
            turns_to_compress=[self._turn_to_text(t) for t in to_compress],
        )

        # 调 LLM
        try:
            resp = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            summary = (resp.content or "").strip()
        except Exception as e:
            logger.warning(f"[ShortTermCompressor] LLM 调用失败，跳过压缩: {e}")
            return CompressionResult(compressed=False)

        if not summary:
            logger.warning("[ShortTermCompressor] LLM 返回空摘要，跳过压缩")
            return CompressionResult(compressed=False)

        # 写槽位：新摘要与旧摘要合并（旧摘要已在 prompt 中合并，这里直接用 LLM 输出）
        tracker.set_slot(SLOT_SESSION_SUMMARY, summary)
        new_covered = covered + len(to_compress)
        tracker.set_slot(SLOT_SUMMARY_COVERED_TURNS, new_covered)

        # 物理裁剪 dialogue_turns：只保留最近 keep 轮
        self._trim_turns(tracker, keep)

        from_turn = covered + 1
        to_turn = new_covered
        event = {
            "event": "memory_compressed",
            "from_turn": from_turn,
            "to_turn": to_turn,
            "summary": summary,
        }
        logger.info(
            f"[ShortTermCompressor] 压缩 {len(to_compress)} 轮 (第{from_turn}-{to_turn}轮)，"
            f"保留最近 {keep} 轮，摘要 {len(summary)} 字"
        )
        return CompressionResult(
            compressed=True,
            summary=summary,
            from_turn=from_turn,
            to_turn=to_turn,
            event=event,
        )

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _get_completed_turns(tracker: Any) -> List[Any]:
        """获取已完成的对话轮次（不含 _current_turn）。"""
        return list(getattr(tracker, "dialogue_turns", []) or [])

    @staticmethod
    def _get_covered_turns(tracker: Any) -> int:
        """已累计压缩的轮次数。"""
        val = None
        if hasattr(tracker, "get_slot"):
            val = tracker.get_slot(SLOT_SUMMARY_COVERED_TURNS)
        try:
            return int(val) if val is not None else 0
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _get_previous_summary(tracker: Any) -> str:
        if hasattr(tracker, "get_slot"):
            s = tracker.get_slot(SLOT_SESSION_SUMMARY)
            return str(s) if s else ""
        return ""

    @staticmethod
    def _trim_turns(tracker: Any, keep: int) -> None:
        """裁剪 dialogue_turns 到最近 keep 轮。"""
        turns = list(getattr(tracker, "dialogue_turns", []) or [])
        if len(turns) > keep:
            tracker.dialogue_turns = turns[-keep:]

    @staticmethod
    def _turn_to_text(turn: Any) -> Dict[str, str]:
        """把 DialogueTurn 转为 {user, bot} 文本。"""
        user_text = ""
        bot_text = ""
        um = getattr(turn, "user_message", None)
        if um is not None:
            user_text = getattr(um, "text", "") or ""
        bot_msgs = getattr(turn, "bot_messages", []) or []
        bot_parts = []
        for bm in bot_msgs:
            t = getattr(bm, "text", None)
            if t:
                bot_parts.append(t)
        bot_text = " / ".join(bot_parts)
        return {"user": user_text, "bot": bot_text}


__all__ = ["ShortTermCompressor", "CompressionResult", "SLOT_SESSION_SUMMARY", "SLOT_SUMMARY_COVERED_TURNS"]

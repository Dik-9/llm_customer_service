# -*- coding: utf-8 -*-
"""短期记忆：对话历史压缩。"""

from atguigu_ai.memory.short_term.compressor import (
    ShortTermCompressor,
    CompressionResult,
    SLOT_SESSION_SUMMARY,
    SLOT_SUMMARY_COVERED_TURNS,
)

__all__ = [
    "ShortTermCompressor",
    "CompressionResult",
    "SLOT_SESSION_SUMMARY",
    "SLOT_SUMMARY_COVERED_TURNS",
]

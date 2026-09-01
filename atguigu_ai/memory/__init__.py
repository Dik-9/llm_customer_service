# -*- coding: utf-8 -*-
"""
记忆系统（短期 + 长期，SPEC §4）

- 短期记忆：对话历史压缩（short_term/compressor）
- 长期记忆：Neo4j 图谱（long_term/graph_store + extractor + recaller）
- hooks：三处挂接点（understand 前 / action 后 / save tracker 前）

所有改造为增量式，memory.enabled=false 时 hook 直接 no-op，等价基线（SPEC §6.4）。
"""

from atguigu_ai.memory.hooks import (
    MemoryHooks,
    before_understand,
    after_each_action,
    before_save_tracker,
)

__all__ = [
    "MemoryHooks",
    "before_understand",
    "after_each_action",
    "before_save_tracker",
]

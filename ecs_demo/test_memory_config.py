# -*- coding: utf-8 -*-
"""
M2.1 验证脚本：MemoryConfig 配置数据类 + config.yml 解析

确定性验证（纯配置解析，无外部依赖）：
1. MemoryConfig 默认值：全部关闭（等价基线，SPEC §6.4）
2. ShortTermMemoryConfig.from_dict：自定义阈值
3. LongTermMemoryConfig.from_dict：自定义 database + 开关
4. MemoryConfig.from_dict：完整 memory 节解析
5. MemoryConfig.enabled：任一子模块启用即为 True
6. 环境变量替换：${VAR:default} 在 memory 配置中生效
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atguigu_ai.shared.config import (
    MemoryConfig,
    ShortTermMemoryConfig,
    LongTermMemoryConfig,
)


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


def test_default_disabled() -> None:
    """测试 1：默认全部关闭。"""
    print("[测试 1] MemoryConfig 默认值")
    cfg = MemoryConfig()
    if cfg.short_term.enabled is not False:
        _fail(f"short_term.enabled 默认应为 False: {cfg.short_term.enabled}")
    if cfg.long_term.enabled is not False:
        _fail(f"long_term.enabled 默认应为 False: {cfg.long_term.enabled}")
    if cfg.enabled is not False:
        _fail(f"整体 enabled 默认应为 False: {cfg.enabled}")
    # 默认阈值
    if cfg.short_term.max_raw_turns != 20:
        _fail(f"max_raw_turns 默认应为 20: {cfg.short_term.max_raw_turns}")
    if cfg.short_term.keep_recent_turns != 10:
        _fail(f"keep_recent_turns 默认应为 10: {cfg.short_term.keep_recent_turns}")
    if cfg.long_term.graph_database != "user_memory":
        _fail(f"graph_database 默认应为 user_memory: {cfg.long_term.graph_database}")
    if cfg.long_term.idle_timeout_minutes != 30:
        _fail(f"idle_timeout_minutes 默认应为 30: {cfg.long_term.idle_timeout_minutes}")
    _ok("默认全部关闭 + 阈值正确（等价基线，SPEC §6.4）")


def test_short_term_from_dict() -> None:
    """测试 2：ShortTermMemoryConfig 自定义阈值。"""
    print("[测试 2] ShortTermMemoryConfig.from_dict")
    st = ShortTermMemoryConfig.from_dict({
        "enabled": True,
        "max_raw_turns": 30,
        "keep_recent_turns": 15,
        "llm": "rag",
    })
    if st.enabled is not True:
        _fail(f"enabled 应为 True: {st.enabled}")
    if st.max_raw_turns != 30:
        _fail(f"max_raw_turns 应为 30: {st.max_raw_turns}")
    if st.keep_recent_turns != 15:
        _fail(f"keep_recent_turns 应为 15: {st.keep_recent_turns}")
    if st.llm != "rag":
        _fail(f"llm 应为 rag: {st.llm}")
    _ok("短期记忆自定义阈值解析正确")


def test_long_term_from_dict() -> None:
    """测试 3：LongTermMemoryConfig 自定义 database + 开关。"""
    print("[测试 3] LongTermMemoryConfig.from_dict")
    lt = LongTermMemoryConfig.from_dict({
        "enabled": True,
        "idle_timeout_minutes": 60,
        "graph_database": "custom_mem",
        "llm": "command",
        "realtime_extract": False,
        "end_of_session_extract": True,
    })
    if lt.enabled is not True:
        _fail(f"enabled 应为 True: {lt.enabled}")
    if lt.idle_timeout_minutes != 60:
        _fail(f"idle_timeout_minutes 应为 60: {lt.idle_timeout_minutes}")
    if lt.graph_database != "custom_mem":
        _fail(f"graph_database 应为 custom_mem: {lt.graph_database}")
    if lt.realtime_extract is not False:
        _fail(f"realtime_extract 应为 False: {lt.realtime_extract}")
    if lt.end_of_session_extract is not True:
        _fail(f"end_of_session_extract 应为 True: {lt.end_of_session_extract}")
    _ok("长期记忆自定义 database + 开关解析正确")


def test_full_memory_config() -> None:
    """测试 4：MemoryConfig.from_dict 完整 memory 节。"""
    print("[测试 4] MemoryConfig.from_dict 完整解析")
    cfg = MemoryConfig.from_dict({
        "short_term": {
            "enabled": True,
            "max_raw_turns": 25,
            "keep_recent_turns": 12,
        },
        "long_term": {
            "enabled": True,
            "graph_database": "user_memory",
            "realtime_extract": True,
        },
    })
    if not cfg.short_term.enabled:
        _fail("short_term.enabled 应为 True")
    if cfg.short_term.max_raw_turns != 25:
        _fail(f"max_raw_turns 应为 25: {cfg.short_term.max_raw_turns}")
    if not cfg.long_term.enabled:
        _fail("long_term.enabled 应为 True")
    if not cfg.long_term.realtime_extract:
        _fail("realtime_extract 应为 True")
    _ok("完整 memory 节解析正确（短期+长期同时启用）")


def test_enabled_property() -> None:
    """测试 5：MemoryConfig.enabled 任一启用即为 True。"""
    print("[测试 5] MemoryConfig.enabled 聚合")
    # 仅短期启用
    cfg1 = MemoryConfig(
        short_term=ShortTermMemoryConfig(enabled=True),
        long_term=LongTermMemoryConfig(enabled=False),
    )
    if cfg1.enabled is not True:
        _fail("仅短期启用时 enabled 应为 True")
    # 仅长期启用
    cfg2 = MemoryConfig(
        short_term=ShortTermMemoryConfig(enabled=False),
        long_term=LongTermMemoryConfig(enabled=True),
    )
    if cfg2.enabled is not True:
        _fail("仅长期启用时 enabled 应为 True")
    # 全关
    cfg3 = MemoryConfig()
    if cfg3.enabled is not False:
        _fail("全关时 enabled 应为 False")
    _ok("enabled 聚合逻辑正确（任一启用→True，全关→False）")


def test_env_var_substitution() -> None:
    """测试 6：环境变量替换在 memory 配置中生效。"""
    print("[测试 6] 环境变量替换")
    os.environ["TEST_MEM_DB"] = "env_mem_db"
    try:
        lt = LongTermMemoryConfig.from_dict({
            "graph_database": "${TEST_MEM_DB}",
        })
        if lt.graph_database != "env_mem_db":
            _fail(f"graph_database 应被环境变量替换为 env_mem_db: {lt.graph_database}")
        _ok("环境变量 ${TEST_MEM_DB} 替换生效")
    finally:
        del os.environ["TEST_MEM_DB"]


def main() -> int:
    print("=" * 60)
    print("M2.1 MemoryConfig 配置数据类 单元测试")
    print("=" * 60)
    print()
    tests = [
        test_default_disabled,
        test_short_term_from_dict,
        test_long_term_from_dict,
        test_full_memory_config,
        test_enabled_property,
        test_env_var_substitution,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M2.1 MemoryConfig 就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

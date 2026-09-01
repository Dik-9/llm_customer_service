# -*- coding: utf-8 -*-
"""
M2.5 验证脚本：ShortTermCompressor 历史压缩

确定性验证（注入 _StubLLM + _FakeTracker，不连真实 LLM）：
1. should_compress：轮次 <= 阈值 → False
2. should_compress：轮次 > 阈值 → True
3. compress：22 轮 → 压缩前 12 轮，保留最近 10 轮，session_summary 槽有内容
4. compress：summary_covered_turns 累计正确（from_turn/to_turn）
5. compress：dialogue_turns 物理裁剪到 keep_recent
6. compress：无需压缩 → compressed=False
7. compress：LLM 失败 → 不裁剪不写槽位，compressed=False
8. compress：LLM 返回空 → compressed=False
9. 多次压缩：summary_covered_turns 单调递增 + 新旧摘要合并
10. _turn_to_text：正确提取 user/bot 文本
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atguigu_ai.memory.short_term.compressor import (
    ShortTermCompressor,
    SLOT_SESSION_SUMMARY,
    SLOT_SUMMARY_COVERED_TURNS,
)
from atguigu_ai.shared.llm.base_client import LLMClient, LLMResponse


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- Stub LLM ----------

class _StubLLM(LLMClient):
    def __init__(self, content: str = "压缩摘要", exc: Optional[Exception] = None) -> None:
        super().__init__(model="stub", api_key="stub")
        self._content = content
        self._exc = exc
        self.call_count = 0
        self.last_prompt: str = ""

    async def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        self.last_prompt = messages[0]["content"] if messages else ""
        if self._exc is not None:
            raise self._exc
        return LLMResponse(content=self._content, model="stub")

    def complete_sync(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self._content, model="stub")


# ---------- Fake tracker ----------

class _FakeMsg:
    def __init__(self, text: str) -> None:
        self.text = text

class _FakeTurn:
    def __init__(self, user: str, bot: str) -> None:
        self.user_message = _FakeMsg(user)
        self.bot_messages = [_FakeMsg(bot)] if bot else []

class _FakeTracker:
    def __init__(self, turns: List[_FakeTurn]) -> None:
        self.dialogue_turns = list(turns)
        self._slots: Dict[str, Any] = {}

    def get_slot(self, name: str) -> Any:
        return self._slots.get(name)

    def set_slot(self, name: str, value: Any) -> None:
        self._slots[name] = value


def _make_turns(n: int) -> List[_FakeTurn]:
    return [_FakeTurn(f"用户消息{i}", f"机器人回复{i}") for i in range(n)]


# ---------- 测试 ----------

def test_should_compress_false() -> None:
    print("[测试 1] should_compress 轮次不足")
    cmp = ShortTermCompressor(_StubLLM(), max_raw_turns=20, keep_recent_turns=10)
    tr = _FakeTracker(_make_turns(15))
    if cmp.should_compress(tr):
        _fail("15 轮 <= 20 阈值不应压缩")
    _ok("轮次 <= 阈值 → False")


def test_should_compress_true() -> None:
    print("[测试 2] should_compress 轮次超阈值")
    cmp = ShortTermCompressor(_StubLLM(), max_raw_turns=20, keep_recent_turns=10)
    tr = _FakeTracker(_make_turns(22))
    if not cmp.should_compress(tr):
        _fail("22 轮 > 20 阈值应压缩")
    _ok("轮次 > 阈值 → True")


def test_compress_basic() -> None:
    print("[测试 3] compress 22 轮 → 压缩前12 保留后10")
    llm = _StubLLM(content="用户查询了多个订单，要求用顺丰配送")
    cmp = ShortTermCompressor(llm, max_raw_turns=20, keep_recent_turns=10)
    tr = _FakeTracker(_make_turns(22))
    result = asyncio.run(cmp.compress(tr))

    if not result.compressed:
        _fail("应 compressed=True")
    if result.summary != "用户查询了多个订单，要求用顺丰配送":
        _fail(f"summary 错: {result.summary}")
    # 槽位写入
    if tr.get_slot(SLOT_SESSION_SUMMARY) != result.summary:
        _fail(f"session_summary 槽未写入: {tr.get_slot(SLOT_SESSION_SUMMARY)}")
    if tr.get_slot(SLOT_SUMMARY_COVERED_TURNS) != 12:
        _fail(f"summary_covered_turns 应为 12: {tr.get_slot(SLOT_SUMMARY_COVERED_TURNS)}")
    # 物理裁剪到 10
    if len(tr.dialogue_turns) != 10:
        _fail(f"dialogue_turns 应裁剪到 10: {len(tr.dialogue_turns)}")
    # 保留的是最后 10 轮
    last = tr.dialogue_turns[-1].user_message.text
    if last != "用户消息21":
        _fail(f"应保留最后 10 轮（12-21），最后一条: {last}")
    # event
    if not result.event or result.event.get("event") != "memory_compressed":
        _fail(f"event 错: {result.event}")
    _ok("压缩 12 轮 → session_summary 写入 + dialogue_turns 裁剪到 10 + event 正确")


def test_compress_from_to_turn() -> None:
    print("[测试 4] compress from_turn/to_turn")
    llm = _StubLLM(content="摘要")
    cmp = ShortTermCompressor(llm, max_raw_turns=20, keep_recent_turns=10)
    tr = _FakeTracker(_make_turns(22))
    result = asyncio.run(cmp.compress(tr))
    if result.from_turn != 1 or result.to_turn != 12:
        _fail(f"from_turn=1 to_turn=12 错: from={result.from_turn} to={result.to_turn}")
    _ok("from_turn=1 / to_turn=12（累计原始轮次）")


def test_compress_noop() -> None:
    print("[测试 5] compress 无需压缩")
    llm = _StubLLM()
    cmp = ShortTermCompressor(llm, max_raw_turns=20, keep_recent_turns=10)
    tr = _FakeTracker(_make_turns(15))
    result = asyncio.run(cmp.compress(tr))
    if result.compressed:
        _fail("15 轮不应压缩")
    if llm.call_count != 0:
        _fail(f"不应调 LLM: {llm.call_count}")
    _ok("轮次不足 → compressed=False，不调 LLM")


def test_compress_llm_failure() -> None:
    print("[测试 6] compress LLM 失败容错")
    llm = _StubLLM(exc=RuntimeError("llm down"))
    cmp = ShortTermCompressor(llm, max_raw_turns=20, keep_recent_turns=10)
    tr = _FakeTracker(_make_turns(22))
    original_len = len(tr.dialogue_turns)
    result = asyncio.run(cmp.compress(tr))
    if result.compressed:
        _fail("LLM 失败应 compressed=False")
    # 不裁剪、不写槽位
    if len(tr.dialogue_turns) != original_len:
        _fail(f"LLM 失败不应裁剪: {len(tr.dialogue_turns)} != {original_len}")
    if tr.get_slot(SLOT_SESSION_SUMMARY) is not None:
        _fail("LLM 失败不应写 session_summary")
    _ok("LLM 失败 → 不裁剪/不写槽位/compressed=False（保障主链路）")


def test_compress_empty_summary() -> None:
    print("[测试 7] compress LLM 返回空")
    llm = _StubLLM(content="   ")
    cmp = ShortTermCompressor(llm, max_raw_turns=20, keep_recent_turns=10)
    tr = _FakeTracker(_make_turns(22))
    result = asyncio.run(cmp.compress(tr))
    if result.compressed:
        _fail("空摘要应 compressed=False")
    if len(tr.dialogue_turns) != 22:
        _fail("空摘要不应裁剪")
    _ok("LLM 返回空 → compressed=False，不裁剪")


def test_compress_multiple_rounds() -> None:
    print("[测试 8] 多次压缩 summary_covered_turns 递增")
    llm = _StubLLM(content="合并后的摘要")
    cmp = ShortTermCompressor(llm, max_raw_turns=20, keep_recent_turns=10)
    tr = _FakeTracker(_make_turns(22))

    # 第一次压缩：22 轮 → 压缩 12，保留 10，covered=12
    r1 = asyncio.run(cmp.compress(tr))
    if r1.to_turn != 12:
        _fail(f"第一次 to_turn 应 12: {r1.to_turn}")
    if tr.get_slot(SLOT_SUMMARY_COVERED_TURNS) != 12:
        _fail(f"第一次 covered 应 12")

    # 模拟又新增 13 轮（10 + 13 = 23 > 20）→ 再压缩 13，保留 10，covered=25
    tr.dialogue_turns.extend(_make_turns_from(22, 13))
    r2 = asyncio.run(cmp.compress(tr))
    if not r2.compressed:
        _fail("第二次应压缩")
    if r2.from_turn != 13 or r2.to_turn != 25:
        _fail(f"第二次 from=13 to=25 错: from={r2.from_turn} to={r2.to_turn}")
    if tr.get_slot(SLOT_SUMMARY_COVERED_TURNS) != 25:
        _fail(f"第二次 covered 应 25: {tr.get_slot(SLOT_SUMMARY_COVERED_TURNS)}")
    if len(tr.dialogue_turns) != 10:
        _fail(f"第二次后应保留 10 轮: {len(tr.dialogue_turns)}")
    _ok("多次压缩 → covered 单调递增（12→25），每次保留 10 轮")


def _make_turns_from(start: int, n: int) -> List[_FakeTurn]:
    return [_FakeTurn(f"用户消息{start + i}", f"机器人回复{start + i}") for i in range(n)]


def test_turn_to_text() -> None:
    print("[测试 9] _turn_to_text 文本提取")
    llm = _StubLLM()
    cmp = ShortTermCompressor(llm)
    turn = _FakeTurn("你好", "您好有什么帮您")
    d = cmp._turn_to_text(turn)
    if d.get("user") != "你好" or d.get("bot") != "您好有什么帮您":
        _fail(f"文本提取错: {d}")
    # 多条 bot 消息
    turn2 = _FakeTurn("问", "")
    turn2.bot_messages = [_FakeMsg("回复1"), _FakeMsg("回复2")]
    d2 = cmp._turn_to_text(turn2)
    if d2.get("bot") != "回复1 / 回复2":
        _fail(f"多 bot 消息应用 / 连接: {d2.get('bot')}")
    _ok("_turn_to_text → user/bot 文本正确（多条 bot 用 / 连接）")


def main() -> int:
    print("=" * 60)
    print("M2.5 ShortTermCompressor 历史压缩 单元测试")
    print("=" * 60)
    print()
    tests = [
        test_should_compress_false,
        test_should_compress_true,
        test_compress_basic,
        test_compress_from_to_turn,
        test_compress_noop,
        test_compress_llm_failure,
        test_compress_empty_summary,
        test_compress_multiple_rounds,
        test_turn_to_text,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M2.5 ShortTermCompressor 就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
M2.6 验证脚本：MemoryHooks 三处挂接点编排

确定性验证（注入 fake recaller/extractor/compressor/graph_store + fake tracker）：
1. before_understand：召回画像/提及 → 写 tracker.memory_context
2. before_understand：指代消歧命中 → 写 order_id 槽
3. before_understand：指代消歧命中但 order_id 已有值 → 不覆盖
4. before_understand：短期 session_summary 注入 memory_context
5. after_each_action：实时抽取 → 写图谱（preference/address/order_ref）
6. after_each_action：无记忆信号 → 不抽取（extractor 粗筛）
7. before_save_tracker：短期压缩触发 → 调 compressor
8. before_save_tracker：会话结束信号 → 兜底抽取写图谱
9. before_save_tracker：无结束信号 → 不兜底抽取
10. memory 关闭（enabled=false）→ 全部 no-op
11. 模块级函数 hooks=None → no-op（基线等价）
12. 组件异常容错：recaller 抛错 → 降级空 ctx，不抛出
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atguigu_ai.memory.hooks import MemoryHooks, before_understand, after_each_action, before_save_tracker
from atguigu_ai.memory.long_term.extractor import ExtractedFact
from atguigu_ai.memory.short_term.compressor import CompressionResult, SLOT_SESSION_SUMMARY
from atguigu_ai.shared.config import MemoryConfig, ShortTermMemoryConfig, LongTermMemoryConfig


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- Fakes ----------

class _FakeMsg:
    def __init__(self, text: str) -> None:
        self.text = text

class _FakeTurn:
    def __init__(self, user: str, bot: str = "") -> None:
        self.user_message = _FakeMsg(user)
        self.bot_messages = [_FakeMsg(bot)] if bot else []

class _FakeTracker:
    def __init__(self, user_id: str = "1001", latest: str = "你好", turns: Optional[List[_FakeTurn]] = None) -> None:
        self.sender_id = user_id
        self.latest_message = _FakeMsg(latest)
        self.dialogue_turns = turns or []
        self._slots: Dict[str, Any] = {"user_id": user_id}
        self.memory_context: Dict[str, Any] = {}

    def get_slot(self, name: str) -> Any:
        return self._slots.get(name)

    def set_slot(self, name: str, value: Any) -> None:
        self._slots[name] = value


class _FakeRecaller:
    def __init__(self, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None) -> None:
        self._ctx = ctx or {"profile_text": "", "mentions_text": "", "resolved_order_id": None}
        self._exc = exc

    def get_memory_context(self, user_id: str, message: str) -> Dict[str, Any]:
        if self._exc:
            raise self._exc
        return dict(self._ctx)


class _FakeExtractor:
    def __init__(self, facts: Optional[List[ExtractedFact]] = None, realtime_exc: Optional[Exception] = None) -> None:
        self._facts = facts or []
        self._realtime_exc = realtime_exc
        self.realtime_calls = 0
        self.session_calls = 0

    async def extract_realtime(self, latest_message: str, recent_turns: Optional[List] = None) -> List[ExtractedFact]:
        self.realtime_calls += 1
        if self._realtime_exc:
            raise self._realtime_exc
        return list(self._facts)

    async def extract_end_of_session(self, all_turns: List) -> List[ExtractedFact]:
        self.session_calls += 1
        return list(self._facts)


class _FakeCompressor:
    def __init__(self, should: bool = True, result: Optional[CompressionResult] = None) -> None:
        self._should = should
        self._result = result or CompressionResult(compressed=True, summary="压缩摘要", from_turn=1, to_turn=12)
        self.compress_calls = 0

    def should_compress(self, tracker: Any) -> bool:
        return self._should

    async def compress(self, tracker: Any) -> CompressionResult:
        self.compress_calls += 1
        return self._result


class _FakeGraphStore:
    def __init__(self) -> None:
        self.preferences: List[Dict[str, Any]] = []
        self.addresses: List[Dict[str, Any]] = []
        self.mentions: List[Dict[str, Any]] = []

    def upsert_preference(self, user_id, pref_type, value, confidence=0.8, source="realtime") -> None:
        self.preferences.append({"user_id": user_id, "type": pref_type, "value": value, "confidence": confidence, "source": source})

    def upsert_address(self, user_id, address) -> None:
        self.addresses.append({"user_id": user_id, "address": address})

    def add_order_mention(self, user_id, order_id, context="", turn_id=None) -> None:
        self.mentions.append({"user_id": user_id, "order_id": order_id, "context": context})

    def close(self) -> None:
        pass


def _cfg(short=False, long=False, realtime=True, eos=True) -> MemoryConfig:
    return MemoryConfig(
        short_term=ShortTermMemoryConfig(enabled=short),
        long_term=LongTermMemoryConfig(enabled=long, realtime_extract=realtime, end_of_session_extract=eos),
    )


# ---------- 测试 ----------

def test_before_understand_recall() -> None:
    print("[测试 1] before_understand 召回 → memory_context")
    rc = _FakeRecaller(ctx={"profile_text": "偏好：快递=顺丰", "mentions_text": "提及 O001", "resolved_order_id": None})
    h = MemoryHooks(_cfg(long=True), recaller=rc)
    tr = _FakeTracker(latest="查询订单")
    ctx = asyncio.run(h.before_understand(tr))
    if ctx["profile_text"] != "偏好：快递=顺丰":
        _fail(f"profile_text 错: {ctx}")
    if ctx["mentions_text"] != "提及 O001":
        _fail(f"mentions_text 错: {ctx}")
    if tr.memory_context.get("profile_text") != "偏好：快递=顺丰":
        _fail(f"tracker.memory_context 未写入: {tr.memory_context}")
    _ok("召回 → memory_context 写入 tracker")


def test_before_understand_coreference_set_slot() -> None:
    print("[测试 2] 指代消歧 → 写 order_id 槽")
    rc = _FakeRecaller(ctx={"profile_text": "", "mentions_text": "", "resolved_order_id": "O001"})
    h = MemoryHooks(_cfg(long=True), recaller=rc)
    tr = _FakeTracker(latest="改下上次那个订单的地址")
    asyncio.run(h.before_understand(tr))
    if tr.get_slot("order_id") != "O001":
        _fail(f"order_id 应被消歧写入 O001: {tr.get_slot('order_id')}")
    _ok("指代消歧命中 → order_id 槽写入 O001")


def test_before_understand_coreference_no_overwrite() -> None:
    print("[测试 3] order_id 已有值 → 不覆盖")
    rc = _FakeRecaller(ctx={"profile_text": "", "mentions_text": "", "resolved_order_id": "O002"})
    h = MemoryHooks(_cfg(long=True), recaller=rc)
    tr = _FakeTracker(latest="改上次那个订单")
    tr.set_slot("order_id", "O001")  # 已有值
    asyncio.run(h.before_understand(tr))
    if tr.get_slot("order_id") != "O001":
        _fail(f"已有 order_id 不应被覆盖: {tr.get_slot('order_id')}")
    _ok("order_id 已有值 → 不覆盖")


def test_before_understand_session_summary() -> None:
    print("[测试 4] session_summary 注入 memory_context")
    h = MemoryHooks(_cfg(short=True, long=True), recaller=_FakeRecaller())
    tr = _FakeTracker(latest="你好")
    tr.set_slot(SLOT_SESSION_SUMMARY, "历史摘要内容")
    ctx = asyncio.run(h.before_understand(tr))
    if ctx["session_summary"] != "历史摘要内容":
        _fail(f"session_summary 未注入: {ctx}")
    _ok("短期 session_summary → memory_context")


def test_after_each_action_writes_facts() -> None:
    print("[测试 5] after_each_action 实时抽取 → 写图谱")
    facts = [
        ExtractedFact(kind="preference", data={"type": "快递公司", "value": "顺丰"}, confidence=0.9),
        ExtractedFact(kind="address", data={"province": "北京", "city": "北京", "district": "朝阳区"}),
        ExtractedFact(kind="order_ref", data={"order_id": "O001", "context": ""}),
    ]
    gs = _FakeGraphStore()
    h = MemoryHooks(_cfg(long=True), extractor=_FakeExtractor(facts=facts), graph_store=gs)
    tr = _FakeTracker(latest="记住用顺丰")
    written = asyncio.run(h.after_each_action(tr))
    if len(written) != 3:
        _fail(f"应写 3 条: {len(written)}")
    if len(gs.preferences) != 1 or gs.preferences[0]["value"] != "顺丰":
        _fail(f"preference 未写: {gs.preferences}")
    if len(gs.addresses) != 1:
        _fail(f"address 未写: {gs.addresses}")
    if len(gs.mentions) != 1 or gs.mentions[0]["order_id"] != "O001":
        _fail(f"order_ref 未写: {gs.mentions}")
    _ok("实时抽取 → preference/address/order_ref 全部写图谱")


def test_after_each_action_no_signal() -> None:
    print("[测试 6] 无记忆信号 → extractor 不返回事实")
    # _FakeExtractor 不做粗筛，直接返回 facts；这里测 hooks 在 enabled 时正常调用
    # 真实粗筛在 MemoryExtractor 内部（已在 M2.3 验证）。这里验证空 facts 不写图谱
    gs = _FakeGraphStore()
    h = MemoryHooks(_cfg(long=True), extractor=_FakeExtractor(facts=[]), graph_store=gs)
    tr = _FakeTracker(latest="查询订单")
    written = asyncio.run(h.after_each_action(tr))
    if written:
        _fail(f"空 facts 应返回空: {written}")
    if gs.preferences or gs.addresses or gs.mentions:
        _fail("空 facts 不应写图谱")
    _ok("无事实 → 不写图谱")


def test_before_save_tracker_compress() -> None:
    print("[测试 7] before_save_tracker 短期压缩")
    cmp = _FakeCompressor(should=True)
    h = MemoryHooks(_cfg(short=True), compressor=cmp)
    tr = _FakeTracker(latest="你好")
    res = asyncio.run(h.before_save_tracker(tr))
    if cmp.compress_calls != 1:
        _fail(f"应调 1 次 compress: {cmp.compress_calls}")
    if not res["compression"].compressed:
        _fail(f"compression.compressed 应为 True: {res['compression']}")
    _ok("短期压缩触发 → compressor.compress 调用")

    # 不需压缩
    cmp2 = _FakeCompressor(should=False)
    h2 = MemoryHooks(_cfg(short=True), compressor=cmp2)
    res2 = asyncio.run(h2.before_save_tracker(tr2 := _FakeTracker()))
    if cmp2.compress_calls != 0:
        _fail("should_compress=False 不应调 compress")
    _ok("should_compress=False → 不调 compress")


def test_before_save_tracker_session_end() -> None:
    print("[测试 8] 会话结束信号 → 兜底抽取")
    facts = [ExtractedFact(kind="preference", data={"type": "快递", "value": "顺丰"}, confidence=1.0)]
    gs = _FakeGraphStore()
    ext = _FakeExtractor(facts=facts)
    h = MemoryHooks(_cfg(long=True), extractor=ext, graph_store=gs)
    tr = _FakeTracker(latest="再见", turns=[_FakeTurn("查询", "好的")])
    res = asyncio.run(h.before_save_tracker(tr))
    if not res["session_ended"]:
        _fail("应检测到会话结束")
    if ext.session_calls != 1:
        _fail(f"应调 1 次兜底抽取: {ext.session_calls}")
    if len(gs.preferences) != 1:
        _fail(f"兜底事实应写图谱: {gs.preferences}")
    _ok("会话结束信号 → 兜底抽取 + 写图谱")


def test_before_save_tracker_no_session_end() -> None:
    print("[测试 9] 无结束信号 → 不兜底抽取")
    ext = _FakeExtractor(facts=[])
    h = MemoryHooks(_cfg(long=True), extractor=ext, graph_store=_FakeGraphStore())
    tr = _FakeTracker(latest="查询我的订单")
    res = asyncio.run(h.before_save_tracker(tr))
    if res["session_ended"]:
        _fail("无结束信号不应触发兜底")
    if ext.session_calls != 0:
        _fail(f"不应调兜底抽取: {ext.session_calls}")
    _ok("无结束信号 → 不兜底抽取")


def test_memory_disabled_noop() -> None:
    print("[测试 10] memory 关闭 → 全部 no-op")
    h = MemoryHooks(_cfg())  # 全关
    tr = _FakeTracker(latest="再见")
    ctx = asyncio.run(h.before_understand(tr))
    if ctx["profile_text"] or ctx["session_summary"]:
        _fail("关闭时 before_understand 应返回空 ctx")
    written = asyncio.run(h.after_each_action(tr))
    if written:
        _fail("关闭时 after_each_action 应返回空")
    res = asyncio.run(h.before_save_tracker(tr))
    if res["session_ended"] or res["compression"].compressed:
        _fail("关闭时 before_save_tracker 应全 no-op")
    _ok("memory 关闭 → 三处 hook 全 no-op（等价基线）")


def test_module_level_none_noop() -> None:
    print("[测试 11] 模块级函数 hooks=None → no-op")
    tr = _FakeTracker()
    ctx = asyncio.run(before_understand(tr, None))
    if ctx["profile_text"] != "":
        _fail("hooks=None 应返回空 ctx")
    written = asyncio.run(after_each_action(tr, None, None))
    if written:
        _fail("hooks=None 应返回空")
    res = asyncio.run(before_save_tracker(tr, None))
    if res["session_ended"]:
        _fail("hooks=None 应 no-op")
    _ok("模块级 hooks=None → no-op（基线等价，SPEC §6.4）")


def test_recaller_exception_tolerance() -> None:
    print("[测试 12] recaller 抛错 → 降级空 ctx，不抛出")
    rc = _FakeRecaller(exc=RuntimeError("neo4j down"))
    h = MemoryHooks(_cfg(long=True), recaller=rc)
    tr = _FakeTracker(latest="查询订单")
    ctx = asyncio.run(h.before_understand(tr))
    if ctx["profile_text"] or ctx["resolved_order_id"]:
        _fail(f"recaller 异常应降级空 ctx: {ctx}")
    _ok("recaller 异常 → 降级空 memory_context，不抛出（保障主链路）")


def main() -> int:
    print("=" * 60)
    print("M2.6 MemoryHooks 三处挂接点 单元测试")
    print("=" * 60)
    print()
    tests = [
        test_before_understand_recall,
        test_before_understand_coreference_set_slot,
        test_before_understand_coreference_no_overwrite,
        test_before_understand_session_summary,
        test_after_each_action_writes_facts,
        test_after_each_action_no_signal,
        test_before_save_tracker_compress,
        test_before_save_tracker_session_end,
        test_before_save_tracker_no_session_end,
        test_memory_disabled_noop,
        test_module_level_none_noop,
        test_recaller_exception_tolerance,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M2.6 MemoryHooks 就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
M2.4 验证脚本：MemoryRecaller 画像召回 + 指代消歧

确定性验证（注入 _FakeStore，不连真实 Neo4j）：
1. get_user_profile → 返回偏好 + 默认地址
2. format_profile_text → 生成画像文本（偏好+地址）
3. format_profile_text 无画像 → 空串
4. format_mentions_text → 生成提及文本
5. has_coreference_signal → 命中/不命中
6. resolve_coreference 有信号 → 返回最近 order_id
7. resolve_coreference 无信号 → None
8. resolve_coreference 无提及 → None
9. get_memory_context → 统一召回（profile/mentions/resolved 齐全）
10. 空 user_id → 全部返回空，不抛错
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atguigu_ai.memory.long_term.recaller import MemoryRecaller


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- Fake graph store ----------

class _FakeStore:
    """可编程的 graph_store 替身。"""
    def __init__(
        self,
        prefs: Optional[List[Dict[str, Any]]] = None,
        addr: Optional[Dict[str, Any]] = None,
        mentions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._prefs = prefs or []
        self._addr = addr
        self._mentions = mentions or []

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        return {"user_id": user_id, "preferences": self._prefs, "default_address": self._addr}

    def get_recent_mentions(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        return self._mentions[:limit]


# ---------- 测试 ----------

def test_get_user_profile() -> None:
    print("[测试 1] get_user_profile")
    store = _FakeStore(
        prefs=[{"type": "快递公司", "value": "顺丰", "confidence": 0.9}],
        addr={"label": "公司", "province": "北京", "city": "北京", "district": "朝阳区"},
    )
    rc = MemoryRecaller(store)
    p = rc.get_user_profile("1001")
    if len(p["preferences"]) != 1:
        _fail(f"应 1 条偏好: {p}")
    if p["default_address"].get("label") != "公司":
        _fail(f"默认地址错: {p['default_address']}")
    _ok("get_user_profile → 偏好 + 默认地址")


def test_format_profile_text_full() -> None:
    print("[测试 2] format_profile_text 画像+地址")
    store = _FakeStore(
        prefs=[{"type": "快递公司", "value": "顺丰"}, {"type": "支付方式", "value": "微信"}],
        addr={"label": "公司", "province": "北京", "city": "北京", "district": "朝阳区"},
    )
    rc = MemoryRecaller(store)
    text = rc.format_profile_text("1001")
    if "快递公司=顺丰" not in text or "支付方式=微信" not in text:
        _fail(f"偏好文本错: {text}")
    if "默认收货地址" not in text or "北京北京朝阳区" not in text:
        _fail(f"地址文本错: {text}")
    _ok(f"画像文本生成正确：{text[:50]}...")


def test_format_profile_text_empty() -> None:
    print("[测试 3] format_profile_text 无画像")
    rc = MemoryRecaller(_FakeStore())
    if rc.format_profile_text("1001") != "":
        _fail("无画像应返回空串")
    _ok("无画像 → 空串")


def test_format_mentions_text() -> None:
    print("[测试 4] format_mentions_text")
    store = _FakeStore(mentions=[
        {"order_id": "O002", "mentioned_at": "2026-09-01T12:00:00", "context": ""},
        {"order_id": "O001", "mentioned_at": "2026-09-01T10:00:00", "context": ""},
    ])
    rc = MemoryRecaller(store)
    text = rc.format_mentions_text("1001")
    if "O002" not in text or "O001" not in text:
        _fail(f"提及文本错: {text}")
    if "【参考记忆】" not in text:
        _fail(f"应含【参考记忆】前缀: {text}")
    _ok(f"提及文本生成正确：{text}")

    # 无提及
    rc2 = MemoryRecaller(_FakeStore())
    if rc2.format_mentions_text("1001") != "":
        _fail("无提及应返回空串")
    _ok("无提及 → 空串")


def test_coreference_signal() -> None:
    print("[测试 5] has_coreference_signal")
    hit = ["上次那个订单", "刚才说的那个", "就是那个", "那个订单", "这个订单帮我改下"]
    miss = ["查询我的订单", "帮我取消", "你好"]
    for t in hit:
        if not MemoryRecaller.has_coreference_signal(t):
            _fail(f"应命中指代: {t}")
    for t in miss:
        if MemoryRecaller.has_coreference_signal(t):
            _fail(f"不应命中指代: {t}")
    _ok("指代信号：5 命中 / 3 不命中 全部正确")


def test_resolve_coreference_hit() -> None:
    print("[测试 6] resolve_coreference 有信号")
    store = _FakeStore(mentions=[
        {"order_id": "O002", "context": ""},
        {"order_id": "O001", "context": ""},
    ])
    rc = MemoryRecaller(store)
    resolved = rc.resolve_coreference("1001", "帮我改下上次那个订单的地址")
    if resolved != "O002":
        _fail(f"应消歧到最近的 O002: {resolved}")
    _ok("指代消歧 → 返回最近提及 order_id（O002）")


def test_resolve_coreference_no_signal() -> None:
    print("[测试 7] resolve_coreference 无信号")
    store = _FakeStore(mentions=[{"order_id": "O001", "context": ""}])
    rc = MemoryRecaller(store)
    if rc.resolve_coreference("1001", "查询我的订单") is not None:
        _fail("无信号应返回 None")
    _ok("无指代信号 → None")


def test_resolve_coreference_no_mentions() -> None:
    print("[测试 8] resolve_coreference 无提及")
    rc = MemoryRecaller(_FakeStore())
    if rc.resolve_coreference("1001", "上次那个订单") is not None:
        _fail("无提及应返回 None")
    _ok("有信号但无提及 → None")


def test_get_memory_context() -> None:
    print("[测试 9] get_memory_context 统一召回")
    store = _FakeStore(
        prefs=[{"type": "快递公司", "value": "顺丰"}],
        addr={"label": "公司", "province": "北京", "city": "北京", "district": "朝阳区"},
        mentions=[{"order_id": "O001", "context": ""}],
    )
    rc = MemoryRecaller(store)
    ctx = rc.get_memory_context("1001", "改下上次那个订单")
    if "快递公司=顺丰" not in ctx["profile_text"]:
        _fail(f"profile_text 错: {ctx['profile_text']}")
    if "O001" not in ctx["mentions_text"]:
        _fail(f"mentions_text 错: {ctx['mentions_text']}")
    if ctx["resolved_order_id"] != "O001":
        _fail(f"resolved_order_id 应为 O001: {ctx['resolved_order_id']}")
    if len(ctx["preferences"]) != 1:
        _fail(f"preferences 错: {ctx['preferences']}")
    _ok("get_memory_context → profile/mentions/resolved 齐全")


def test_empty_user_id() -> None:
    print("[测试 10] 空 user_id 防御")
    rc = MemoryRecaller(_FakeStore())
    if rc.get_user_profile("")["preferences"] != []:
        _fail("空 user_id 应返回空偏好")
    if rc.get_recent_mentions("") != []:
        _fail("空 user_id 应返回空提及")
    if rc.format_profile_text("") != "":
        _fail("空 user_id 应返回空画像")
    if rc.resolve_coreference("", "上次那个") is not None:
        _fail("空 user_id 应返回 None")
    _ok("空 user_id → 全部返回空，不抛错")


def main() -> int:
    print("=" * 60)
    print("M2.4 MemoryRecaller 画像召回 + 指代消歧 单元测试")
    print("=" * 60)
    print()
    tests = [
        test_get_user_profile,
        test_format_profile_text_full,
        test_format_profile_text_empty,
        test_format_mentions_text,
        test_coreference_signal,
        test_resolve_coreference_hit,
        test_resolve_coreference_no_signal,
        test_resolve_coreference_no_mentions,
        test_get_memory_context,
        test_empty_user_id,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M2.4 MemoryRecaller 就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

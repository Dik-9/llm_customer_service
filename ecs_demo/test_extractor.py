# -*- coding: utf-8 -*-
"""
M2.3 验证脚本：MemoryExtractor LLM 结构化抽取

确定性验证（注入 _StubLLM，不连真实 LLM）：
1. has_memory_signal：正则粗筛命中/不命中
2. extract_realtime 无信号 → 不调 LLM 返回空（省 token）
3. extract_realtime 有信号 → 调 LLM 解析 preference
4. extract_realtime 解析 address（含字段校验）
5. extract_realtime 解析 order_ref
6. extract_realtime 无事实 → 空数组
7. extract_end_of_session → 对完整会话抽取多事实
8. LLM 调用抛错 → 返回空，不抛出（保障主链路）
9. JSON 容错：markdown 代码块 / icts / 多余文本
10. ExtractedFact.from_dict 字段校验 + confidence 截断
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atguigu_ai.memory.long_term.extractor import MemoryExtractor, ExtractedFact
from atguigu_ai.shared.llm.base_client import LLMClient, LLMResponse


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- Stub LLM ----------

class _StubLLM(LLMClient):
    """记录调用，返回预设 content。"""
    def __init__(self, content: str = "[]", exc: Optional[Exception] = None) -> None:
        super().__init__(model="stub", api_key="stub")
        self._content = content
        self._exc = exc
        self.call_count = 0
        self.last_messages: Optional[List[Dict[str, str]]] = None

    async def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages
        if self._exc is not None:
            raise self._exc
        return LLMResponse(content=self._content, model="stub")

    def complete_sync(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self._content, model="stub")


# ---------- 信号粗筛 ----------

def test_has_memory_signal() -> None:
    print("[测试 1] has_memory_signal 正则粗筛")
    hit_cases = [
        "记住我默认快递用顺丰",
        "以后都用京东物流",
        "我的地址是北京朝阳区建国路1号",
        "就是上次那个订单",
        "刚才说的那个",
        "别再问我手机号了",
    ]
    for t in hit_cases:
        if not MemoryExtractor.has_memory_signal(t):
            _fail(f"应命中信号: {t}")
    miss_cases = ["查询我的订单", "你好", "帮我取消订单", "12345"]
    for t in miss_cases:
        if MemoryExtractor.has_memory_signal(t):
            _fail(f"不应命中信号: {t}")
    _ok("正则粗筛：6 命中 / 4 不命中 全部正确")


# ---------- 实时抽取 ----------

def test_extract_realtime_no_signal() -> None:
    print("[测试 2] extract_realtime 无信号不调 LLM")
    llm = _StubLLM(content='[{"kind":"preference","type":"x","value":"y"}]')
    ext = MemoryExtractor(llm)
    facts = asyncio.run(ext.extract_realtime("查询我的订单", []))
    if facts:
        _fail(f"无信号应返回空: {facts}")
    if llm.call_count != 0:
        _fail(f"无信号不应调 LLM: call_count={llm.call_count}")
    _ok("无记忆信号 → 直接返回空，不调 LLM（省 token）")


def test_extract_realtime_preference() -> None:
    print("[测试 3] extract_realtime preference")
    llm = _StubLLM(content='[{"kind":"preference","type":"快递公司","value":"顺丰","confidence":0.9}]')
    ext = MemoryExtractor(llm)
    facts = asyncio.run(ext.extract_realtime("记住以后快递都用顺丰", []))
    if len(facts) != 1:
        _fail(f"应抽 1 条: {facts}")
    f = facts[0]
    if f.kind != "preference":
        _fail(f"kind 应为 preference: {f.kind}")
    if f.data.get("type") != "快递公司" or f.data.get("value") != "顺丰":
        _fail(f"偏好字段错: {f.data}")
    if f.confidence != 0.9:
        _fail(f"confidence 错: {f.confidence}")
    if llm.call_count != 1:
        _fail(f"应调 1 次 LLM: {llm.call_count}")
    _ok("实时抽取 preference：kind/type/value/confidence 正确")


def test_extract_realtime_address() -> None:
    print("[测试 4] extract_realtime address")
    llm = _StubLLM(content='[{"kind":"address","province":"北京","city":"北京","district":"朝阳区","street":"建国路1号","is_default":true}]')
    ext = MemoryExtractor(llm)
    facts = asyncio.run(ext.extract_realtime("我的地址是北京朝阳区建国路1号", []))
    if len(facts) != 1 or facts[0].kind != "address":
        _fail(f"应抽 1 条 address: {facts}")
    d = facts[0].data
    if d.get("province") != "北京" or d.get("district") != "朝阳区":
        _fail(f"地址字段错: {d}")
    if d.get("is_default") is not True:
        _fail(f"is_default 应为 True: {d.get('is_default')}")
    _ok("实时抽取 address：province/city/district/is_default 正确")


def test_extract_realtime_order_ref() -> None:
    print("[测试 5] extract_realtime order_ref")
    llm = _StubLLM(content='[{"kind":"order_ref","order_id":"O001","context":"查询订单详情"}]')
    ext = MemoryExtractor(llm)
    facts = asyncio.run(ext.extract_realtime("上次那个订单 O001 帮我改下地址", []))
    if len(facts) != 1 or facts[0].kind != "order_ref":
        _fail(f"应抽 1 条 order_ref: {facts}")
    if facts[0].data.get("order_id") != "O001":
        _fail(f"order_id 错: {facts[0].data}")
    _ok("实时抽取 order_ref：order_id/context 正确")


def test_extract_realtime_empty() -> None:
    print("[测试 6] extract_realtime 无事实")
    llm = _StubLLM(content="[]")
    ext = MemoryExtractor(llm)
    facts = asyncio.run(ext.extract_realtime("记住我的偏好", []))
    if facts:
        _fail(f"无事实应返回空: {facts}")
    _ok("LLM 返回 [] → 空事实列表")


# ---------- 会话结束兜底 ----------

def test_extract_end_of_session() -> None:
    print("[测试 7] extract_end_of_session 多事实")
    llm = _StubLLM(content='[{"kind":"preference","type":"快递公司","value":"顺丰","confidence":1.0},'
                          '{"kind":"order_ref","order_id":"O002","context":"取消订单"}]')
    ext = MemoryExtractor(llm)
    turns = [
        {"user": "查询订单", "bot": "请选择订单"},
        {"user": "记住以后用顺丰", "bot": "好的"},
        {"user": "取消 O002", "bot": "已取消"},
    ]
    facts = asyncio.run(ext.extract_end_of_session(turns))
    if len(facts) != 2:
        _fail(f"应抽 2 条: {facts}")
    kinds = {f.kind for f in facts}
    if kinds != {"preference", "order_ref"}:
        _fail(f"kind 集合错: {kinds}")
    _ok("会话结束兜底抽取：多事实（preference + order_ref）")


# ---------- 容错 ----------

def test_llm_exception() -> None:
    print("[测试 8] LLM 调用抛错容错")
    llm = _StubLLM(exc=RuntimeError("llm timeout"))
    ext = MemoryExtractor(llm)
    facts = asyncio.run(ext.extract_realtime("记住用顺丰", []))
    if facts:
        _fail(f"LLM 异常应返回空: {facts}")
    _ok("LLM 异常 → 返回空列表，不抛出")


def test_json_robustness() -> None:
    print("[测试 9] JSON 容错解析")
    # markdown 代码块
    llm1 = _StubLLM(content='```json\n[{"kind":"preference","type":"支付方式","value":"微信"}]\n```')
    ext1 = MemoryExtractor(llm1)
    f1 = asyncio.run(ext1.extract_realtime("记住默认用微信支付", []))
    if len(f1) != 1 or f1[0].data.get("value") != "微信":
        _fail(f"markdown 代码块解析错: {f1}")
    # 含思考标签 + 多余文本
    llm2 = _StubLLM(content='好的我来分析\n[{"kind":"preference","type":"快递公司","value":"京东"}]\n以上是结果')
    ext2 = MemoryExtractor(llm2)
    f2 = asyncio.run(ext2.extract_realtime("以后用京东", []))
    if len(f2) != 1 or f2[0].data.get("value") != "京东":
        _fail(f"多余文本解析错: {f2}")
    _ok("JSON 容错：markdown 代码块 / 多余文本 均可解析")


def test_fact_from_dict_validation() -> None:
    print("[测试 10] ExtractedFact.from_dict 字段校验")
    # confidence 截断
    f1 = ExtractedFact.from_dict({"kind": "preference", "type": "x", "value": "y", "confidence": 1.5})
    if f1 is None or f1.confidence != 1.0:
        _fail(f"confidence 应截断到 1.0: {f1}")
    f2 = ExtractedFact.from_dict({"kind": "preference", "type": "x", "value": "y", "confidence": -0.5})
    if f2 is None or f2.confidence != 0.0:
        _fail(f"confidence 应截断到 0.0: {f2}")
    # address 缺 district → None
    f3 = ExtractedFact.from_dict({"kind": "address", "province": "北京", "city": "北京"})
    if f3 is not None:
        _fail(f"address 缺 district 应返回 None: {f3}")
    # order_ref 缺 order_id → None
    f4 = ExtractedFact.from_dict({"kind": "order_ref", "context": "x"})
    if f4 is not None:
        _fail(f"order_ref 缺 order_id 应返回 None: {f4}")
    # preference 缺 value → None
    f5 = ExtractedFact.from_dict({"kind": "preference", "type": "x"})
    if f5 is not None:
        _fail(f"preference 缺 value 应返回 None: {f5}")
    # 字段推断：有 order_id 自动识别
    f6 = ExtractedFact.from_dict({"order_id": "O009", "context": ""})
    if f6 is None or f6.kind != "order_ref":
        _fail(f"字段推断 order_ref 失败: {f6}")
    _ok("字段校验：confidence 截断 / 必填校验 / 字段推断 全部正确")


def main() -> int:
    print("=" * 60)
    print("M2.3 MemoryExtractor LLM 结构化抽取 单元测试")
    print("=" * 60)
    print()
    tests = [
        test_has_memory_signal,
        test_extract_realtime_no_signal,
        test_extract_realtime_preference,
        test_extract_realtime_address,
        test_extract_realtime_order_ref,
        test_extract_realtime_empty,
        test_extract_end_of_session,
        test_llm_exception,
        test_json_robustness,
        test_fact_from_dict_validation,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M2.3 MemoryExtractor 就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
M2.2 验证脚本：GraphMemoryStore Neo4j 图谱 CRUD

确定性验证（注入 _FakeDriver，不连真实 Neo4j）：
1. ensure_user → MERGE User 节点 + 正确 user_id
2. upsert_preference 首次写入 → CREATE Preference + 偏好关系
3. upsert_preference 同 type 高置信度覆盖 → SET value/confidence 刷新
4. upsert_preference 低置信度不覆盖（query 含 WHERE $confidence >= p.confidence）
5. add_order_mention → CREATE OrderRef + 提及关系，记录 turn_id
6. upsert_address → MERGE Address + 常用地址关系
7. get_preferences → 解析返回偏好列表
8. get_default_address → 返回 is_default 排序首条
9. get_recent_mentions → 按 mentioned_at 倒序返回
10. database 隔离回退：_ensure_database_or_fallback 失败 → 默认 db + Mem 前缀
11. 异常容错：driver.execute_query 抛错 → 返回空列表，不抛出
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atguigu_ai.memory.long_term.graph_store import GraphMemoryStore


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- Fake Neo4j driver ----------

class _FakeResult:
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self.records = records


class _FakeDriver:
    """记录所有 execute_query 调用，按 query 关键字匹配返回预设记录。"""
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []  # {query, params, database}
        self._handlers: List = []  # [(substr, records)]

    def set_response(self, query_substr: str, records: List[Dict[str, Any]]) -> None:
        self._handlers.append((query_substr, records))

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None, database: Optional[str] = None, **kw: Any) -> _FakeResult:
        params = parameters or {}
        self.calls.append({"query": query, "params": params, "database": database})
        for substr, records in self._handlers:
            if substr in query:
                return _FakeResult(records)
        return _FakeResult([])

    def close(self) -> None:
        pass


class _ErrorDriver(_FakeDriver):
    """execute_query 恒抛错，用于容错测试。"""
    def execute_query(self, *a: Any, **kw: Any) -> _FakeResult:
        raise RuntimeError("neo4j connection refused")


def _store(driver: Optional[_FakeDriver] = None) -> GraphMemoryStore:
    """构造一个不触发 connect 的 store（跳过 _ensure_database_or_fallback）。"""
    d = driver if driver is not None else _FakeDriver()
    s = GraphMemoryStore.__new__(GraphMemoryStore)
    s._driver = d
    s._database = "user_memory"
    s._label_prefix = ""
    s._use_default_db = False
    return s


# ---------- 测试 ----------

def test_ensure_user() -> None:
    print("[测试 1] ensure_user")
    drv = _FakeDriver()
    store = _store(drv)
    store.ensure_user("1001")
    if len(drv.calls) != 1:
        _fail(f"应执行 1 次 Cypher: {len(drv.calls)}")
    q = drv.calls[0]
    if "MERGE" not in q["query"] or ":User" not in q["query"]:
        _fail(f"应为 MERGE User: {q['query'][:80]}")
    if q["params"].get("user_id") != "1001":
        _fail(f"user_id 参数错: {q['params']}")
    if q["database"] != "user_memory":
        _fail(f"database 应为 user_memory: {q['database']}")
    _ok("ensure_user → MERGE :User {user_id} + 正确 database")


def test_upsert_preference_create() -> None:
    print("[测试 2] upsert_preference 首次写入")
    drv = _FakeDriver()
    store = _store(drv)
    store.upsert_preference("1001", "快递公司", "顺丰", confidence=0.9, source="realtime")
    # ensure_user + upsert 共 2 次
    if len(drv.calls) != 2:
        _fail(f"应执行 2 次 Cypher（ensure_user + upsert）: {len(drv.calls)}")
    upsert_call = drv.calls[1]
    if "CREATE" not in upsert_call["query"]:
        _fail(f"首次写入应有 CREATE 分支: {upsert_call['query'][:80]}")
    p = upsert_call["params"]
    if p.get("type") != "快递公司" or p.get("value") != "顺丰":
        _fail(f"偏好参数错: {p}")
    if p.get("confidence") != 0.9 or p.get("source") != "realtime":
        _fail(f"置信度/source 错: {p}")
    _ok("upsert_preference 首次 → CREATE :Preference + 偏好关系")


def test_upsert_preference_merge_high_confidence() -> None:
    print("[测试 3] upsert_preference 同 type 高置信度覆盖")
    drv = _FakeDriver()
    store = _store(drv)
    # 让 OPTIONAL MATCH 命中已存在节点（fake 不真正执行 FOREACH 逻辑，
    # 这里只校验生成的 Cypher 包含 SET 分支与 WHERE 覆盖条件）
    store.upsert_preference("1001", "快递公司", "京东物流", confidence=0.95, source="session_end")
    upsert_call = drv.calls[1]
    if "SET p.value" not in upsert_call["query"]:
        _fail(f"应含 SET 覆盖分支: {upsert_call['query'][:120]}")
    if "$confidence >= p.confidence" not in upsert_call["query"]:
        _fail(f"应含置信度覆盖条件: {upsert_call['query'][:120]}")
    if upsert_call["params"].get("value") != "京东物流":
        _fail(f"value 应为新值: {upsert_call['params']}")
    _ok("upsert_preference 同 type → SET 覆盖（含 $confidence >= p.confidence 条件）")


def test_upsert_preference_empty_skipped() -> None:
    print("[测试 4] upsert_preference 空 user_id/pref_type 跳过")
    drv = _FakeDriver()
    store = _store(drv)
    store.upsert_preference("", "快递", "顺丰")
    store.upsert_preference("1001", "", "顺丰")
    if drv.calls:
        _fail(f"空参数应跳过不执行 Cypher: {len(drv.calls)}")
    _ok("空 user_id / pref_type → 直接跳过（防御）")


def test_add_order_mention() -> None:
    print("[测试 5] add_order_mention")
    drv = _FakeDriver()
    store = _store(drv)
    store.add_order_mention("1001", "O001", context="查询订单详情", turn_id=3)
    calls = [c for c in drv.calls if "OrderRef" in c["query"]]
    if len(calls) != 1:
        _fail(f"应 1 次 OrderRef 写入: {len(calls)}")
    c = calls[0]
    if "CREATE (o:" not in c["query"] or ":提及" not in c["query"]:
        _fail(f"应 CREATE OrderRef + 提及关系: {c['query'][:100]}")
    if c["params"].get("order_id") != "O001" or c["params"].get("turn_id") != 3:
        _fail(f"参数错: {c['params']}")
    _ok("add_order_mention → CREATE :OrderRef + :提及{turn_id}")


def test_upsert_address() -> None:
    print("[测试 6] upsert_address")
    drv = _FakeDriver()
    store = _store(drv)
    store.upsert_address("1001", {
        "label": "公司", "province": "北京", "city": "北京",
        "district": "朝阳区", "street": "建国路1号",
        "phone": "13800000000", "contact": "张三", "is_default": True,
    })
    addr_calls = [c for c in drv.calls if ":Address" in c["query"]]
    if len(addr_calls) != 1:
        _fail(f"应 1 次 Address 写入: {len(addr_calls)}")
    c = addr_calls[0]
    if "MERGE (a:" not in c["query"] or ":常用地址" not in c["query"]:
        _fail(f"应 MERGE Address + 常用地址关系: {c['query'][:100]}")
    if c["params"].get("is_default") is not True:
        _fail(f"is_default 应为 True: {c['params'].get('is_default')}")
    _ok("upsert_address → MERGE :Address + :常用地址")


def test_get_preferences() -> None:
    print("[测试 7] get_preferences")
    drv = _FakeDriver()
    drv.set_response(":偏好]->(p:Preference)", [
        {"type": "快递公司", "value": "顺丰", "confidence": 0.9, "source": "realtime", "updated_at": "2026-09-01"},
    ])
    store = _store(drv)
    prefs = store.get_preferences("1001")
    if len(prefs) != 1 or prefs[0].get("value") != "顺丰":
        _fail(f"偏好解析错: {prefs}")
    _ok("get_preferences → 解析返回偏好列表")


def test_get_default_address() -> None:
    print("[测试 8] get_default_address")
    drv = _FakeDriver()
    drv.set_response(":常用地址]->(a:Address)", [
        {"label": "公司", "province": "北京", "city": "北京", "district": "朝阳区",
         "street": "建国路1号", "phone": "138", "contact": "张三", "is_default": True},
    ])
    store = _store(drv)
    addr = store.get_default_address("1001")
    if not addr or addr.get("label") != "公司":
        _fail(f"默认地址解析错: {addr}")
    _ok("get_default_address → 返回 is_default 排序首条")


def test_get_recent_mentions() -> None:
    print("[测试 9] get_recent_mentions")
    drv = _FakeDriver()
    drv.set_response(":提及]->(o:OrderRef)", [
        {"order_id": "O002", "mentioned_at": "2026-09-01T12:00:00", "context": ""},
        {"order_id": "O001", "mentioned_at": "2026-09-01T10:00:00", "context": "查询详情"},
    ])
    store = _store(drv)
    mentions = store.get_recent_mentions("1001", limit=5)
    if len(mentions) != 2:
        _fail(f"应返回 2 条提及: {len(mentions)}")
    if mentions[0].get("order_id") != "O002":
        _fail(f"应按时间倒序 O002 在前: {mentions}")
    _ok("get_recent_mentions → 解析返回提及列表")


def test_database_fallback() -> None:
    print("[测试 10] database 隔离回退")
    drv = _FakeDriver()
    # CREATE DATABASE 在 system db 上返回空（模拟成功）→ 不回退
    store = GraphMemoryStore(driver=drv, database="user_memory")
    store._ensure_database_or_fallback()
    if store._use_default_db:
        _fail("system db 成功时不应回退")
    if store._label_prefix != "":
        _fail(f"成功时不应有前缀: {store._label_prefix}")
    _ok("database 创建成功 → 使用独立 database，无前缀")

    # 模拟失败：让 system db 调用抛错
    class _SystemFailDriver(_FakeDriver):
        def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None, database: Optional[str] = None, **kw: Any) -> _FakeResult:
            if database == "system":
                raise RuntimeError("Unsupported administration command")
            return super().execute_query(query, parameters, database=database, **kw)
    drv2 = _SystemFailDriver()
    store2 = GraphMemoryStore(driver=drv2, database="user_memory")
    store2._ensure_database_or_fallback()
    if not store2._use_default_db:
        _fail("system db 失败应回退默认 database")
    if store2._label_prefix != "Mem":
        _fail(f"回退后应有 Mem 前缀: {store2._label_prefix}")
    if store2._database is not None:
        _fail(f"回退后 database 应为 None: {store2._database}")
    _ok("database 创建失败 → 回退默认 database + 'Mem' 前缀（隔离 GraphRAG）")


def test_exception_tolerance() -> None:
    print("[测试 11] 异常容错")
    store = _store(_ErrorDriver())
    # 读取不应抛错，返回空
    prefs = store.get_preferences("1001")
    if prefs != []:
        _fail(f"异常时应返回空列表: {prefs}")
    mentions = store.get_recent_mentions("1001")
    if mentions != []:
        _fail(f"异常时应返回空列表: {mentions}")
    # 写入不应抛错
    store.upsert_preference("1001", "快递", "顺丰")
    store.add_order_mention("1001", "O001")
    _ok("driver 异常 → 返回空/静默忽略，不抛错（保障主链路）")


def main() -> int:
    print("=" * 60)
    print("M2.2 GraphMemoryStore Neo4j 图谱 CRUD 单元测试")
    print("=" * 60)
    print()
    tests = [
        test_ensure_user,
        test_upsert_preference_create,
        test_upsert_preference_merge_high_confidence,
        test_upsert_preference_empty_skipped,
        test_add_order_mention,
        test_upsert_address,
        test_get_preferences,
        test_get_default_address,
        test_get_recent_mentions,
        test_database_fallback,
        test_exception_tolerance,
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
    print(f"\u2713 全部 {len(tests)} 个测试通过，M2.2 GraphMemoryStore 就绪")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

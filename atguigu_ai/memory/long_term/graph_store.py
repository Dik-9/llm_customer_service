# -*- coding: utf-8 -*-
"""
长期记忆图谱存储（Neo4j，SPEC §4.3）

数据模型（独立 database：user_memory，与 GraphRAG 隔离）：
    (:User {user_id})-[:偏好 {created_at, confidence}]->(:Preference {type, value, confidence, source, updated_at})
    (:User)-[:常用地址 {tag}]->(:Address {label, province, city, district, street, phone, contact, is_default})
    (:User)-[:提及 {turn_id}]->(:OrderRef {order_id, mentioned_at, context})

隔离策略：
    优先使用独立 database（Neo4j 企业版/4.x+ 支持）。
    若创建/切换 database 失败（社区版无多 database），回退默认 database 并给所有标签加
    `Mem` 前缀（MemUser/MemPreference/...），避免与 GraphRAG 的 User/SKU 等标签冲突。

driver 可注入：单元测试用 _FakeDriver，真实环境用 neo4j.GraphDatabase.driver。
所有方法捕获 Neo4j 异常并记录警告，不向主对话链路抛错（SPEC §1.2 多路径保障）。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# 标签名常量（无前缀时的标准名，对应 SPEC §4.3 数据模型）
_LABEL_USER = "User"
_LABEL_PREFERENCE = "Preference"
_LABEL_ADDRESS = "Address"
_LABEL_ORDER_REF = "OrderRef"


class GraphMemoryStore:
    """Neo4j 图谱长期记忆 CRUD。

    构造方式：
        # 真实环境（自动建库/回退）
        store = GraphMemoryStore.connect(uri, user, password, database="user_memory")
        # 测试（注入 fake driver）
        store = GraphMemoryStore(driver=fake_driver, database="user_memory")
    """

    def __init__(
        self,
        driver: Any,
        database: str = "user_memory",
        label_prefix: str = "",
    ) -> None:
        self._driver = driver
        self._database: Optional[str] = database
        self._label_prefix = label_prefix
        # database 是否可用（回退到默认 database 时置 False）
        self._use_default_db = False

    # ------------------------------------------------------------------
    # 连接 / 隔离
    # ------------------------------------------------------------------

    @classmethod
    def connect(
        cls,
        uri: str,
        user: str,
        password: str,
        database: str = "user_memory",
    ) -> "GraphMemoryStore":
        """创建 Neo4j driver 并尝试建立独立 database 隔离。"""
        from neo4j import GraphDatabase

        # 抑制 driver 的 WARNING 级通知日志（如空库首次查询 "label does not exist"，
        # 属正常现象，但 driver 会打一大段 GqlStatusObject 噪音；真正的错误是 ERROR 级仍保留）
        import logging as _logging
        _logging.getLogger("neo4j").setLevel(_logging.ERROR)

        # neo4j 5.8+/6.x 支持实例级通知关闭（更精准），旧版 fallback
        try:
            driver = GraphDatabase.driver(
                uri, auth=(user, password), notifications_min_severity="OFF"
            )
        except TypeError:
            driver = GraphDatabase.driver(uri, auth=(user, password))

        store = cls(driver=driver, database=database)
        store._ensure_database_or_fallback()
        return store

    def _ensure_database_or_fallback(self) -> None:
        """尝试创建并使用独立 database；失败则回退默认 database + 标签前缀。

        注：CREATE DATABASE 属 DDL，不支持参数化标识符，database 名走字面量拼接
        （值为内部配置，非用户输入，安全）。此步直接调 driver 以便异常能传播
        触发回退（普通 _execute 会吞异常保障主链路）。
        """
        if self._database is None:
            self._use_default_db = True
            return

        db_name = self._database
        # 简单白名单校验，防止注入
        if not db_name.replace("_", "").isalnum():
            logger.warning(f"database 名含非法字符: {db_name}，回退默认 database")
            self._use_default_db = True
            self._label_prefix = "Mem"
            self._database = None
            return

        try:
            result = self._driver.execute_query(
                f"CREATE DATABASE {db_name} IF NOT EXISTS",
                {},
                database="system",
            )
            _ = result  # 触发执行
            logger.info(f"长期记忆 Neo4j database '{db_name}' 已就绪")
            return
        except Exception as e:
            logger.warning(
                f"无法创建/使用 Neo4j database '{db_name}'（可能是社区版或多库未启用）: {e}，"
                f"回退默认 database + 标签前缀 'Mem' 以隔离 GraphRAG"
            )
            self._use_default_db = True
            self._label_prefix = "Mem"
            self._database = None

    def _L(self, name: str) -> str:
        """返回带前缀的标签名。"""
        return f"{self._label_prefix}{name}"

    def _execute(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
        read: bool = False,
    ) -> List[Dict[str, Any]]:
        """执行 Cypher 并返回记录列表（每条记录转 dict）。

        database 优先级：显式传入 > self._database > 默认。
        任何异常都记录警告并返回空列表（不抛错，保障主链路）。
        """
        db = database if database is not None else self._database
        try:
            if db is not None:
                result = self._driver.execute_query(query, parameters or {}, database=db)
            else:
                result = self._driver.execute_query(query, parameters or {})
            records = []
            for rec in result.records:
                # neo4j Record 支持 dict(rec) / .data()
                try:
                    records.append(rec.data() if hasattr(rec, "data") else dict(rec))
                except Exception:
                    records.append(dict(rec))
            return records
        except Exception as e:
            logger.warning(f"[GraphMemoryStore] Cypher 执行失败: {e} | query={query[:120]}")
            return []

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def ensure_user(self, user_id: str) -> None:
        """确保 User 节点存在。"""
        if not user_id:
            return
        self._execute(
            f"MERGE (u:{self._L(_LABEL_USER)} {{user_id: $user_id}})",
            {"user_id": str(user_id)},
        )

    def upsert_preference(
        self,
        user_id: str,
        pref_type: str,
        value: str,
        confidence: float = 0.8,
        source: str = "realtime",
    ) -> None:
        """写入/更新偏好。

        合并策略（SPEC §4.3 兜底去重）：同 type 已存在时，仅当新置信度 >= 旧值才覆盖 value。
        新 type 直接创建。confidence/source/updated_at 始终刷新。
        """
        if not user_id or not pref_type or value is None:
            return
        self.ensure_user(user_id)
        self._execute(
            f"""
            MATCH (u:{self._L(_LABEL_USER)} {{user_id: $user_id}})
            OPTIONAL MATCH (u)-[:偏好]->(p:{self._L(_LABEL_PREFERENCE)} {{type: $type}})
            WITH u, p
            WHERE p IS NULL OR $confidence >= p.confidence
            FOREACH (_ IN CASE WHEN p IS NULL THEN [1] ELSE [] END |
                CREATE (np:{self._L(_LABEL_PREFERENCE)} {{type: $type, value: $value, confidence: $confidence, source: $source, updated_at: datetime()}})
                CREATE (u)-[:偏好 {{created_at: datetime(), confidence: $confidence}}]->(np)
            )
            FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
                SET p.value = $value, p.confidence = $confidence, p.source = $source, p.updated_at = datetime()
            )
            """,
            {
                "user_id": str(user_id),
                "type": pref_type,
                "value": str(value),
                "confidence": float(confidence),
                "source": source,
            },
        )

    def upsert_address(
        self,
        user_id: str,
        address: Dict[str, Any],
    ) -> None:
        """写入/更新地址。address 至少含 province/city/district，可选 label/street/phone/contact/is_default。"""
        if not user_id or not address:
            return
        self.ensure_user(user_id)
        label = address.get("label") or f"{address.get('province', '')}{address.get('city', '')}"
        self._execute(
            f"""
            MATCH (u:{self._L(_LABEL_USER)} {{user_id: $user_id}})
            MERGE (a:{self._L(_LABEL_ADDRESS)} {{label: $label, province: $province, city: $city, district: $district}})
            SET a.street = $street, a.phone = $phone, a.contact = $contact, a.is_default = $is_default
            MERGE (u)-[:常用地址]->(a)
            """,
            {
                "user_id": str(user_id),
                "label": label,
                "province": address.get("province", ""),
                "city": address.get("city", ""),
                "district": address.get("district", ""),
                "street": address.get("street", ""),
                "phone": address.get("phone", ""),
                "contact": address.get("contact", ""),
                "is_default": bool(address.get("is_default", False)),
            },
        )

    def add_order_mention(
        self,
        user_id: str,
        order_id: str,
        context: str = "",
        turn_id: Optional[int] = None,
    ) -> None:
        """记录一次订单提及（CREATE，每次提及独立保留以支持时间排序）。"""
        if not user_id or not order_id:
            return
        self.ensure_user(user_id)
        self._execute(
            f"""
            MATCH (u:{self._L(_LABEL_USER)} {{user_id: $user_id}})
            CREATE (o:{self._L(_LABEL_ORDER_REF)} {{order_id: $order_id, mentioned_at: datetime(), context: $context}})
            CREATE (u)-[:提及 {{turn_id: $turn_id}}]->(o)
            """,
            {
                "user_id": str(user_id),
                "order_id": str(order_id),
                "context": context or "",
                "turn_id": turn_id,
            },
        )

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def get_preferences(self, user_id: str) -> List[Dict[str, Any]]:
        """返回用户全部偏好。"""
        if not user_id:
            return []
        return self._execute(
            f"""
            MATCH (u:{self._L(_LABEL_USER)} {{user_id: $user_id}})-[:偏好]->(p:{self._L(_LABEL_PREFERENCE)})
            RETURN p.type AS type, p.value AS value, p.confidence AS confidence,
                   p.source AS source, p.updated_at AS updated_at
            """,
            {"user_id": str(user_id)},
            read=True,
        )

    def get_default_address(self, user_id: str) -> Optional[Dict[str, Any]]:
        """返回用户默认地址（无默认地址时返回最近一条地址）。"""
        if not user_id:
            return None
        records = self._execute(
            f"""
            MATCH (u:{self._L(_LABEL_USER)} {{user_id: $user_id}})-[:常用地址]->(a:{self._L(_LABEL_ADDRESS)})
            RETURN a.label AS label, a.province AS province, a.city AS city,
                   a.district AS district, a.street AS street, a.phone AS phone,
                   a.contact AS contact, a.is_default AS is_default
            ORDER BY a.is_default DESC
            LIMIT 1
            """,
            {"user_id": str(user_id)},
            read=True,
        )
        return records[0] if records else None

    def get_recent_mentions(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """返回用户最近提及的订单（按提及时间倒序）。"""
        if not user_id:
            return []
        return self._execute(
            f"""
            MATCH (u:{self._L(_LABEL_USER)} {{user_id: $user_id}})-[:提及]->(o:{self._L(_LABEL_ORDER_REF)})
            RETURN o.order_id AS order_id, o.mentioned_at AS mentioned_at, o.context AS context
            ORDER BY o.mentioned_at DESC
            LIMIT $limit
            """,
            {"user_id": str(user_id), "limit": int(limit)},
            read=True,
        )

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """返回用户画像：偏好列表 + 默认地址。"""
        return {
            "user_id": user_id,
            "preferences": self.get_preferences(user_id),
            "default_address": self.get_default_address(user_id),
        }

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------

    def forget_all(self, user_id: str) -> None:
        """清除用户全部记忆（SPEC §9 风险缓解：用户侧"忘记记忆"指令）。"""
        if not user_id:
            return
        self._execute(
            f"""
            MATCH (u:{self._L(_LABEL_USER)} {{user_id: $user_id}})
            OPTIONAL MATCH (u)-[r]->(n)
            DETACH DELETE u, n
            """,
            {"user_id": str(user_id)},
        )

    def close(self) -> None:
        """关闭 driver 连接。"""
        if self._driver is not None and hasattr(self._driver, "close"):
            try:
                self._driver.close()
            except Exception as e:
                logger.warning(f"[GraphMemoryStore] 关闭 driver 失败: {e}")


__all__ = ["GraphMemoryStore"]

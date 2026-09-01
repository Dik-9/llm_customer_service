# -*- coding: utf-8 -*-
"""
长期记忆抽取器（LLM 结构化抽取，SPEC §4.3）

两种模式：
- 实时抽取（extract_realtime）：每轮 action 后，对用户最新消息 + 最近3轮上下文抽取
  触发前先用正则粗筛记忆信号（"记住/以后/默认/我的地址是/上次那个..."），命中才调 LLM，省 token
- 会话结束兜底抽取（extract_end_of_session）：对完整会话跑一次，产出事实合并入库

输出统一的 ExtractedFact 列表，由 graph_store 写入图谱。
LLMClient 可注入：单元测试用 stub，真实环境复用 endpoints.yml 的 command LLM。
LLM/解析失败时返回空列表，不抛错（SPEC §1.2 多路径保障）。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from atguigu_ai.shared.llm.base_client import LLMClient

logger = logging.getLogger(__name__)


_PROMPTS_DIR = Path(__file__).parent / "prompts"


# 实时抽取信号正则（SPEC §4.3 触发关键词/句式）
# 命中任一则进入 LLM 细确认，未命中直接返回空（省 token）
_MEMORY_SIGNAL_PATTERNS = [
    re.compile(r"记住"),
    re.compile(r"以后(用|都|默认|要)"),
    re.compile(r"默认"),
    re.compile(r"别再(问|让我)"),
    re.compile(r"我的地址(是|在)"),
    re.compile(r"就寄到(这里|这)"),
    re.compile(r"上次(那个|说的|提的)"),
    re.compile(r"刚才(那个|说的|提的)"),
    re.compile(r"就是(那个|刚才)"),
    re.compile(r"我的偏好"),
    re.compile(r"习惯(用|是)"),
]


@dataclass
class ExtractedFact:
    """抽取出的单条记忆事实。

    Attributes:
        kind: 事实类型 "preference" | "address" | "order_ref"
        data: 事实字段字典（结构见 prompts 模板说明）
        confidence: 置信度 0.0~1.0（preference 用，其余默认 1.0）
    """
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Optional["ExtractedFact"]:
        """从 LLM 输出的单条 dict 构造，校验不通过返回 None。"""
        kind = d.get("kind") or d.get("type")
        # LLM 可能用 "type" 字段同时表达事实类型与偏好类型，需二次判别
        if kind not in ("preference", "address", "order_ref"):
            # 兜底：根据字段特征推断
            if "order_id" in d:
                kind = "order_ref"
            elif {"province", "city", "district"} & set(d.keys()):
                kind = "address"
            elif "value" in d and ("type" in d or "pref_type" in d):
                kind = "preference"
            else:
                return None

        if kind == "preference":
            pref_type = d.get("type") or d.get("pref_type") or ""
            value = d.get("value")
            if not pref_type or value is None:
                return None
            conf = float(d.get("confidence", 0.8) or 0.8)
            conf = max(0.0, min(1.0, conf))
            return cls(kind="preference", data={"type": str(pref_type), "value": str(value)}, confidence=conf)

        if kind == "address":
            province = d.get("province")
            city = d.get("city")
            district = d.get("district")
            if not (province and city and district):
                return None
            data = {
                "province": str(province), "city": str(city), "district": str(district),
                "label": d.get("label"), "street": d.get("street"),
                "phone": d.get("phone"), "contact": d.get("contact"),
                "is_default": bool(d.get("is_default", False)),
            }
            return cls(kind="address", data=data)

        if kind == "order_ref":
            order_id = d.get("order_id")
            if not order_id:
                return None
            return cls(kind="order_ref", data={"order_id": str(order_id), "context": str(d.get("context", ""))})

        return None


class MemoryExtractor:
    """LLM 结构化记忆抽取器。

    Args:
        llm_client: LLM 客户端（可注入）
        llm_config: 可选的 LLM 参数（temperature/max_tokens），覆盖默认
    """

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self._llm = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._env = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # 信号粗筛
    # ------------------------------------------------------------------

    @staticmethod
    def has_memory_signal(text: str) -> bool:
        """正则粗筛：用户消息是否可能含记忆信号。"""
        if not text:
            return False
        return any(p.search(text) for p in _MEMORY_SIGNAL_PATTERNS)

    # ------------------------------------------------------------------
    # 抽取
    # ------------------------------------------------------------------

    async def extract_realtime(
        self,
        latest_message: str,
        recent_turns: Optional[List[Dict[str, str]]] = None,
    ) -> List[ExtractedFact]:
        """实时抽取：对用户最新消息抽取事实。

        先粗筛，未命中信号直接返回空，不调 LLM。
        """
        if not latest_message:
            return []
        if not self.has_memory_signal(latest_message):
            logger.debug("[MemoryExtractor] 无记忆信号，跳过实时抽取")
            return []

        template = self._env.get_template("extract_realtime.jinja2")
        prompt = template.render(
            latest_message=latest_message,
            recent_turns=recent_turns or [],
        )
        return await self._extract(prompt)

    async def extract_end_of_session(
        self,
        all_turns: List[Dict[str, str]],
    ) -> List[ExtractedFact]:
        """会话结束兜底抽取：对完整会话跑一次。"""
        if not all_turns:
            return []
        template = self._env.get_template("extract_end_of_session.jinja2")
        prompt = template.render(all_turns=all_turns)
        return await self._extract(prompt)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    async def _extract(self, prompt: str) -> List[ExtractedFact]:
        """调用 LLM 并解析 JSON 数组为 ExtractedFact 列表。"""
        try:
            resp = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as e:
            logger.warning(f"[MemoryExtractor] LLM 调用失败，跳过抽取: {e}")
            return []

        raw = resp.content or ""
        facts = self._parse_facts(raw)
        logger.info(f"[MemoryExtractor] 抽取出 {len(facts)} 条事实")
        return facts

    @staticmethod
    def _parse_facts(raw: str) -> List[ExtractedFact]:
        """从 LLM 原始输出解析 JSON 数组为 ExtractedFact 列表。"""
        json_str = MemoryExtractor._extract_json(raw)
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"[MemoryExtractor] JSON 解析失败: {e} | raw={raw[:200]}")
            return []

        if not isinstance(data, list):
            # 单个对象 → 包装成单元素列表
            if isinstance(data, dict):
                data = [data]
            else:
                return []

        facts: List[ExtractedFact] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            fact = ExtractedFact.from_dict(item)
            if fact is not None:
                facts.append(fact)
        return facts

    @staticmethod
    def _extract_json(text: str) -> str:
        """从 LLM 输出中提取 JSON 部分（处理 <think>、markdown 代码块等）。"""
        if not text:
            return "[]"
        text = text.strip()
        # 移除思考模式 <think>...</think>
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        # 尝试直接解析
        if text.startswith("[") or text.startswith("{"):
            return text
        # markdown 代码块
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if m:
            return m.group(1).strip()
        # 兜底：找首个 JSON 数组/对象
        m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        if m:
            return m.group(1)
        return "[]"


__all__ = ["MemoryExtractor", "ExtractedFact"]

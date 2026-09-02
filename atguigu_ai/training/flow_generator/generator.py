# -*- coding: utf-8 -*-
"""FlowGenerator：LLM 调用 + YAML 解析 + Trainer._validate 校验闭环 + 自动修复重试。

SPEC §5.2 流程：
    1. 渲染提示词（domain 素材 + 项目真实 Flow few-shot + 用户需求）
    2. LLM 生成 YAML 草稿
    3. 提取 ```yaml 代码块 → load_flows_from_string → Trainer._validate 校验
    4. 通过 → 返回；失败 → 把错误塞进提示词 → 最多重试 max_retries 次自动修复
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from atguigu_ai.core.domain import Domain
from atguigu_ai.dialogue_understanding.flow.flow import FlowsList
from atguigu_ai.dialogue_understanding.flow.flow_loader import load_flows_from_string
from atguigu_ai.shared.llm.base_client import LLMClient
from atguigu_ai.training.trainer import Trainer

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_YAML_BLOCK_RE = re.compile(r"```(?:ya?ml)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class FlowGenerationResult:
    """Flow 生成结果。"""
    yaml_string: str = ""               # 提取出的 YAML 文本（最终写入文件的）
    flows: Optional[FlowsList] = None   # 解析后的 FlowsList
    validation_errors: List[str] = field(default_factory=list)
    success: bool = False               # 校验通过
    attempts: int = 0                   # 总尝试次数（含首次）
    raw_llm_outputs: List[str] = field(default_factory=list)  # 每次原始 LLM 输出


class FlowGenerator:
    """从自然语言需求生成 Flow YAML 并校验。

    Args:
        llm_client: LLM 客户端（complete_sync 同步调用）
        trainer: 用于复用 _validate 的 Trainer 实例
        max_retries: 校验失败后最多自动修复重试次数（默认 2）
        prompt_path: 提示词模板路径（默认用内置 flow_generate.jinja2）
    """

    def __init__(
        self,
        llm_client: LLMClient,
        trainer: Optional[Trainer] = None,
        max_retries: int = 2,
        prompt_path: Optional[str] = None,
    ) -> None:
        self.llm_client = llm_client
        self.trainer = trainer or Trainer()
        self.max_retries = max_retries
        self.prompt_path = prompt_path or str(_PROMPT_DIR / "flow_generate.jinja2")

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def generate(
        self,
        user_prompt: str,
        domain: Domain,
        example_flows: Optional[FlowsList] = None,
    ) -> FlowGenerationResult:
        """生成 Flow YAML 并校验。

        Args:
            user_prompt: 用户的自然语言需求描述
            domain: 当前项目的 Domain 定义（提供槽位 + action 素材）
            example_flows: 项目现有 Flow 列表（做 few-shot 示例，最多取 2 个）

        Returns:
            FlowGenerationResult
        """
        result = FlowGenerationResult()
        validation_errors: List[str] = []

        for attempt in range(1, self.max_retries + 2):  # 首次 + max_retries 次重试
            result.attempts = attempt
            prompt = self._render_prompt(
                user_prompt=user_prompt,
                domain=domain,
                example_flows=example_flows,
                validation_errors=validation_errors,
            )
            try:
                raw = self.llm_client.complete_sync(
                    messages=[{"role": "user", "content": prompt}]
                )
                raw_text = getattr(raw, "content", str(raw))
            except Exception as e:
                logger.error(f"[FlowGenerator] LLM 调用失败（第 {attempt} 次）: {e}")
                result.validation_errors = [f"LLM 调用失败: {e}"]
                return result

            result.raw_llm_outputs.append(raw_text)
            yaml_text = _extract_yaml_block(raw_text)
            if not yaml_text:
                validation_errors = ["LLM 输出未包含 ```yaml 代码块，请按输出要求格式生成"]
                logger.warning(f"[FlowGenerator] 第 {attempt} 次输出无 yaml 代码块")
                continue

            result.yaml_string = yaml_text

            # 解析 + 校验
            flows, errors = self._parse_and_validate(yaml_text, domain)
            result.validation_errors = errors
            if not errors:
                result.flows = flows
                result.success = True
                logger.info(f"[FlowGenerator] 第 {attempt} 次生成校验通过")
                return result

            logger.warning(
                f"[FlowGenerator] 第 {attempt} 次校验失败（{len(errors)} 个错误），"
                f"{'准备重试' if attempt <= self.max_retries else '已达重试上限'}"
            )
            validation_errors = errors

        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _render_prompt(
        self,
        user_prompt: str,
        domain: Domain,
        example_flows: Optional[FlowsList],
        validation_errors: List[str],
    ) -> str:
        """渲染 Jinja2 提示词。"""
        from jinja2 import Environment, FileSystemLoader

        env = Environment(
            loader=FileSystemLoader(str(Path(self.prompt_path).parent)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.filters["to_yaml"] = lambda obj: yaml.dump(obj, allow_unicode=True, sort_keys=False)

        template = env.get_template(Path(self.prompt_path).name)

        # 槽位素材：{slot_name: {type, ...}}
        slots_material = {
            name: (slot.to_dict() if hasattr(slot, "to_dict") else {"type": str(slot)})
            for name, slot in domain.slots.items()
        }
        # action 素材：排序列表
        actions_material = sorted(domain.actions)

        # few-shot 示例：取最多 2 个项目真实 Flow
        examples = []
        if example_flows:
            for flow in list(example_flows)[:2]:
                examples.append({
                    "flow_id": flow.id,
                    "yaml": _flow_to_yaml_text(flow),
                })

        return template.render(
            domain_slots=slots_material,
            available_actions=actions_material,
            examples=examples,
            user_prompt=user_prompt,
            validation_errors=validation_errors,
        )

    def _parse_and_validate(
        self,
        yaml_text: str,
        domain: Domain,
    ) -> tuple:
        """解析 YAML 文本为 FlowsList 并调用 Trainer._validate 校验。

        Returns:
            (FlowsList | None, errors: List[str])
        """
        try:
            flows = load_flows_from_string(yaml_text)
        except Exception as e:
            return None, [f"YAML 解析失败: {e}"]

        if len(flows) == 0:
            return None, ["YAML 中未解析出任何 flow（检查顶层结构是否为 version+flows）"]

        errors = self.trainer._validate(domain, flows)
        return flows, errors


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------

def _extract_yaml_block(text: str) -> str:
    """从 LLM 输出中提取 ```yaml ... ``` 代码块内容。

    若无代码块，返回空字符串。
    """
    if not text:
        return ""
    m = _YAML_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # 宽容回退：如果整段看起来就是 YAML（含 version/flows），直接返回
    stripped = text.strip()
    if stripped.startswith("version:") or stripped.startswith("flows:"):
        return stripped
    return ""


def _flow_to_yaml_text(flow: Any) -> str:
    """把 Flow 对象转回 YAML 文本（用于 few-shot 示例）。"""
    try:
        data = flow.as_dict()
        wrapped = {"version": "3.1", "flows": {flow.id: data}}
        return yaml.dump(wrapped, allow_unicode=True, sort_keys=False, default_flow_style=False)
    except Exception:
        return f"# {flow.id}（序列化失败）"

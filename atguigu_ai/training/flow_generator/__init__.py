# -*- coding: utf-8 -*-
"""Flow LLM 生成工具（M3，SPEC §5）。

纯开发期 CLI，不进入 Agent 主链路。
通过 LLM + 提示词模板 + Trainer._validate 校验闭环，从自然语言需求生成 Flow YAML。
"""
from atguigu_ai.training.flow_generator.generator import FlowGenerator, FlowGenerationResult

__all__ = ["FlowGenerator", "FlowGenerationResult"]

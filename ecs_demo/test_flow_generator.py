# -*- coding: utf-8 -*-
"""
M3.4 验证脚本：FlowGenerator 生成 + 校验闭环 + 自动修复重试

确定性验证（mock LLM + 真实 ecs_demo domain/flows）：
1. mock LLM 返回有效 Flow YAML → 解析+校验通过 → success=True, attempts=1
2. mock LLM 第一次返回引用不存在 action 的 Flow → 校验失败 → 第二次返回有效 → success=True, attempts=2
3. mock LLM 返回无 yaml 代码块 → 提取失败 → 重试
4. mock LLM 全部返回无效 → 达到 max_retries → success=False
5. _extract_yaml_block 辅助函数：代码块/纯 YAML/无代码块
6. 真实 ecs_demo domain + flows 渲染 prompt 验证（含素材/示例）
7. CLI flow-generate --help 注册验证
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")
    raise AssertionError(msg)


# ---------- Fakes ----------

class _FakeLLMResponse:
    """模拟 LLMResponse。"""
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLMClient:
    """按预设序列返回内容的假 LLM 客户端。"""

    def __init__(self, outputs: List[str]) -> None:
        self._outputs = list(outputs)
        self._idx = 0
        self.call_count = 0

    def complete_sync(self, messages, **kwargs):
        self.call_count += 1
        if self._idx >= len(self._outputs):
            # 超出预设时重复最后一个
            content = self._outputs[-1] if self._outputs else ""
        else:
            content = self._outputs[self._idx]
            self._idx += 1
        return _FakeLLMResponse(content)


# 有效 Flow YAML（引用 action_get_order_detail + order_id，均在 ecs_demo domain 中）
_VALID_FLOW_YAML = """```yaml
version: "3.1"
flows:
  gen_test_query:
    name: 测试查询流程
    description: 测试查询流程
    steps:
      - collect: order_id  # 收集订单ID
        next:
          - if: slots.order_id != "false"
            then:
              - action: action_get_order_detail  # 查询订单详情
                next: END
          - else: END
```"""

# 无效 Flow YAML（引用不存在的 action_xxx_not_exist）
_INVALID_ACTION_YAML = """```yaml
version: "3.1"
flows:
  gen_test_bad:
    name: 测试无效流程
    description: 测试无效流程
    steps:
      - action: action_xxx_not_exist  # 不存在的action
        next: END
```"""

# 无 yaml 代码块
_NO_YAML_BLOCK = "这是生成的流程：我没有用代码块包裹"

# YAML 解析错误
_BROKEN_YAML = """```yaml
version: "3.1"
flows:
  gen_broken:
    name: [unclosed bracket
```"""


def _load_real_domain():
    """加载真实 ecs_demo domain。"""
    from atguigu_ai.core.domain import Domain
    domain_path = Path(__file__).resolve().parent / "domain"
    return Domain.load(str(domain_path))


def _load_real_flows():
    """加载真实 ecs_demo flows。"""
    from atguigu_ai.dialogue_understanding.flow import FlowLoader
    data_path = Path(__file__).resolve().parent / "data"
    loader = FlowLoader()
    return loader.load(data_path)


# ---------- 测试 ----------

def test_generate_valid_first_try() -> None:
    print("[测试 1] mock LLM 返回有效 Flow → 首次校验通过")
    from atguigu_ai.training.flow_generator import FlowGenerator
    from atguigu_ai.training.trainer import Trainer

    llm = _FakeLLMClient([_VALID_FLOW_YAML])
    gen = FlowGenerator(llm_client=llm, trainer=Trainer(), max_retries=2)
    domain = _load_real_domain()
    flows = _load_real_flows()

    result = gen.generate("查询订单详情", domain, example_flows=flows)

    if not result.success:
        _fail(f"应成功，实际失败: {result.validation_errors}")
    if result.attempts != 1:
        _fail(f"应 attempts=1，实际 {result.attempts}")
    if result.flows is None or len(result.flows) != 1:
        _fail(f"应解析出 1 个 flow，实际 {result.flows}")
    flow = next(iter(result.flows))
    if flow.id != "gen_test_query":
        _fail(f"flow_id 应为 gen_test_query，实际 {flow.id}")
    if llm.call_count != 1:
        _fail(f"LLM 应调用 1 次，实际 {llm.call_count}")
    _ok("有效 Flow 首次校验通过，attempts=1")


def test_generate_retry_then_success() -> None:
    print("[测试 2] 第一次无效 action → 第二次有效 → success=True, attempts=2")
    from atguigu_ai.training.flow_generator import FlowGenerator
    from atguigu_ai.training.trainer import Trainer

    llm = _FakeLLMClient([_INVALID_ACTION_YAML, _VALID_FLOW_YAML])
    gen = FlowGenerator(llm_client=llm, trainer=Trainer(), max_retries=2)
    domain = _load_real_domain()

    result = gen.generate("查询订单", domain, example_flows=None)

    if not result.success:
        _fail(f"应成功（重试后），实际失败: {result.validation_errors}")
    if result.attempts != 2:
        _fail(f"应 attempts=2，实际 {result.attempts}")
    if llm.call_count != 2:
        _fail(f"LLM 应调用 2 次，实际 {llm.call_count}")
    # 验证第二次 prompt 包含了第一次的校验错误
    _ok("第一次无效 → 第二次有效，attempts=2，自动修复成功")


def test_generate_no_yaml_block_retry() -> None:
    print("[测试 3] 无 yaml 代码块 → 提取失败 → 重试")
    from atguigu_ai.training.flow_generator import FlowGenerator
    from atguigu_ai.training.trainer import Trainer

    llm = _FakeLLMClient([_NO_YAML_BLOCK, _VALID_FLOW_YAML])
    gen = FlowGenerator(llm_client=llm, trainer=Trainer(), max_retries=2)
    domain = _load_real_domain()

    result = gen.generate("查询订单", domain, example_flows=None)

    if not result.success:
        _fail(f"应成功（重试后），实际失败: {result.validation_errors}")
    if result.attempts != 2:
        _fail(f"应 attempts=2，实际 {result.attempts}")
    _ok("无 yaml 代码块 → 第二次有效，attempts=2")


def test_generate_all_fail_reach_max_retries() -> None:
    print("[测试 4] 全部无效 → 达到 max_retries → success=False")
    from atguigu_ai.training.flow_generator import FlowGenerator
    from atguigu_ai.training.trainer import Trainer

    # 3 次都返回无效 action（首次 + max_retries=2 次重试 = 3 次）
    llm = _FakeLLMClient([_INVALID_ACTION_YAML, _INVALID_ACTION_YAML, _INVALID_ACTION_YAML])
    gen = FlowGenerator(llm_client=llm, trainer=Trainer(), max_retries=2)
    domain = _load_real_domain()

    result = gen.generate("查询订单", domain, example_flows=None)

    if result.success:
        _fail("应失败（全部无效），实际成功")
    if result.attempts != 3:
        _fail(f"应 attempts=3（首次+2重试），实际 {result.attempts}")
    if llm.call_count != 3:
        _fail(f"LLM 应调用 3 次，实际 {llm.call_count}")
    if not result.validation_errors:
        _fail("应有校验错误")
    _ok(f"全部无效 → attempts={result.attempts}, success=False, errors={len(result.validation_errors)}")


def test_extract_yaml_block() -> None:
    print("[测试 5] _extract_yaml_block 辅助函数")
    from atguigu_ai.training.flow_generator.generator import _extract_yaml_block

    # 标准 yaml 代码块
    text1 = "说明文字\n```yaml\nversion: '3.1'\nflows: {}\n```\n后续"
    r1 = _extract_yaml_block(text1)
    if "version" not in r1 or "flows" not in r1:
        _fail(f"代码块提取失败: {r1}")

    # 纯 YAML（无代码块，但以 version 开头）
    text2 = "version: '3.1'\nflows: {}"
    r2 = _extract_yaml_block(text2)
    if "version" not in r2:
        _fail(f"纯 YAML 回退失败: {r2}")

    # 无 YAML
    text3 = "这是一段普通文字，没有 yaml"
    r3 = _extract_yaml_block(text3)
    if r3 != "":
        _fail(f"无 YAML 应返回空字符串，实际 {r3}")

    # 空输入
    if _extract_yaml_block("") != "":
        _fail("空输入应返回空字符串")

    _ok("代码块/纯 YAML/无 YAML/空输入 四种情况正确")


def test_prompt_renders_with_real_domain_flows() -> None:
    print("[测试 6] 真实 ecs_demo domain + flows 渲染 prompt 验证")
    from atguigu_ai.training.flow_generator import FlowGenerator
    from atguigu_ai.training.trainer import Trainer

    llm = _FakeLLMClient([_VALID_FLOW_YAML])
    gen = FlowGenerator(llm_client=llm, trainer=Trainer(), max_retries=0)
    domain = _load_real_domain()
    flows = _load_real_flows()

    # 调用内部 _render_prompt 验证渲染
    prompt = gen._render_prompt(
        user_prompt="查询我的订单",
        domain=domain,
        example_flows=flows,
        validation_errors=[],
    )

    # 验证素材渲染
    if "查询我的订单" not in prompt:
        _fail("prompt 缺少用户需求")
    if "槽位定义" not in prompt:
        _fail("prompt 缺少槽位定义素材")
    if "可用 Action 列表" not in prompt:
        _fail("prompt 缺少 action 素材")
    if "action_get_order_detail" not in prompt:
        _fail("prompt 缺少 action_get_order_detail（domain 中存在）")
    if "order_id" not in prompt:
        _fail("prompt 缺少 order_id slot（domain 中存在）")

    # 验证 few-shot 示例渲染（ecs_demo 有 flows）
    if "示例" not in prompt:
        _fail("prompt 缺少 few-shot 示例段")
    # ecs_demo 至少有 query_order_detail / cancel_order
    if "query_order_detail" not in prompt and "cancel_order" not in prompt:
        _fail("prompt few-shot 示例应包含真实 flow_id")

    # 验证语法约束渲染
    if "语法约束" not in prompt:
        _fail("prompt 缺少语法约束段")
    if "collect" not in prompt or "action" not in prompt:
        _fail("prompt 缺少 step 类型说明")

    _ok("真实 domain/flows 渲染 prompt：素材+示例+语法约束+需求 全部命中")


def test_prompt_includes_validation_errors_on_retry() -> None:
    print("[测试 7] 重试时 prompt 包含上次校验错误")
    from atguigu_ai.training.flow_generator import FlowGenerator
    from atguigu_ai.training.trainer import Trainer

    llm = _FakeLLMClient([_INVALID_ACTION_YAML, _VALID_FLOW_YAML])
    gen = FlowGenerator(llm_client=llm, trainer=Trainer(), max_retries=2)
    domain = _load_real_domain()

    # 捕获第二次 LLM 调用的 prompt
    captured_prompts = []
    original_complete = llm.complete_sync

    def _capture(messages, **kwargs):
        captured_prompts.append(messages[0]["content"] if messages else "")
        return original_complete(messages, **kwargs)

    llm.complete_sync = _capture
    gen.generate("查询订单", domain, example_flows=None)

    if len(captured_prompts) < 2:
        _fail(f"应至少捕获 2 次 prompt，实际 {len(captured_prompts)}")
    # 第二次 prompt 应包含校验错误信息
    second_prompt = captured_prompts[1]
    if "校验失败" not in second_prompt and "修复" not in second_prompt:
        _fail("第二次 prompt 应包含校验失败/修复提示")
    if "action_xxx_not_exist" not in second_prompt:
        _fail("第二次 prompt 应包含具体错误（action_xxx_not_exist）")
    _ok("重试 prompt 包含上次校验错误（错误塞回提示词）")


def test_cli_help_registered() -> None:
    print("[测试 8] CLI flow-generate --help 注册验证")
    import io
    from contextlib import redirect_stdout, redirect_stderr
    from atguigu_ai.cli import main

    f_out = io.StringIO()
    f_err = io.StringIO()
    try:
        with redirect_stdout(f_out), redirect_stderr(f_err):
            main(["flow-generate", "--help"])
    except SystemExit as e:
        # --help 会触发 SystemExit(0)
        if e.code != 0:
            _fail(f"--help 应 exit 0，实际 {e.code}")

    help_text = f_out.getvalue() + f_err.getvalue()
    if "flow-generate" not in help_text:
        _fail("help 输出缺少 flow-generate")
    if "PROMPT" not in help_text:
        _fail("help 输出缺少 PROMPT 参数")
    if "--no-write" not in help_text:
        _fail("help 输出缺少 --no-write 选项")
    if "--max-retries" not in help_text:
        _fail("help 输出缺少 --max-retries 选项")
    _ok("CLI flow-generate 已注册，--help 正常显示")


def test_cli_no_write_mode() -> None:
    print("[测试 9] CLI --no-write 模式输出到 stdout 不写文件")
    from atguigu_ai.cli import main
    from atguigu_ai.shared.llm.base_client import LLMClient

    # mock LLM client 构造
    fake_llm = _FakeLLMClient([_VALID_FLOW_YAML])

    with patch("atguigu_ai.shared.llm.create_llm_client", return_value=fake_llm):
        import io
        from contextlib import redirect_stdout
        f_out = io.StringIO()
        try:
            with redirect_stdout(f_out):
                main(["flow-generate", "查询订单详情", "--no-write", "--model", "command"])
        except SystemExit as e:
            if e.code != 0:
                _fail(f"--no-write 应 exit 0，实际 {e.code}: {f_out.getvalue()}")

        output = f_out.getvalue()
        if "生成成功" not in output:
            _fail(f"输出缺少'生成成功': {output[-300:]}")
        if "gen_test_query" not in output:
            _fail("输出缺少生成的 flow_id")
        if "已写入" in output:
            _fail("--no-write 模式不应写入文件")
    _ok("--no-write 模式：生成成功 + 输出到 stdout + 不写文件")


# ---------- 运行器 ----------

def main() -> None:
    tests = [
        test_generate_valid_first_try,
        test_generate_retry_then_success,
        test_generate_no_yaml_block_retry,
        test_generate_all_fail_reach_max_retries,
        test_extract_yaml_block,
        test_prompt_renders_with_real_domain_flows,
        test_prompt_includes_validation_errors_on_retry,
        test_cli_help_registered,
        test_cli_no_write_mode,
    ]
    print(f"\n=== M3.4 FlowGenerator 生成+校验闭环测试（共 {len(tests)} 项）===\n")
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print()
        except Exception:
            failed += 1
            import traceback
            traceback.print_exc()
            print()
    print(f"=== 结果: {passed} 通过 / {failed} 失败 / 共 {len(tests)} ===")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

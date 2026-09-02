# -*- coding: utf-8 -*-
"""
flow-generate 命令（M3，SPEC §5.2）

开发期 CLI：从自然语言需求生成 Flow YAML，通过 Trainer._validate 校验闭环。

用法：
    atguigu flow-generate "用户说改收货地址，先选未签收订单，显示详情，再选改姓名/电话/地址..."
    atguigu flow-generate "查询我的退款进度" --model command --no-write
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)


@click.command("flow-generate", help="从自然语言需求生成 Flow YAML（开发期工具）")
@click.argument("prompt", required=True)
@click.option(
    "--domain", "-d",
    type=click.Path(exists=True),
    default=None,
    help="Domain 文件或目录路径（默认自动检测 domain/ 或 domain.yml）",
)
@click.option(
    "--data",
    type=click.Path(exists=True),
    default="data",
    help="Flow 数据目录（用于 few-shot 示例，默认 data/）",
)
@click.option(
    "--model", "-m",
    default="command",
    help="LLM 模型引用名（endpoints.yml 中 models.<name>，默认 command）",
)
@click.option(
    "--endpoints",
    type=click.Path(exists=True),
    default="endpoints.yml",
    help="端点配置文件路径（默认 endpoints.yml）",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="输出文件路径（默认 data/flows/gen_<flow_id>.yml）；与 --no-write 互斥",
)
@click.option(
    "--no-write", "-n",
    is_flag=True,
    default=False,
    help="仅输出到 stdout，不写入文件",
)
@click.option(
    "--max-retries",
    type=int,
    default=2,
    help="校验失败后最多自动修复重试次数（默认 2）",
)
@click.pass_context
def flow_generate_command(
    ctx: click.Context,
    prompt: str,
    domain: Optional[str],
    data: str,
    model: str,
    endpoints: str,
    output: Optional[str],
    no_write: bool,
    max_retries: int,
) -> None:
    """从自然语言需求生成 Flow YAML。"""
    click.echo("=" * 50)
    click.echo("Atguigu AI - Flow 生成工具")
    click.echo("=" * 50)

    # 1. 自动检测 domain 路径
    if domain is None:
        if Path("domain").is_dir():
            domain = "domain"
        elif Path("domain.yml").exists():
            domain = "domain.yml"
        else:
            click.echo("错误: 未找到 domain/ 目录或 domain.yml 文件", err=True)
            raise SystemExit(1)

    domain_path = Path(domain)
    data_path = Path(data)

    click.echo(f"Domain: {domain_path.absolute()}")
    click.echo(f"Flow 数据目录（few-shot 示例源）: {data_path.absolute()}")
    click.echo(f"LLM 模型引用: {model}")
    click.echo(f"需求: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    click.echo()

    # 2. 加载 Domain + 现有 Flows
    try:
        from atguigu_ai.core.domain import Domain
        from atguigu_ai.dialogue_understanding.flow import FlowLoader

        click.echo("加载 Domain...")
        domain_obj = Domain.load(str(domain_path))
        click.echo(f"  Domain 加载成功: {len(domain_obj.slots)} 槽位, {len(domain_obj.actions)} actions")

        click.echo("加载现有 Flows（作为 few-shot 示例）...")
        loader = FlowLoader()
        existing_flows = loader.load(data_path) if data_path.exists() else None
        if existing_flows and len(existing_flows) > 0:
            click.echo(f"  现有 Flows 加载成功: {len(existing_flows)} 个（取最多 2 个做示例）")
        else:
            click.echo("  无现有 Flows，将无 few-shot 示例")
    except Exception as e:
        click.echo(f"错误: 加载 Domain/Flows 失败: {e}", err=True)
        raise SystemExit(1)

    # 3. 构造 LLM 客户端
    try:
        from atguigu_ai.shared.config import EndpointsConfig
        from atguigu_ai.shared.llm import create_llm_client

        click.echo("构造 LLM 客户端...")
        endpoints_config = EndpointsConfig.load(endpoints)
        llm_cfg = endpoints_config.get_model_config(model)
        if llm_cfg is None:
            click.echo(
                f"错误: endpoints.yml 中未找到 models.{model}，"
                f"可用: {list(endpoints_config.models.keys())}",
                err=True,
            )
            raise SystemExit(1)
        llm_client = create_llm_client(
            type=llm_cfg.type,
            model=llm_cfg.model,
            api_key=llm_cfg.api_key,
            api_base=llm_cfg.api_base,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            enable_thinking=llm_cfg.enable_thinking,
        )
        click.echo(f"  LLM 客户端就绪: model={llm_cfg.model}")
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"错误: 构造 LLM 客户端失败: {e}", err=True)
        raise SystemExit(1)

    # 4. 调用 FlowGenerator 生成 + 校验
    try:
        from atguigu_ai.training.flow_generator import FlowGenerator
        from atguigu_ai.training.trainer import Trainer

        click.echo()
        click.echo("开始生成 Flow（LLM 调用 + 校验闭环）...")
        generator = FlowGenerator(
            llm_client=llm_client,
            trainer=Trainer(),
            max_retries=max_retries,
        )
        result = generator.generate(
            user_prompt=prompt,
            domain=domain_obj,
            example_flows=existing_flows,
        )
    except Exception as e:
        click.echo(f"错误: 生成过程异常: {e}", err=True)
        raise SystemExit(1)

    # 5. 输出结果
    click.echo()
    click.echo("=" * 50)
    if result.success:
        click.echo(click.style(f"✓ 生成成功（第 {result.attempts} 次尝试校验通过）", fg="green"))
        click.echo()
        click.echo("生成的 YAML：")
        click.echo("-" * 50)
        click.echo(result.yaml_string)
        click.echo("-" * 50)

        if no_write:
            click.echo("\n--no-write 模式，不写入文件")
        else:
            # 确定输出路径
            if output:
                out_path = Path(output)
            else:
                # 从生成的 flow 提取 flow_id 作为文件名
                flow_id = "generated"
                if result.flows and len(result.flows) > 0:
                    flow_id = result.flows[0].id or "generated"
                out_path = data_path / "flows" / f"gen_{flow_id}.yml"

            # 交互确认
            click.echo()
            click.echo(f"将写入: {out_path.absolute()}")
            click.confirm("确认写入？", default=True, abort=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result.yaml_string, encoding="utf-8")
            click.echo(click.style(f"✓ 已写入: {out_path.absolute()}", fg="green"))
            click.echo("提示: 运行 'atguigu train' 后新 Flow 即生效")
    else:
        click.echo(click.style(f"✗ 生成失败（尝试 {result.attempts} 次仍未通过校验）", fg="red"))
        click.echo()
        click.echo("校验错误：")
        for err in result.validation_errors:
            click.echo(f"  - {err}")
        if result.yaml_string:
            click.echo()
            click.echo("最后一次生成的 YAML（未通过校验）：")
            click.echo("-" * 50)
            click.echo(result.yaml_string)
            click.echo("-" * 50)
        raise SystemExit(1)

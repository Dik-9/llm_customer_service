# -*- coding: utf-8 -*-
"""
理解节点

负责调用 LLMCommandGenerator 生成命令，并调用 CommandProcessor 处理命令。
这是消息处理流程的第一个核心节点。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from atguigu_ai.core.tracker import UserMessage
from atguigu_ai.dialogue_understanding.commands.slot_commands import SetSlotCommand

if TYPE_CHECKING:
    from atguigu_ai.agent.graph.state import MessageProcessingState
    from atguigu_ai.dialogue_understanding.commands.base import Command

logger = logging.getLogger(__name__)


def parse_set_slots_payload(payload: str) -> List["Command"]:
    """解析 /SetSlots(slot=value) 格式的 payload。
    
    支持按钮点击时直接解析槽位设置，绕过 LLM 处理。
    
    支持的格式：
    - /SetSlots(order_id=123)
    - /SetSlots(order_id="订单123")
    - /SetSlots(slot1=value1, slot2=value2)
    
    Args:
        payload: 以 /SetSlots( 开头的字符串
        
    Returns:
        SetSlotCommand 列表
    """
    commands: List["Command"] = []
    
    # 提取括号内的内容
    match = re.match(r'/SetSlots\((.+)\)$', payload.strip())
    if not match:
        logger.warning(f"[parse_set_slots_payload] 无法解析 payload: {payload}")
        return commands
    
    content = match.group(1)
    
    # 解析 key=value 对
    # 支持格式: slot=value, slot="value with spaces", slot='value'
    pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s]+))'
    for m in re.finditer(pattern, content):
        slot_name = m.group(1)
        # 取第一个非空的值（带引号或不带引号）
        slot_value = m.group(2) or m.group(3) or m.group(4)
        
        # 尝试转换数字
        if slot_value.isdigit():
            slot_value = int(slot_value)
        elif slot_value.lower() == 'true':
            slot_value = True
        elif slot_value.lower() == 'false':
            slot_value = False
        
        commands.append(SetSlotCommand(name=slot_name, value=slot_value))
        logger.info(f"[parse_set_slots_payload] 解析槽位: {slot_name}={slot_value}")
    
    return commands


async def understand_node(state: "MessageProcessingState") -> Dict[str, Any]:
    """理解节点：生成命令并处理。
    
    该节点执行以下步骤：
    1. 检测 /SetSlots payload（按钮点击），直接解析绕过 LLM
    2. 将用户输入封装为 UserMessage 并更新 tracker
    3. [记忆 hook] 召回画像/指代消歧 → 写 tracker.memory_context + order_id 槽（SPEC §6.1）
    4. 调用 LLMCommandGenerator 生成命令（prompt 渲染 memory_context）
    5. 调用 CommandProcessor 处理命令
    
    Args:
        state: 当前图状态
        
    Returns:
        状态更新字典
    """
    tracker = state["tracker"]
    input_message = state["input_message"]
    metadata = state.get("metadata", {})
    domain = state.get("domain")
    flows = state.get("flows")
    
    command_generator = state.get("_command_generator")
    command_processor = state.get("_command_processor")
    memory_hooks = state.get("_memory_hooks")
    
    logger.info(f"[understand_node] 处理消息: {input_message[:50]}...")
    
    # 1. 检测 /SetSlots payload（按钮点击，绕过 LLM）
    if input_message.strip().startswith("/SetSlots("):
        logger.info("[understand_node] 检测到 /SetSlots payload，绕过 LLM 直接解析")
        commands = parse_set_slots_payload(input_message)
        
        if commands and command_processor:
            # 创建用户消息（记录原始输入）
            user_message = UserMessage(
                text=input_message,
                sender_id=tracker.sender_id,
                metadata={"payload": True, **metadata},
            )
            tracker.update_with_message(user_message)
            
            # 直接处理解析出的命令
            process_result = command_processor.process(commands, tracker)
            
            logger.info(
                f"[understand_node] payload 解析了 {len(commands)} 个命令, "
                f"处理了 {process_result.commands_executed} 个"
            )
            
            return {
                "tracker": tracker,
                "current_commands": None,  # payload 不经过 LLM，没有 generation_result
                "process_result": process_result,
                "node_history": state.get("node_history", []) + ["understand"],
            }
    
    # 2. 创建用户消息并更新 tracker
    user_message = UserMessage(
        text=input_message,
        sender_id=tracker.sender_id,
        metadata=metadata,
    )
    tracker.update_with_message(user_message)
    
    # 3. 记忆 hook：understand 前召回画像/指代消歧（SPEC §6.1）
    # hooks 为 None 时 no-op；命中的画像/摘要写入 tracker.memory_context 供 prompt 渲染，
    # 指代消歧命中时把 order_id 写入 tracker 槽位
    if memory_hooks is not None:
        try:
            from atguigu_ai.memory.hooks import before_understand as _before_understand
            await _before_understand(tracker, memory_hooks)
        except Exception as e:
            logger.warning(f"[understand_node] 记忆召回 hook 异常，降级无记忆: {e}")
    
    # 初始化结果
    current_commands = None
    events = []
    process_result = None
    
    try:
        # 4. 使用命令生成器生成命令
        if command_generator:
            flows_list = flows.flows if flows else []
            generation_result = await command_generator.generate(
                tracker, domain, flows_list
            )
            current_commands = generation_result
            
            logger.warning(
                f"[understand_node] 生成了 {len(generation_result.commands)} 个命令: "
                f"{[str(c) for c in generation_result.commands]}"
            )
            
            # 5. 使用命令处理器处理命令
            if generation_result.commands and command_processor:
                process_result = command_processor.process(
                    generation_result.commands, tracker
                )
                events = process_result.events
                
                logger.info(
                    f"[understand_node] 处理了 {process_result.commands_executed} 个命令, "
                    f"产生 {len(events)} 个事件, "
                    f"下一动作: {process_result.next_action}"
                )
        else:
            logger.warning("[understand_node] 未配置命令生成器，跳过命令生成")
            
    except Exception as e:
        logger.error(f"[understand_node] 处理失败: {e}")
        return {
            "tracker": tracker,
            "current_commands": None,
            "error": str(e),
            "node_history": state.get("node_history", []) + ["understand"],
        }
    
    return {
        "tracker": tracker,
        "current_commands": current_commands,
        "process_result": process_result,
        "node_history": state.get("node_history", []) + ["understand"],
    }


# 导出
__all__ = ["understand_node"]

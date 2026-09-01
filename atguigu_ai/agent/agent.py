# -*- coding: utf-8 -*-
"""
Agent主类

提供对话系统的核心Agent实现。
基于 LangGraph 图式编排核心组件的执行流程。
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from atguigu_ai.agent.message_processor import (
    MessageProcessor,
    ProcessorConfig,
    MessageResponse,
)
from atguigu_ai.agent.actions import register_action, Action
from atguigu_ai.agent.graph import (
    get_message_processing_graph,
    create_initial_state,
)
from atguigu_ai.core.tracker import DialogueStateTracker
from atguigu_ai.core.domain import Domain
from atguigu_ai.core.stores import create_tracker_store, TrackerStore
from atguigu_ai.dialogue_understanding.flow import FlowsList, FlowLoader
from atguigu_ai.dialogue_understanding.generator import LLMCommandGenerator
from atguigu_ai.dialogue_understanding.processor import CommandProcessor
from atguigu_ai.policies import PolicyEnsemble, FlowPolicy, EnterpriseSearchPolicy
from atguigu_ai.shared.yaml_loader import read_yaml_file
from atguigu_ai.shared.config import AtguiguConfig, LLMConfig

logger = logging.getLogger(__name__)


def _load_custom_actions(actions_path: Path) -> List[str]:
    """从用户工程的 actions 目录自动加载自定义 Action。
    
    扫描指定目录下的所有 Python 文件，发现继承自 Action 基类的类，
    自动实例化并注册。
    
    Args:
        actions_path: actions 目录路径
        
    Returns:
        成功注册的 Action 名称列表
    """
    if not actions_path.exists() or not actions_path.is_dir():
        return []
    
    registered_actions = []
    
    # 将 actions 目录的父目录添加到 sys.path，以便正确导入
    parent_path = str(actions_path.parent)
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)
    
    try:
        # 扫描 actions 目录下的所有 .py 文件
        for py_file in actions_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue  # 跳过 __init__.py 等
            
            module_name = f"actions.{py_file.stem}"
            
            try:
                # 动态导入模块
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # 扫描模块中的类，找到继承自 Action 的类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # 检查是否是 Action 的子类（但不是 Action 本身）
                    if (issubclass(obj, Action) and 
                        obj is not Action and
                        obj.__module__ == module_name):
                        try:
                            # 实例化并注册
                            action_instance = obj()
                            register_action(action_instance)
                            logger.info(f"Registered custom action: {action_instance.name}")
                            registered_actions.append(action_instance.name)
                        except Exception as e:
                            logger.warning(f"Failed to register action {name}: {e}")
                            
            except Exception as e:
                logger.warning(f"Failed to load actions from {py_file}: {e}")
                
    finally:
        # 清理 sys.path（可选，保留以便后续使用）
        pass
    
    return registered_actions


def _build_memory_hooks(
    config_data: Dict[str, Any],
    endpoints_config: "EndpointsConfig",
) -> Optional[Any]:
    """从 config.yml 的 memory 节点构建 MemoryHooks（SPEC §4.5 / §6.1）。

    memory.enabled=false 或 memory 节点缺失时返回 None（等价基线，SPEC §6.4）。
    任一子组件构造失败仅记日志降级，不阻断 Agent.load（多路径保障，SPEC §1.2）。

    LLM 配置：short_term.llm / long_term.llm 引用 endpoints.yml 的 models.<name>，
    默认 "command"；两个子模块共用同一 LLM 客户端（摘要/抽取均不需强推理）。
    Neo4j 连接：复用 endpoints.yml 的 vector_store 节点（uri/user/password），
    长期记忆使用独立 database（memory.long_term.graph_database，默认 user_memory）。
    """
    memory_data = config_data.get("memory") if config_data else None
    if not memory_data:
        return None

    from atguigu_ai.shared.config import MemoryConfig
    memory_config = MemoryConfig.from_dict(memory_data)
    if not memory_config.enabled:
        logger.info("[memory] 记忆系统未启用（memory.enabled=false），等价基线")
        return None

    # 解析 LLM 配置（两个子模块共用）
    short_llm_ref = memory_config.short_term.llm or "command"
    long_llm_ref = memory_config.long_term.llm or "command"
    llm_cfg = endpoints_config.get_model_config(short_llm_ref) or endpoints_config.get_model_config(long_llm_ref)
    llm_client = None
    if llm_cfg:
        try:
            from atguigu_ai.shared.llm import create_llm_client
            llm_client = create_llm_client(
                type=llm_cfg.type,
                model=llm_cfg.model,
                api_key=llm_cfg.api_key,
                api_base=llm_cfg.api_base,
                temperature=llm_cfg.temperature,
                max_tokens=llm_cfg.max_tokens,
                enable_thinking=llm_cfg.enable_thinking,
            )
            logger.info(f"[memory] LLM 客户端就绪: model={llm_cfg.model}")
        except Exception as e:
            logger.warning(f"[memory] LLM 客户端构造失败，记忆抽取/压缩将降级: {e}")

    # 构造长期记忆组件（Neo4j 图谱）
    graph_store = None
    recaller = None
    extractor = None
    if memory_config.long_term.enabled:
        # Neo4j 连接（复用 vector_store 配置）
        vs_config = endpoints_config.vector_store.to_connect_config() if endpoints_config.vector_store else {}
        uri = vs_config.get("uri")
        user = vs_config.get("user")
        password = vs_config.get("password")
        database = memory_config.long_term.graph_database or "user_memory"
        if uri and user and password:
            try:
                from atguigu_ai.memory.long_term.graph_store import GraphMemoryStore
                graph_store = GraphMemoryStore.connect(uri, user, password, database=database)
                logger.info(f"[memory] Neo4j 图谱存储就绪: database={database}")
            except Exception as e:
                logger.warning(f"[memory] Neo4j 图谱存储构造失败，长期记忆降级: {e}")
        else:
            logger.warning("[memory] endpoints.yml 缺少 vector_store(uri/user/password)，长期记忆降级")

        if graph_store is not None:
            try:
                from atguigu_ai.memory.long_term.recaller import MemoryRecaller
                recaller = MemoryRecaller(graph_store)
            except Exception as e:
                logger.warning(f"[memory] MemoryRecaller 构造失败: {e}")

        if llm_client is not None:
            try:
                from atguigu_ai.memory.long_term.extractor import MemoryExtractor
                extractor = MemoryExtractor(llm_client)
            except Exception as e:
                logger.warning(f"[memory] MemoryExtractor 构造失败: {e}")

    # 构造短期记忆组件（对话压缩）
    compressor = None
    if memory_config.short_term.enabled and llm_client is not None:
        try:
            from atguigu_ai.memory.short_term.compressor import ShortTermCompressor
            compressor = ShortTermCompressor(
                llm_client=llm_client,
                max_raw_turns=memory_config.short_term.max_raw_turns,
                keep_recent_turns=memory_config.short_term.keep_recent_turns,
            )
            logger.info(
                f"[memory] 短期压缩器就绪: max_raw_turns={memory_config.short_term.max_raw_turns}, "
                f"keep_recent_turns={memory_config.short_term.keep_recent_turns}"
            )
        except Exception as e:
            logger.warning(f"[memory] ShortTermCompressor 构造失败: {e}")

    # 任一子组件就绪即构造 MemoryHooks（其内部按子开关 no-op 缺失组件）
    if recaller is None and extractor is None and compressor is None and graph_store is None:
        logger.warning("[memory] 所有记忆子组件构造失败，memory_hooks=None 等价基线")
        return None

    try:
        from atguigu_ai.memory.hooks import MemoryHooks
        hooks = MemoryHooks(
            config=memory_config,
            recaller=recaller,
            extractor=extractor,
            compressor=compressor,
            graph_store=graph_store,
        )
        enabled_parts = []
        if memory_config.short_term.enabled:
            enabled_parts.append(f"short_term({'on' if compressor else 'degraded'})")
        if memory_config.long_term.enabled:
            enabled_parts.append(f"long_term(recaller={'on' if recaller else 'off'},"
                                 f"extractor={'on' if extractor else 'off'},"
                                 f"graph={'on' if graph_store else 'off'})")
        logger.info(f"[memory] MemoryHooks 注入成功: {', '.join(enabled_parts)}")
        return hooks
    except Exception as e:
        logger.warning(f"[memory] MemoryHooks 构造失败，等价基线: {e}")
        return None


@dataclass
class AgentConfig:
    """Agent配置。
    
    Attributes:
        domain_path: Domain文件路径
        flows_path: Flows文件/目录路径
        config_path: 配置文件路径
        endpoints_path: 端点配置路径
        tracker_store_type: Tracker存储类型
        tracker_store_config: Tracker存储配置
        llm_config: LLM配置
    """
    domain_path: str = "domain.yml"
    flows_path: str = "data/flows"
    config_path: str = "config.yml"
    endpoints_path: str = "endpoints.yml"
    tracker_store_type: str = "memory"
    tracker_store_config: Dict[str, Any] = field(default_factory=dict)
    llm_config: Optional[LLMConfig] = None


class Agent:
    """对话系统Agent。
    
    Agent是对话系统的核心类，负责：
    - 加载和管理配置
    - 处理用户消息
    - 管理对话状态
    - 协调各个组件
    
    使用示例：
    ```python
    agent = Agent.load("./my_bot")
    response = await agent.handle_message("你好", sender_id="user1")
    print(response.messages)
    ```
    """
    
    def __init__(
        self,
        domain: Optional[Domain] = None,
        flows: Optional[FlowsList] = None,
        tracker_store: Optional[TrackerStore] = None,
        policy_ensemble: Optional[PolicyEnsemble] = None,
        command_generator: Optional[LLMCommandGenerator] = None,
        nlg_generator: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[Any] = None,
        memory_hooks: Optional[Any] = None,
    ):
        """初始化Agent。

        Args:
            domain: Domain定义
            flows: Flow列表
            tracker_store: Tracker存储
            policy_ensemble: 策略集成器
            command_generator: 命令生成器
            nlg_generator: NLG生成器（可选，用于响应重述）
            config: Agent配置
            tool_registry: 工具注册表（统一执行入口，SPEC §6.2）
            memory_hooks: 记忆系统 hook 编排器（SPEC §6.1；None 时等价基线）
        """
        self.domain = domain or Domain()
        self.flows = flows or FlowsList()
        self.config = config or AgentConfig()

        # ToolRegistry：统一执行入口（MCP 启用时走 MCP，否则本地直调，SPEC §6.2）
        # 惰性导入避免顶层循环依赖；未注入时构造本地注册表，等价基线行为
        from atguigu_ai.mcp.tool_registry import ToolRegistry
        self.tool_registry = tool_registry or ToolRegistry()

        # MemoryHooks：记忆系统 hook 编排器（SPEC §6.1）
        # None 时 understand_node/action_node/handle_message 中的 hook 调用全部 no-op，等价基线
        self.memory_hooks = memory_hooks

        # 初始化Tracker存储
        if tracker_store:
            self.tracker_store = tracker_store
        else:
            self.tracker_store = create_tracker_store(
                self.config.tracker_store_type,
                **self.config.tracker_store_config,
            )
        self.tracker_store.set_domain(self.domain)
        
        # 初始化策略
        if policy_ensemble:
            self.policy_ensemble = policy_ensemble
        else:
            self.policy_ensemble = PolicyEnsemble(policies=[
                FlowPolicy(flows=self.flows),
                EnterpriseSearchPolicy(),
            ])
        
        # 初始化命令生成器
        self.command_generator = command_generator
        
        # 初始化NLG生成器
        self.nlg_generator = nlg_generator
        
        # 初始化命令处理器
        self.command_processor = CommandProcessor(
            domain=self.domain,
            flows=self.flows.flows if self.flows else [],
        )
        
        # 获取 LangGraph 消息处理图（惰性初始化的单例）
        self.graph = get_message_processing_graph()
        
        # 保留消息处理器作为备用（向后兼容）
        self.message_processor = MessageProcessor(
            domain=self.domain,
            flows=self.flows,
            policy_ensemble=self.policy_ensemble,
            command_generator=self.command_generator,
        )
    
    async def handle_message(
        self,
        message: str,
        sender_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageResponse:
        """处理用户消息。
        
        使用 LangGraph 图式编排执行消息处理流程。
        
        Args:
            message: 用户消息文本
            sender_id: 发送者ID
            metadata: 消息元数据
            
        Returns:
            处理响应
        """
        # 获取或创建Tracker
        tracker = await self.tracker_store.get_or_create_tracker(sender_id)
        
        # 构建初始状态
        initial_state = create_initial_state(
            tracker=tracker,
            input_message=message,
            domain=self.domain,
            flows=self.flows,
            metadata=metadata,
            max_actions=10,
            command_generator=self.command_generator,
            command_processor=self.command_processor,
            policy_ensemble=self.policy_ensemble,
            tool_registry=self.tool_registry,
            memory_hooks=self.memory_hooks,
        )
        
        # 执行图
        logger.info(f"[Agent] 使用 LangGraph 处理消息: {message[:50]}...")
        final_state = await self.graph.ainvoke(initial_state)
        
        # 从最终状态提取结果
        updated_tracker = final_state.get("tracker", tracker)
        final_responses = final_state.get("final_responses", [])
        node_history = final_state.get("node_history", [])
        error = final_state.get("error")

        # 记忆 hook：save tracker 前短期压缩 + 会话结束兜底抽取（SPEC §6.1）
        # hooks 为 None 时 no-op；压缩/抽取失败仅记日志，不影响主链路
        if self.memory_hooks is not None:
            try:
                from atguigu_ai.memory.hooks import before_save_tracker as _before_save_tracker
                await _before_save_tracker(updated_tracker, self.memory_hooks)
            except Exception as e:
                logger.warning(f"[Agent] 记忆 save 前 hook 异常，跳过: {e}")

        # 保存Tracker
        await self.tracker_store.save(updated_tracker)
        
        # 构建响应
        response = MessageResponse(
            messages=final_responses,
            metadata={
                "node_history": node_history,
                "error": error,
            },
        )
        
        logger.info(
            f"[Agent] 处理完成, 节点路径: {' -> '.join(node_history)}, "
            f"响应数: {len(final_responses)}"
        )
        
        return response
    
    def handle_message_sync(
        self,
        message: str,
        sender_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageResponse:
        """同步版本的消息处理。"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.handle_message(message, sender_id, metadata)
        )
    
    async def get_tracker(self, sender_id: str) -> Optional[DialogueStateTracker]:
        """获取指定用户的Tracker。
        
        Args:
            sender_id: 发送者ID
            
        Returns:
            Tracker实例，如果不存在则返回None
        """
        return await self.tracker_store.retrieve(sender_id)
    
    async def reset_tracker(self, sender_id: str) -> None:
        """重置指定用户的对话状态。
        
        Args:
            sender_id: 发送者ID
        """
        tracker = await self.tracker_store.retrieve(sender_id)
        if tracker:
            tracker.restart()
            await self.tracker_store.save(tracker)
    
    def register_action(self, action: Action) -> None:
        """注册自定义动作。
        
        Args:
            action: 动作实例
        """
        register_action(action)
    
    @classmethod
    def load(
        cls,
        project_path: Union[str, Path],
        config: Optional[AgentConfig] = None,
    ) -> "Agent":
        """从项目目录或模型压缩包加载Agent。
        
        支持以下输入：
        - .tar.gz 模型压缩包路径
        - 包含 .tar.gz 文件的目录（自动选择最新）
        - 项目目录（直接加载配置文件）
        
        Args:
            project_path: 项目目录路径或模型压缩包路径
            config: Agent配置（覆盖默认值）
            
        Returns:
            Agent实例
        """
        import tempfile
        from atguigu_ai.training.model_storage import (
            extract_model_archive,
            get_model_path,
            get_latest_model,
        )
        
        project_path = Path(project_path)
        
        if config is None:
            config = AgentConfig()
        
        # 确定实际的工作目录
        # 情况1: 输入是 .tar.gz 文件
        if project_path.is_file() and project_path.name.endswith(".tar.gz"):
            logger.info(f"Loading agent from model archive: {project_path}")
            # 解压到临时目录
            temp_dir = tempfile.mkdtemp(prefix="atguigu_model_")
            extract_model_archive(project_path, temp_dir)
            working_path = Path(temp_dir)
            logger.info(f"Extracted model to: {working_path}")
        
        # 情况2: 输入是目录
        elif project_path.is_dir():
            # 检查是否有 models/ 子目录包含 .tar.gz 文件
            models_dir = project_path / "models"
            latest_model = None
            if models_dir.exists():
                latest_model = get_latest_model(models_dir)
            
            if latest_model:
                # 找到了模型压缩包，解压并使用
                logger.info(f"Found model archive: {latest_model}")
                temp_dir = tempfile.mkdtemp(prefix="atguigu_model_")
                extract_model_archive(latest_model, temp_dir)
                working_path = Path(temp_dir)
                logger.info(f"Extracted model to: {working_path}")
            else:
                # 没有找到压缩包，直接使用项目目录（向后兼容）
                working_path = project_path
                logger.info(f"Loading agent from project directory: {project_path}")
        else:
            raise FileNotFoundError(f"Path not found: {project_path}")
        
        logger.info(f"Working path: {working_path}")
        
        # 将项目目录和工作目录添加到 sys.path，以便加载用户自定义模块（如 addons/）
        # 注意：当使用模型压缩包时，working_path 是临时目录，但用户自定义代码在原始 project_path 中
        project_path_str = str(project_path.absolute())
        working_path_str = str(working_path.absolute())
        
        # 优先添加原始项目目录（用户自定义代码所在位置）
        if project_path_str not in sys.path:
            sys.path.insert(0, project_path_str)
            logger.info(f"Added project path to sys.path: {project_path_str}")
        
        # 如果工作目录与项目目录不同，也添加工作目录
        if working_path_str != project_path_str and working_path_str not in sys.path:
            sys.path.insert(0, working_path_str)
            logger.info(f"Added working path to sys.path: {working_path_str}")
        
        # 加载Domain
        # 支持两种格式: domain.yml 文件或 domain/ 目录
        domain_path = working_path / config.domain_path
        domain = None
        if domain_path.exists():
            domain = Domain.load(str(domain_path))
            logger.info(f"Loaded domain from {domain_path}")
        else:
            # 如果配置的路径不存在，尝试查找 domain 目录（兼容模型压缩包）
            domain_dir = working_path / "domain"
            if domain_dir.exists() and domain_dir.is_dir():
                domain = Domain.load(str(domain_dir))
                logger.info(f"Loaded domain from {domain_dir}")
        
        # 加载Flows
        flows_path = working_path / config.flows_path
        flows = FlowsList()
        if flows_path.exists():
            loader = FlowLoader()
            flows = loader.load(flows_path)
            logger.info(f"Loaded {len(flows)} flows from {flows_path}")
        
        # 加载用户自定义 Actions
        # 自动发现 actions/ 目录中的 Action 类并注册
        actions_path = working_path / "actions"
        custom_action_names = _load_custom_actions(actions_path)
        if custom_action_names:
            logger.info(f"Loaded {len(custom_action_names)} custom actions from {actions_path}")
            # 将自动发现的 actions 同步到 domain 中
            if domain:
                for action_name in custom_action_names:
                    domain.add_action(action_name)
                logger.debug(f"Synced custom actions to domain: {custom_action_names}")
        
        # 加载endpoints配置（包含模型定义）
        endpoints_path = working_path / config.endpoints_path
        from atguigu_ai.shared.config import EndpointsConfig
        endpoints_config = EndpointsConfig.load(endpoints_path) if endpoints_path.exists() else EndpointsConfig()
        
        # 加载config配置
        config_path = working_path / config.config_path
        llm_config = None
        retrieval_config = None
        nlg_config = None
        enterprise_llm_config = None
        enterprise_embeddings_config = None
        retriever_class_path = None
        config_data = {}
        if config_path.exists():
            config_data = read_yaml_file(str(config_path)) or {}
            if config_data:
                # 从 pipeline 配置中获取 LLMCommandGenerator 的 llm 引用
                pipeline = config_data.get("pipeline", [])
                for component in pipeline:
                    if component.get("name") == "LLMCommandGenerator":
                        llm_ref = component.get("llm", "default")
                        llm_config = endpoints_config.get_model_config(llm_ref)
                        if llm_config:
                            logger.info(f"从 pipeline 配置加载 LLM '{llm_ref}'")
                        else:
                            logger.warning(f"endpoints.yml 中未找到模型 '{llm_ref}'")
                        break
                
                # 从 policies 配置中获取 EnterpriseSearchPolicy 的参数
                policies = config_data.get("policies", [])
                retriever_class_path = None
                graphrag_llm_config = None  # GraphRAG 内部 LLM 配置
                for policy in policies:
                    if policy.get("name") == "EnterpriseSearchPolicy":
                        # 获取检索器类路径
                        retriever_class_path = policy.get("vector_store")
                        if retriever_class_path:
                            logger.info(f"从 policies 配置读取检索器类: {retriever_class_path}")

                        # 获取策略的 llm 引用（RAG 回答生成）
                        policy_llm_ref = policy.get("llm", "rag")
                        enterprise_llm_config = endpoints_config.get_model_config(policy_llm_ref)
                        if enterprise_llm_config:
                            logger.info(f"从 policies 配置加载 EnterpriseSearchPolicy LLM '{policy_llm_ref}'")

                        # 获取 GraphRAG 内部 LLM 引用（Cypher 生成/验证/校正）
                        graphrag_llm_ref = policy.get("graphrag_llm", "graphrag")
                        graphrag_llm_config = endpoints_config.get_model_config(graphrag_llm_ref)
                        if graphrag_llm_config:
                            logger.info(f"从 policies 配置加载 GraphRAG LLM '{graphrag_llm_ref}'")

                        # 获取策略的 embeddings 引用
                        policy_embeddings_ref = policy.get("embeddings", "default")
                        enterprise_embeddings_config = endpoints_config.get_embeddings_config(policy_embeddings_ref)
                        if enterprise_embeddings_config:
                            logger.info(f"从 policies 配置加载 EnterpriseSearchPolicy embeddings '{policy_embeddings_ref}'")
                        break
                
                # 加载检索配置
                if "retrieval" in config_data:
                    from atguigu_ai.shared.config import RetrievalConfig
                    retrieval_config = RetrievalConfig.from_dict(config_data.get("retrieval", {}))
        
        # 从 endpoints.yml 获取 NLG 配置
        nlg_config = endpoints_config.nlg
        
        # 创建命令生成器
        command_generator = None
        if llm_config:
            from atguigu_ai.dialogue_understanding.generator import (
                LLMCommandGenerator,
                LLMGeneratorConfig,
            )
            generator_config = LLMGeneratorConfig(
                type=llm_config.type,
                model=llm_config.model,
                api_key=llm_config.api_key,
                api_base=llm_config.api_base,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                enable_thinking=llm_config.enable_thinking,
            )
            command_generator = LLMCommandGenerator(config=generator_config)
        
        # 从 endpoints.yml 获取 Tracker 存储配置
        tracker_store_config = endpoints_config.tracker_store
        tracker_store = create_tracker_store(
            tracker_store_config.type,
            path=tracker_store_config.path,
        )
        logger.info(f"创建 TrackerStore: type={tracker_store_config.type}, path={tracker_store_config.path}")
        
        # 创建策略
        from atguigu_ai.policies import EnterpriseSearchPolicyConfig
        flow_policy = FlowPolicy(flows=flows)
        
        # 创建 Retriever（类路径从 config.yml policies 读取，连接配置从 endpoints.yml 读取）
        retriever = None
        if retriever_class_path:
            try:
                from atguigu_ai.retrieval import create_retriever
                connect_config = endpoints_config.vector_store.to_connect_config()
                # 将 GraphRAG LLM 配置注入到 connect_config，供检索器内部使用
                graph_llm_config = graphrag_llm_config or enterprise_llm_config or llm_config
                if graph_llm_config:
                    connect_config["llm"] = {
                        "type": graph_llm_config.type,
                        "model": graph_llm_config.model,
                        "api_key": graph_llm_config.api_key,
                        "api_base": graph_llm_config.api_base,
                        "temperature": graph_llm_config.temperature,
                        "max_tokens": graph_llm_config.max_tokens,
                    }
                retriever = create_retriever(retriever_class_path, connect_config)
                if retriever:
                    logger.info(f"创建检索器: {retriever_class_path}")
            except Exception as e:
                logger.warning(f"创建检索器失败: {e}")
        
        # 创建NLG生成器（如果配置了重述）
        nlg_generator = None
        if nlg_config and nlg_config.rephrase_enabled:
            try:
                from atguigu_ai.nlg import ResponseRephraser, RephraserConfig, TemplateNLG
                
                # 获取重述用的LLM配置
                rephrase_llm_config = None
                if nlg_config.rephrase_model:
                    rephrase_llm_config = endpoints_config.get_model_config(nlg_config.rephrase_model)
                if not rephrase_llm_config and llm_config:
                    rephrase_llm_config = llm_config  # 回退到主LLM配置
                
                if rephrase_llm_config:
                    rephrase_config = RephraserConfig(
                        enabled=True,
                        llm_type=rephrase_llm_config.type,
                        llm_model=rephrase_llm_config.model,
                        style=nlg_config.rephrase_style,
                        rephrase_threshold=nlg_config.rephrase_threshold,
                        preserve_slots=nlg_config.preserve_slots,
                        language=nlg_config.language,
                    )
                    
                    # 创建LLM客户端
                    from atguigu_ai.shared.llm import create_llm_client
                    rephrase_llm = create_llm_client(
                        type=rephrase_llm_config.type,
                        model=rephrase_llm_config.model,
                        api_key=rephrase_llm_config.api_key,
                        api_base=rephrase_llm_config.api_base,
                        temperature=0.7,  # 重述使用较高温度
                    )
                    
                    # 创建模板NLG作为底层
                    template_nlg = TemplateNLG(domain=domain)
                    
                    nlg_generator = ResponseRephraser(
                        config=rephrase_config,
                        base_generator=template_nlg,
                        llm_client=rephrase_llm,
                    )
                    logger.info(f"Loaded NLG rephraser with style: {nlg_config.rephrase_style}")
            except Exception as e:
                logger.warning(f"Failed to create NLG generator: {e}")
        
        # 使用 policies 配置中的 LLM 配置创建 EnterpriseSearchPolicy
        # 优先使用 policies 中指定的 llm，否则回退到 pipeline 中的 llm
        policy_llm_config = enterprise_llm_config or llm_config
        if policy_llm_config:
            enterprise_config = EnterpriseSearchPolicyConfig(
                llm_type=policy_llm_config.type,
                llm_model=policy_llm_config.model,
            )
            from atguigu_ai.shared.llm import create_llm_client
            # RAG 回答生成需要比命令生成更大的 max_tokens（模型 thinking 会消耗大量输出预算）
            rag_max_tokens = max(policy_llm_config.max_tokens, 4096)
            llm_client = create_llm_client(
                type=policy_llm_config.type,
                model=policy_llm_config.model,
                api_key=policy_llm_config.api_key,
                api_base=policy_llm_config.api_base,
                temperature=policy_llm_config.temperature,
                max_tokens=rag_max_tokens,
                enable_thinking=policy_llm_config.enable_thinking,
            )
            enterprise_policy = EnterpriseSearchPolicy(
                config=enterprise_config,
                llm_client=llm_client,
                retriever=retriever,
            )
            logger.info(f"创建 EnterpriseSearchPolicy: llm={policy_llm_config.model}")
        else:
            enterprise_policy = EnterpriseSearchPolicy(retriever=retriever)
        
        policy_ensemble = PolicyEnsemble(policies=[
            flow_policy,
            enterprise_policy,
        ])

        # 构建 ToolRegistry（SPEC §6.2）：MCP enabled 时注入 MCPClient + 电商映射，否则本地直调
        # 注：build_tool_registry 不会触发 initialize（Agent.load 为同步方法，连接由首次 call_tool 惰性建立）
        from atguigu_ai.mcp.tool_registry import build_tool_registry
        tool_registry = build_tool_registry(endpoints_config.mcp)

        # 构建记忆系统 hook（SPEC §6.1）：memory.enabled=false 时 memory_hooks=None，等价基线
        # 任一子组件构造失败仅记日志降级，不阻断 Agent.load（多路径保障，SPEC §1.2）
        memory_hooks = _build_memory_hooks(config_data, endpoints_config)

        return cls(
            domain=domain,
            flows=flows,
            tracker_store=tracker_store,
            policy_ensemble=policy_ensemble,
            command_generator=command_generator,
            nlg_generator=nlg_generator,
            config=config,
            tool_registry=tool_registry,
            memory_hooks=memory_hooks,
        )
    
    @classmethod
    async def create(
        cls,
        domain: Optional[Domain] = None,
        flows: Optional[FlowsList] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> "Agent":
        """创建Agent实例的便捷方法。
        
        Args:
            domain: Domain定义
            flows: Flow列表
            llm_provider: LLM提供商
            llm_model: LLM模型
            **kwargs: 额外配置
            
        Returns:
            Agent实例
        """
        from atguigu_ai.dialogue_understanding.generator import (
            LLMCommandGenerator,
            LLMGeneratorConfig,
        )
        
        # 创建命令生成器
        generator_config = LLMGeneratorConfig(
            provider=llm_provider,
            model=llm_model,
        )
        command_generator = LLMCommandGenerator(config=generator_config)
        
        return cls(
            domain=domain,
            flows=flows,
            command_generator=command_generator,
        )


# 导出
__all__ = [
    "Agent",
    "AgentConfig",
]

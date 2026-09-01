# -*- coding: utf-8 -*-
"""
config - 配置管理模块

提供配置文件的加载、解析和管理功能。
支持从YAML文件加载配置，并支持环境变量替换。
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Union

from atguigu_ai.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_SERVER_PORT,
    LLM_TYPE_OPENAI,
    TRACKER_STORE_JSON,
)
from atguigu_ai.shared.exceptions import (
    ConfigurationException,
    InvalidConfigException,
    MissingConfigException,
)
from atguigu_ai.shared.yaml_loader import read_yaml_file


# 环境变量替换的正则表达式: ${VAR_NAME} 或 ${VAR_NAME:default}
ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _resolve_env_vars(value: Any) -> Any:
    """解析配置值中的环境变量
    
    支持格式：
    - ${VAR_NAME}: 从环境变量获取，不存在则为空
    - ${VAR_NAME:default}: 从环境变量获取，不存在则使用默认值
    
    参数：
        value: 待解析的配置值
        
    返回：
        解析后的配置值
    """
    if isinstance(value, str):
        def replace_env_var(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_value)
        
        return ENV_VAR_PATTERN.sub(replace_env_var, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    else:
        return value


@dataclass
class LLMConfig:
    """LLM配置
    
    定义LLM服务的连接和调用参数。
    
    属性：
        type: LLM类型 (openai/qwen/azure/anthropic)
            - openai: OpenAI API或兼容服务(如vLLM)
            - qwen: 阿里云DashScope通义千问
            - azure: Azure OpenAI
            - anthropic: Anthropic Claude
        model: 模型名称
        api_key: API密钥
        api_base: API基础URL(可选，用于vLLM等自部署服务)
        temperature: 温度参数
        max_tokens: 最大token数
        timeout: 请求超时(秒)
        enable_thinking: 是否启用深度思考模式
            - qwen类型：DashScope原生支持
            - openai类型+api_base：通过chat_template_kwargs传递(vLLM)
        extra: 额外配置参数(如azure_endpoint, api_version等)
    """
    type: str = LLM_TYPE_OPENAI
    model: str = "gpt-3.5-turbo"
    api_key: str = ""
    api_base: Optional[str] = None
    temperature: float = DEFAULT_LLM_TEMPERATURE
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    timeout: int = DEFAULT_LLM_TIMEOUT
    enable_thinking: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LLMConfig":
        """从字典创建LLM配置
        
        参数：
            config: 配置字典
            
        返回：
            LLMConfig实例
        """
        # 解析环境变量
        config = _resolve_env_vars(config)
        
        return cls(
            type=config.get("type", LLM_TYPE_OPENAI),
            model=config.get("model", "gpt-3.5-turbo"),
            api_key=config.get("api_key", ""),
            api_base=config.get("api_base"),
            temperature=config.get("temperature", DEFAULT_LLM_TEMPERATURE),
            max_tokens=config.get("max_tokens", DEFAULT_LLM_MAX_TOKENS),
            timeout=config.get("timeout", DEFAULT_LLM_TIMEOUT),
            enable_thinking=config.get("enable_thinking", False),
            extra={k: v for k, v in config.items() 
                   if k not in ["type", "model", "api_key", "api_base", 
                               "temperature", "max_tokens", "timeout", "enable_thinking"]},
        )


@dataclass
class EmbeddingsConfig:
    """向量化配置
    
    定义文本向量化服务的连接参数。
    
    属性：
        type: 向量化类型 (openai/qwen/azure)
        model: 模型名称
        api_key: API密钥
        api_base: API基础URL(可选)
        dimensions: 向量维度(可选)
    """
    type: str = LLM_TYPE_OPENAI
    model: str = "text-embedding-ada-002"
    api_key: str = ""
    api_base: Optional[str] = None
    dimensions: Optional[int] = None
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EmbeddingsConfig":
        """从字典创建向量化配置"""
        config = _resolve_env_vars(config)
        
        return cls(
            type=config.get("type", LLM_TYPE_OPENAI),
            model=config.get("model", "text-embedding-ada-002"),
            api_key=config.get("api_key", ""),
            api_base=config.get("api_base"),
            dimensions=config.get("dimensions"),
        )


@dataclass
class TrackerStoreConfig:
    """Tracker存储配置
    
    属性：
        type: 存储类型 (json/mysql/memory)
        path: JSON存储路径(仅json类型)
        url: 数据库连接URL(仅mysql类型)
        db: 数据库名称
        host: 数据库主机
        port: 数据库端口
        username: 数据库用户名
        password: 数据库密码
    """
    type: str = TRACKER_STORE_JSON
    path: str = "trackers"
    url: Optional[str] = None
    db: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrackerStoreConfig":
        """从字典创建Tracker存储配置"""
        config = _resolve_env_vars(config)
        
        return cls(
            type=config.get("type", TRACKER_STORE_JSON),
            path=config.get("path", "trackers"),
            url=config.get("url"),
            db=config.get("db"),
            host=config.get("host"),
            port=config.get("port"),
            username=config.get("username"),
            password=config.get("password"),
        )


@dataclass
class RetrievalConfig:
    """检索配置
    
    支持两种检索模式：
    1. Flow检索：用于检索对话流程
    2. 知识库检索：用于RAG增强的企业知识检索
    
    属性：
        flow_retrieval_enabled: 是否启用Flow检索
        flow_retrieval_top_k: Flow检索返回数量
        knowledge_retrieval_enabled: 是否启用知识库检索
        knowledge_retrieval_top_k: 知识库检索返回数量
        knowledge_source: 知识库来源路径
        retriever_type: Retriever类型（faiss/knowledge/自定义类路径）
        vector_store_path: 向量存储路径（用于加载已有索引）
        embedding_provider: 嵌入提供商（openai/qwen等）
        embedding_model: 嵌入模型名称
        embedding_dimension: 嵌入向量维度
        similarity_threshold: 相似度阈值
    """
    flow_retrieval_enabled: bool = True
    flow_retrieval_top_k: int = 5
    knowledge_retrieval_enabled: bool = False
    knowledge_retrieval_top_k: int = 3
    knowledge_source: Optional[str] = None
    # 自定义Retriever支持
    retriever_type: str = "faiss"
    vector_store_path: Optional[str] = None
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    similarity_threshold: float = 0.5
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "RetrievalConfig":
        """从字典创建检索配置"""
        config = _resolve_env_vars(config)
        
        flow_config = config.get("flow_retrieval", {})
        knowledge_config = config.get("knowledge_retrieval", {})
        retriever_config = config.get("retriever", {})
        
        return cls(
            flow_retrieval_enabled=flow_config.get("enabled", True),
            flow_retrieval_top_k=flow_config.get("top_k", 5),
            knowledge_retrieval_enabled=knowledge_config.get("enabled", False),
            knowledge_retrieval_top_k=knowledge_config.get("top_k", 3),
            knowledge_source=knowledge_config.get("source"),
            # Retriever配置
            retriever_type=retriever_config.get("type", "faiss"),
            vector_store_path=retriever_config.get("vector_store_path"),
            embedding_provider=retriever_config.get("embedding_provider", "openai"),
            embedding_model=retriever_config.get("embedding_model", "text-embedding-3-small"),
            embedding_dimension=retriever_config.get("embedding_dimension", 1536),
            similarity_threshold=retriever_config.get("similarity_threshold", 0.5),
        )
    
    def to_retriever_config(self) -> Dict[str, Any]:
        """转换为Retriever工厂函数所需的配置格式"""
        return {
            "type": self.retriever_type,
            "vector_store_path": self.vector_store_path,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "top_k": self.knowledge_retrieval_top_k,
            "similarity_threshold": self.similarity_threshold,
        }


@dataclass
class NLGConfig:
    """NLG配置
    
    控制响应生成和重述功能。
    
    属性：
        rephrase_enabled: 是否启用LLM重述
        rephrase_model: 重述使用的LLM模型名称（引用endpoints.yml中的models）
        rephrase_style: 重述风格 (friendly/professional/casual/empathetic)
        rephrase_threshold: 触发重述的最小文本长度
        preserve_slots: 重述时是否保留槽位占位符
        language: 目标语言
    """
    rephrase_enabled: bool = False
    rephrase_model: Optional[str] = None  # 引用endpoints.yml中的models名称
    rephrase_style: str = "friendly"
    rephrase_threshold: int = 10
    preserve_slots: bool = True
    language: str = "zh"
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "NLGConfig":
        """从字典创建NLG配置"""
        config = _resolve_env_vars(config)
        
        rephrase_config = config.get("rephrase", {})
        
        return cls(
            rephrase_enabled=rephrase_config.get("enabled", False),
            rephrase_model=rephrase_config.get("model"),
            rephrase_style=rephrase_config.get("style", "friendly"),
            rephrase_threshold=rephrase_config.get("threshold", 10),
            preserve_slots=rephrase_config.get("preserve_slots", True),
            language=rephrase_config.get("language", "zh"),
        )


@dataclass
class VectorStoreConfig:
    """向量存储连接配置
    
    存储传递给 retriever.connect() 的连接参数。
    检索器类名从 config.yml 的 policies.EnterpriseSearchPolicy.vector_store 读取。
    
    属性：
        config: 连接配置字典（如数据库地址、认证信息等）
    """
    config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "VectorStoreConfig":
        """从字典创建配置"""
        config = _resolve_env_vars(config)
        return cls(config=config)
    
    def to_connect_config(self) -> Dict[str, Any]:
        """转换为 retriever.connect() 使用的配置字典"""
        return self.config.copy()


@dataclass
class MCPServerConfig:
    """单个 MCP Server 连接配置（endpoints.yml 的 mcp.servers.<name>）。

    属性：
        base_url:          MCP Server 完整请求 URL（如 http://127.0.0.1:8765/mcp）
        timeout:           单次 HTTP 调用超时（秒）
        retry:             网络层错误重试次数（不含首次）
        failure_threshold: 熔断器连续失败阈值
        reset_timeout:     熔断器 OPEN 维持秒数后转 HALF_OPEN
    """
    base_url: str = ""
    timeout: float = 10.0
    retry: int = 2
    failure_threshold: int = 5
    reset_timeout: float = 30.0

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MCPServerConfig":
        """从字典创建 MCP Server 配置。"""
        config = _resolve_env_vars(config)
        cb = config.get("circuit_breaker", {}) or {}
        return cls(
            base_url=config.get("base_url", ""),
            timeout=config.get("timeout", 10.0),
            retry=config.get("retry", 2),
            failure_threshold=cb.get("failure_threshold", 5),
            reset_timeout=cb.get("reset_timeout", 30.0),
        )


@dataclass
class MCPConfig:
    """MCP 工具协议层配置（endpoints.yml 的 mcp 节，SPEC §3.6）。

    enabled 为 False 时 ToolRegistry 全走本地 Action 直调，等价基线（SPEC §6.4）。
    """
    enabled: bool = False
    servers: Dict[str, MCPServerConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MCPConfig":
        """从字典创建 MCP 配置。"""
        config = _resolve_env_vars(config)
        servers_dict: Dict[str, MCPServerConfig] = {}
        for name, server_config in (config.get("servers", {}) or {}).items():
            servers_dict[name] = MCPServerConfig.from_dict(server_config or {})
        return cls(
            enabled=config.get("enabled", False),
            servers=servers_dict,
        )

    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """按名取 server 配置。"""
        return self.servers.get(name)


@dataclass
class ShortTermMemoryConfig:
    """短期记忆配置（config.yml 的 memory.short_term，SPEC §4.5）。

    原始轮次超过 max_raw_turns 时，保留最近 keep_recent_turns 轮完整记录，
    前面历史由 LLM 压缩为一段摘要写入 session_summary 槽位。
    """
    enabled: bool = False
    max_raw_turns: int = 20
    keep_recent_turns: int = 10
    llm: str = "command"  # 引用 endpoints.yml 中 models.<name>，摘要不需强推理

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ShortTermMemoryConfig":
        config = _resolve_env_vars(config)
        return cls(
            enabled=config.get("enabled", False),
            max_raw_turns=config.get("max_raw_turns", 20),
            keep_recent_turns=config.get("keep_recent_turns", 10),
            llm=config.get("llm", "command"),
        )


@dataclass
class LongTermMemoryConfig:
    """长期记忆配置（config.yml 的 memory.long_term，SPEC §4.5）。

    graph_database 为 Neo4j 独立 database 名，与 GraphRAG 隔离。
    """
    enabled: bool = False
    idle_timeout_minutes: int = 30
    graph_database: str = "user_memory"
    llm: str = "command"
    realtime_extract: bool = True
    end_of_session_extract: bool = True

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LongTermMemoryConfig":
        config = _resolve_env_vars(config)
        return cls(
            enabled=config.get("enabled", False),
            idle_timeout_minutes=config.get("idle_timeout_minutes", 30),
            graph_database=config.get("graph_database", "user_memory"),
            llm=config.get("llm", "command"),
            realtime_extract=config.get("realtime_extract", True),
            end_of_session_extract=config.get("end_of_session_extract", True),
        )


@dataclass
class MemoryConfig:
    """记忆系统配置（config.yml 的 memory 节，SPEC §4.5）。

    两个子开关任一为 False 时对应 hook 直接 no-op（SPEC §6.4）。
    enabled 为 False 时整体记忆关闭，等价基线。
    """
    short_term: ShortTermMemoryConfig = field(default_factory=ShortTermMemoryConfig)
    long_term: LongTermMemoryConfig = field(default_factory=LongTermMemoryConfig)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MemoryConfig":
        config = _resolve_env_vars(config)
        return cls(
            short_term=ShortTermMemoryConfig.from_dict(config.get("short_term", {})),
            long_term=LongTermMemoryConfig.from_dict(config.get("long_term", {})),
        )

    @property
    def enabled(self) -> bool:
        """任一子模块启用即视为记忆系统启用。"""
        return self.short_term.enabled or self.long_term.enabled


@dataclass
class AtguiguConfig:
    """atguigu_ai主配置类
    
    管理系统的所有配置项，支持从YAML文件加载。
    
    属性：
        llm: LLM配置
        embeddings: 向量化配置
        tracker_store: Tracker存储配置
        retrieval: 检索配置
        pipeline: Pipeline组件配置列表
        policies: 策略配置列表
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    tracker_store: TrackerStoreConfig = field(default_factory=TrackerStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    pipeline: List[Dict[str, Any]] = field(default_factory=list)
    policies: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "zh"
    assistant_id: str = "atguigu_assistant"
    
    @classmethod
    def load(cls, config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> "AtguiguConfig":
        """从配置文件加载配置
        
        参数：
            config_path: 配置文件路径
            
        返回：
            AtguiguConfig实例
            
        异常：
            ConfigurationException: 配置文件不存在或格式错误
        """
        path = Path(config_path)
        
        if not path.exists():
            raise MissingConfigException(f"配置文件不存在: {path}")
        
        try:
            config_dict = read_yaml_file(path)
        except Exception as e:
            raise ConfigurationException(f"配置文件格式错误: {e}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "AtguiguConfig":
        """从字典创建配置
        
        参数：
            config: 配置字典
            
        返回：
            AtguiguConfig实例
        """
        return cls(
            llm=LLMConfig.from_dict(config.get("llm", {})),
            embeddings=EmbeddingsConfig.from_dict(config.get("embeddings", {})),
            tracker_store=TrackerStoreConfig.from_dict(config.get("tracker_store", {})),
            retrieval=RetrievalConfig.from_dict(config.get("retrieval", {})),
            pipeline=config.get("pipeline", []),
            policies=config.get("policies", []),
            language=config.get("language", "zh"),
            assistant_id=config.get("assistant_id", "atguigu_assistant"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        返回：
            配置字典
        """
        from dataclasses import asdict
        return asdict(self)


@dataclass
class EndpointsConfig:
    """端点配置类
    
    管理外部服务端点配置，从endpoints.yml加载。
    
    属性：
        tracker_store: Tracker存储配置
        vector_store: 向量存储配置（知识检索后端）
        nlg: NLG配置（响应重述等）
        models: LLM模型配置字典，key为模型名称
        embeddings: 嵌入模型配置字典，key为模型名称
        mcp: MCP工具协议层配置（SPEC §3.6）
    """
    tracker_store: TrackerStoreConfig = field(default_factory=TrackerStoreConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    nlg: Optional[NLGConfig] = None
    models: Dict[str, LLMConfig] = field(default_factory=dict)
    embeddings: Dict[str, EmbeddingsConfig] = field(default_factory=dict)
    mcp: Optional[MCPConfig] = None
    
    @classmethod
    def load(cls, endpoints_path: Union[str, Path] = DEFAULT_ENDPOINTS_PATH) -> "EndpointsConfig":
        """从端点配置文件加载配置"""
        path = Path(endpoints_path)
        
        if not path.exists():
            # 端点配置文件是可选的，不存在时返回默认配置
            return cls()
        
        try:
            config_dict = read_yaml_file(path)
        except Exception as e:
            raise ConfigurationException(f"端点配置文件格式错误: {e}")
        
        return cls.from_dict(config_dict or {})
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EndpointsConfig":
        """从字典创建配置"""
        # 解析models配置
        models_dict = {}
        models_config = config.get("models", {})
        for model_name, model_config in models_config.items():
            models_dict[model_name] = LLMConfig.from_dict(model_config)
        
        # 解析embeddings配置
        embeddings_dict = {}
        embeddings_config = config.get("embeddings", {})
        for embed_name, embed_config in embeddings_config.items():
            embeddings_dict[embed_name] = EmbeddingsConfig.from_dict(embed_config)
        
        # 解析NLG配置
        nlg_config = None
        if "nlg" in config:
            nlg_config = NLGConfig.from_dict(config.get("nlg", {}))

        # 解析MCP配置（SPEC §3.6）
        mcp_config = None
        if "mcp" in config:
            mcp_config = MCPConfig.from_dict(config.get("mcp", {}))

        # 解析vector_store配置
        vector_store_config = VectorStoreConfig.from_dict(
            config.get("vector_store", {})
        )

        return cls(
            tracker_store=TrackerStoreConfig.from_dict(
                config.get("tracker_store", {})
            ),
            vector_store=vector_store_config,
            nlg=nlg_config,
            models=models_dict,
            embeddings=embeddings_dict,
            mcp=mcp_config,
        )
    
    def get_model_config(self, model_name: str) -> Optional[LLMConfig]:
        """根据模型名称获取LLM配置
        
        参数：
            model_name: 模型名称（在endpoints.yml中定义的key）
            
        返回：
            LLMConfig实例，如果不存在则返回None
        """
        return self.models.get(model_name)
    
    def get_embeddings_config(self, embeddings_name: str) -> Optional[EmbeddingsConfig]:
        """根据名称获取嵌入模型配置
        
        参数：
            embeddings_name: 嵌入模型名称（在endpoints.yml中定义的key）
            
        返回：
            EmbeddingsConfig实例，如果不存在则返回None
        """
        return self.embeddings.get(embeddings_name)

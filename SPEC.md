# LLM 智能客服系统改造 SPEC

> 版本：v1.0  
> 日期：2026-08-31  
> 基线：main 分支 Initial commit  
> 改造分支：dev

---

## 1. 概述

### 1.1 改造背景

现有项目已实现一套基于 LangGraph 的企业级 LLM 智能客服系统，具备图式编排（5 节点循环）、Command/Action 解耦、栈式对话上下文管理、YAML Flow 流程引擎、策略多级降级、GraphRAG 知识检索等核心能力。

本次改造在**不破坏既有架构与执行链路**的前提下，引入三个现代化 Agent 能力：

| 模块 | 定位 | 核心收益 |
|---|---|---|
| **MCP 工具协议层** | 将业务 Action 从同进程函数调用提升为跨进程、跨语言的标准协议接口 | 工具解耦、独立部署、动态发现、多团队协作 |
| **记忆系统** | 短期对话压缩 + 跨会话图谱长期记忆 | 个性化能力、指代消歧、行为偏好继承 |
| **Flow LLM 生成工具** | 开发期 CLI，自然语言→Flow YAML 草稿+校验 | 降低声明式流程的编写门槛 |

**已砍掉的方案**：Skill 技能包架构（当前规模收益过小）、YAML Schema/IDE 补全、Dify/Coze 画布接入。

### 1.2 核心设计原则

1. **向后兼容**：所有改造为增量式，既有 ecs_demo 不加任何配置仍可原路径运行
2. **可选启用**：MCP/记忆/生成工具通过 endpoints.yml / config.yml 开关控制，默认关闭即基线行为
3. **多路径保障**：MCP 失败自动降级回本地直调，记忆写入失败不影响对话主链路
4. **复用基础设施**：记忆复用 Neo4j、生成工具复用现有 LLM 配置、校验复用 Trainer._validate

---

## 2. 改造范围与决策记录

| 决策点 | 结论 | 理由 |
|---|---|---|
| MCP 协议兼容性 | 自研 JSON-RPC 2.0 语义，不追求与官方规范 100% 兼容 | 项目内部够用，面试讲 MCP 核心思想已成立 |
| MCP 传输方式 | HTTP（FastAPI 服务端 + httpx 客户端） | 跨语言、独立部署，支撑未来 Java 业务服务演进故事 |
| MCP 工具范围 | 本期仅包装现有电商 Action（订单/售后/物流） | 范围可控，验证链路完整 |
| Skill 技能包 | 砍掉 | 3 个技能规模下运行时无变化，仅文件组织重构收益不匹配成本 |
| YAML DX（Schema/IDE 补全） | 砍掉 | 手写 YAML 场景下改善几乎为零 |
| Dify/Coze 画布接入 | 不可行，砍掉 | 画布绑定私有 DSL，映射成本高且破坏自研架构叙事 |
| 长期记忆存储 | Neo4j 图谱（新建独立 database 做隔离） | 客服场景记忆是结构化关系，Cypher 精确查询更可控；与 GraphRAG 共用基础设施 |
| 记忆写入时机 | 显式信号实时提取 + 会话结束批量兜底 | 平衡时效性和 token 成本 |
| Flow 生成方式 | 开发期 CLI 工具 + LLM 提示词模板 + 校验闭环 | 声明式 DSL 闭集可靠生成，校验复用现有 Trainer 逻辑 |
| 实施顺序 | MCP → 记忆系统 → Flow 生成工具 | MCP 是工具层地基，记忆影响提示词链路，生成工具独立性最强 |

---

## 3. 模块一：MCP（自研工具协议层）

### 3.1 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    Agent 进程 (Python)                    │
│  ┌─────────────┐    ┌──────────────────────┐             │
│  │ action_node │───→│   ToolRegistry       │             │
│  │ (LangGraph) │    │  ├─ 本地 Action 注册  │             │
│  └─────────────┘    │  └─ MCP Client 发现   │             │
│                     └────────┬─────────────┘             │
│                              │ 命名空间路由               │
│                     ┌────────▼─────────────┐             │
│                     │   MCPClient (httpx)  │             │
│                     │  - initialize 协商    │             │
│                     │  - tools/list 拉取    │             │
│                     │  - tools/call 调用    │             │
│                     │  - 超时/重试/熔断      │             │
│                     └────────┬─────────────┘             │
└──────────────────────────────┼─────────────────────────────┘
                               │ HTTP (JSON-RPC 2.0)
          ┌────────────────────▼────────────────────┐
          │         ecommerce-mcp-server (Python)     │
          │  FastAPI 启动：/mcp POST                  │
          │  内部复用现有 Action 实例 + DB Session     │
          │  暴露：tools/list、tools/call、tools/get   │
          └───────────────────────────────────────────┘
                      (未来可替换为 Java 实现，
                       Agent 端零代码改动)
```

### 3.2 协议设计（JSON-RPC 2.0 语义）

#### 3.2.1 能力协商（initialize）

**请求 →**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "atguigu-mcp/1.0",
    "capabilities": {
      "tools": {}
    },
    "clientInfo": { "name": "atguigu-agent", "version": "1.0.0" }
  }
}
```

**响应 ←**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "atguigu-mcp/1.0",
    "capabilities": { "tools": {} },
    "serverInfo": { "name": "ecommerce-mcp", "version": "1.0.0" }
  }
}
```

#### 3.2.2 工具列表（tools/list）

**响应 ←**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "query_order",
        "description": "按条件查询用户的订单列表，返回可选择的订单按钮数据",
        "inputSchema": {
          "type": "object",
          "properties": {
            "user_id": { "type": "string", "description": "用户ID" },
            "goto": {
              "type": "string",
              "enum": [
                "action_ask_order_id_before_completed_3_days",
                "action_ask_order_id_before_delivered",
                "action_ask_order_id_before_shipped",
                "action_ask_order_id_shipped",
                "action_ask_order_id_shipped_delivered",
                "action_ask_order_id_after_delivered"
              ],
              "description": "订单状态过滤条件"
            }
          },
          "required": ["user_id", "goto"]
        }
      }
    ]
  }
}
```

#### 3.2.3 工具调用（tools/call）

**请求 →**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "query_order",
    "arguments": { "user_id": "1001", "goto": "action_ask_order_id_before_delivered" }
  }
}
```

**响应 ←**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "responses",
        "data": [
          {
            "text": "请选择订单",
            "buttons": [
              { "title": "[待发货]订单ID：O001\n- iPhone 15 × 1", "payload": "/SetSlots(order_id=O001)" }
            ]
          }
        ]
      },
      {
        "type": "slot_sets",
        "data": { "order_id": "false" }
      },
      {
        "type": "reject_action_listen",
        "data": true
      }
    ],
    "isError": false
  }
}
```

**说明**：`result.content` 复用 `ActionResult` 的语义（responses / events / slot 副作用），扁平化成一组 `type+data` 条目。这样 MCP 包装层只需做简单的序列化/反序列化，Action.run 内部逻辑零改动。

### 3.3 工具命名空间与注册策略

- MCP 工具注册进 `ToolRegistry` 时加前缀：`ecommerce__query_order`，防止与本地内置 Action 重名
- Flow YAML 和 Domain 中**无需改动引用名**（仍写 `action_ask_order_id`）：由包装层在 MCP 可用时替换执行路径

#### ToolRegistry 路由优先级

```
用户请求 action_ask_order_id
    │
    ▼
1. MCP 映射表查询：action_ask_order_id → ecommerce__query_order？
    ├─ 是 → MCPClient 调用
    │        ├─ 成功 → 返回结果
    │        └─ 失败（超时/熔断）→ 降级到第 2 步（本地直调）
    └─ 否 → 进入第 2 步
    │
    ▼
2. 本地 Action 注册表（_CUSTOM_ACTIONS → _BUILTIN_ACTIONS → utter_）
```

### 3.4 现有电商 Action → MCP 工具映射表

| 本地 Action 名 | MCP 工具名（含命名空间） | 参数来源 |
|---|---|---|
| `action_ask_order_id` | `ecommerce__query_order` | user_id、goto（槽位） |
| `action_get_order_detail` | `ecommerce__get_order_detail` | order_id（槽位） |
| `action_ask_receive_id` | `ecommerce__query_receive_info` | user_id（槽位） |
| `action_ask_receive_province` | `ecommerce__list_provinces` | 无 |
| `action_ask_receive_city` | `ecommerce__list_cities` | receive_province（槽位） |
| `action_ask_receive_district` | `ecommerce__list_districts` | receive_city（槽位） |
| `action_ask_set_receive_info` | `ecommerce__update_receive_info` | order_id + 所有收货信息槽位 |
| `action_cancel_order` | `ecommerce__cancel_order` | order_id（槽位） |
| （物流/售后同理） | | |

### 3.5 目录与文件新增

```
atguigu_ai/mcp/
├── __init__.py
├── protocol.py          # JSON-RPC 2.0 消息封装（request/response/error）
├── client.py            # MCPClient：initialize / list / call、超时、重试
├── server.py            # FastAPI MCP Server 基类：路由注册、工具包装
├── tool_registry.py     # ToolRegistry：命名空间路由、降级策略
└── exceptions.py        # MCPError、TimeoutError、FallbackTriggered
ecs_demo/
└── ecommerce_mcp_server.py   # 入口：python -m ecs_demo.ecommerce_mcp_server
```

### 3.6 配置扩展（endpoints.yml）

```yaml
# 新增 mcp 节
mcp:
  enabled: true
  servers:
    ecommerce:
      base_url: "http://127.0.0.1:8765/mcp"
      timeout: 10
      retry: 2
      circuit_breaker:
        failure_threshold: 5
        reset_timeout: 30
```

---

## 4. 模块二：记忆系统（短期 + 长期）

### 4.1 整体链路

```
用户消息进入
    │
    ▼
[understand 节点前插入 hook]
    ├─ ① 长期记忆召回（按 sender_id 查 Neo4j）
    │     → 用户画像注入 command_generator 提示词
    │     → 指代消歧："上次那个订单" → 图谱中用户最近提及的 order_id
    │
    ▼
正常 LangGraph 循环执行
    │
    ▼
[每轮 action 执行后 hook]
    ├─ ② 实时提取：检测显式记忆信号
    │     ("记住我地址/以后用顺丰/我的偏好是…")
    │     → LLM 结构化抽取 → 写入 Neo4j 图谱
    │
    ▼
[response 节点后，save tracker 之前]
    ├─ ③ 短期压缩：对话轮次 > 阈值时
    │     → LLM 摘要旧轮次 → 写入 session_summary 槽位
    │
    └─ ④ 会话结束判定（空闲超时/显式结束）
          → LLM 批量提取事实 → 写入 Neo4j
```

### 4.2 短期记忆：历史压缩

#### 触发条件

- 配置项：`memory.short_term.max_raw_turns: 20`（可配）
- Tracker.turns 原始轮次超过阈值时，保留最近 K 轮（如 10 轮）完整记录，前面历史压缩为一段摘要

#### 数据结构（写入 Tracker 槽位与事件）

```python
# Tracker 新增保留槽位（声明在 builtin slots）
session_summary: str      # 压缩摘要文本
summary_covered_turns: int  # 摘要覆盖的起始轮次编号，避免重复压缩

# 产生事件
{
  "event": "memory_compressed",
  "from_turn": 1,
  "to_turn": 11,
  "summary": "用户查询了订单O001，对配送时效不满，要求优先安排发货。"
}
```

#### 提示词模板（给 LLM 做摘要）

```
你是对话摘要器。请将以下多轮对话压缩为一段不超过200字的摘要，保留：
1) 正在执行或已完成的业务流程
2) 用户表达的偏好、情绪、诉求
3) 关键实体（订单号、商品、人名等）
忽略问候语和确认性重复。

=== 对话历史 ===
{% for turn in turns_to_compress %}
[用户] {{ turn.user_message.text }}
[助手] {{ turn.bot_messages | map(attribute='text') | join(' / ') }}
{% endfor %}
=== 请输出摘要 ===
```

### 4.3 长期记忆：Neo4j 图谱

#### 数据模型（独立 database：`user_memory`，与 GraphRAG 隔离）

```cypher
// 核心实体
(:User {user_id: string})
(:Preference {type: string, value: string, confidence: float, source: string, updated_at: datetime})
(:Address {label: string, province: string, city: string, district: string, street: string, phone: string, contact: string, is_default: bool})
(:OrderRef {order_id: string, mentioned_at: datetime, context: string})

// 关系
(:User)-[:偏好 {created_at: datetime, confidence: float}]->(:Preference)
(:User)-[:常用地址 {tag: string}]->(:Address)
(:User)-[:提及 {turn_id: int}]->(:OrderRef)
```

#### 记忆写入：实时提取（显式信号）

触发关键词/句式（可正则粗筛 + LLM 细确认）：

- `"记住我xxx"` / `"以后xxx"` / `"默认xxx"` / `"别再问我xxx"` → 偏好类
- `"我的地址是xxx"` / `"就寄到这里"` → 地址类
- `"就是刚才那个"` / `"上次说的订单"` → 提及类（记录 order_id 关联）

LLM 结构化抽取提示词：

```
你是记忆抽取器。从用户最后一句话中抽取结构化事实，输出 JSON 数组。
支持的事实类型：
- preference: {type: string, value: string, confidence: 0.0~1.0}
- address: {label?, province, city, district, street, phone?, contact?, is_default?}
- order_ref: {order_id: string, context: string}

用户当前对话上下文（含最近3轮）：
{{ recent_turns }}

用户最新输入：
{{ latest_message }}

输出 JSON 数组，无事实返回 []。只输出 JSON。
```

#### 记忆写入：会话结束兜底

会话结束判定：
- 显式 `/restart` 或用户说"再见/结束"
- 空闲超时（配置项 `memory.long_term.idle_timeout_minutes: 30`，Tracker 最后更新时间判断）

兜底提取提示词对完整会话跑一次，产出事实合并入库，与已有事实按 **相似度 + 时间** 做去重/覆盖（新的高置信度偏好覆盖旧的低置信度）。

#### 记忆召回与注入

在 `understand_node` 入口处，**command_generator.generate 调用之前**插入：

```python
# 1. 查用户画像摘要（偏好+默认地址），注入系统提示
user_profile = await long_term_memory.get_user_profile(sender_id)
# → "用户长期偏好：快递公司=顺丰；默认收货地址标签=公司（北京朝阳区…）"

# 2. 最近提及的订单/实体，用于指代消歧
recent_mentions = await long_term_memory.get_recent_mentions(sender_id, limit=5)
# → 若用户说"上次那个订单"，从 recent_mentions 中取最近 order_id
#    注入 command_generator 提示词的 context 部分：
#    "【参考记忆】用户最近提及订单：O001（30分钟前）、O002（昨天）"
```

### 4.4 目录与文件新增

```
atguigu_ai/memory/
├── __init__.py
├── short_term/
│   ├── __init__.py
│   ├── compressor.py       # 历史压缩：LLM 摘要+槽位写回
│   └── prompts/
│       └── compress.jinja2
├── long_term/
│   ├── __init__.py
│   ├── graph_store.py      # Neo4j 图谱 CRUD（用户/偏好/地址/提及）
│   ├── extractor.py        # LLM 结构化抽取（实时+兜底两个模板）
│   ├── recaller.py         # 画像召回 + 指代消歧
│   └── prompts/
│       ├── extract_realtime.jinja2
│       └── extract_end_of_session.jinja2
└── hooks.py                # understand_node_before / after_action / save_tracker_before 三处挂接
```

### 4.5 配置扩展

```yaml
# config.yml 新增
memory:
  short_term:
    enabled: true
    max_raw_turns: 20
    keep_recent_turns: 10
    llm: command      # 复用 endpoints.yml 中 command LLM（摘要不需要强推理）

  long_term:
    enabled: true
    idle_timeout_minutes: 30
    graph_database: "user_memory"   # Neo4j database 名，与 GraphRAG 隔离
    llm: command
    realtime_extract: true
    end_of_session_extract: true
```

---

## 5. 模块三：Flow LLM 生成工具

### 5.1 定位

- **运行期角色**：无。纯开发期 CLI，不进入 Agent 主链路
- **使用方式**：`atguigu flow-generate "用户说改收货地址，先让他选一个未签收的订单，显示详情，再选改姓名/电话/地址，可多次修改，最后确认入库"`

### 5.2 CLI 流程

```
用户描述需求
    │
    ▼
1. 读取现有的 domain 定义（槽位类型、已有 Action 列表）
   ↓ 作为生成的可用素材
2. Few-shot 提示词（附带 flow_order.yml 中 1~2 个真实 Flow 做示例）
   ↓ LLM 生成
3. YAML 草稿输出到 stdout + 临时文件
    │
    ▼
4. 调用 Trainer._validate(domain, flows=草稿) 校验
   ├─ 通过 → 提示"校验通过，写入 ecs_demo/data/flows/gen_xxx.yml？[Y/n]"
   └─ 失败 → 输出错误 → 把错误塞进提示词 → 最多重试 2 次自动修复
```

### 5.3 提示词模板核心结构

```
你是 Flow YAML 生成器，生成 atguigu_ai 框架兼容的 YAML Flow 定义。

=== 可用素材 ===
【槽位定义】
{{ domain_slots | to_yaml }}

【可用 Action 列表】
{{ available_actions | to_yaml }}

=== 语法约束 ===
- 仅支持 7 种 step 类型：action / collect / set_slots / condition / link / call / end
- collect 步骤引用的槽位必须出现在【槽位定义】中
- action 步骤名必须出现在【可用 Action 列表】中
- 步骤跳转 target（next: / if: then: / id: 引用）必须指向存在的步骤 id
- ask_before_filling: true 用于每次要清空槽位重填

=== 示例 ===
（附 flow_order.yml 中 query_order_detail + cancel_order 两个完整示例）

=== 用户需求 ===
{{ user_prompt }}

=== 输出要求 ===
1. 只输出一个 Flow 的 YAML 定义
2. 以 ```yaml 代码块包裹，不要其他说明
3. flow_id 用下划线小写命名
4. 给每个步骤的 collect/action 加一行注释说明该步骤的意图
```

### 5.4 目录与文件新增

```
atguigu_ai/cli/
└── flow_generate.py        # CLI 命令：@app.command("flow-generate")
atguigu_ai/training/
└── flow_generator/
    ├── __init__.py
    ├── generator.py        # LLM 调用+重试修复循环
    └── prompts/
        └── flow_generate.jinja2
```

### 5.5 与现有 Trainer 的协作

- 直接复用 `Trainer._validate(domain, flows)`，新增的 `FlowGenerator` 调用它作为生成质量门
- 若当前项目目录下 flows/ 已存在示例，优先用项目内真实 Flow 做 few-shot（更贴合业务），否则用框架内置示例

---

## 6. 整体集成设计

### 6.1 LangGraph 节点与 hook 改动点

**现有图不变**（START→understand→policy→action→guard→...），**通过在 understand 节点首尾、save tracker 之前插入 hook 函数** 实现记忆注入与写入，避免拆图：

```python
# atguigu_ai/memory/hooks.py
async def before_understand(tracker, command_generator_config, long_term_recaller):
    # 召回画像+指代，写入 tracker 临时上下文（metadata 里），供 command_generator 提示词渲染
    ...

async def after_each_action(tracker, action_result):
    # 实时记忆提取（只看本轮用户输入+action结果）
    ...

async def before_save_tracker(tracker, short_term_compressor, long_term_extractor):
    # 短期压缩判断 + 会话结束兜底提取判断
    ...
```

hook 在 `agent.py:Agent.handle_message` 中显式调用，保持可观测。

### 6.2 ToolRegistry 与 action_node 的集成

`action_node` 中当前使用 `get_action(action_name)` → 改为 `tool_registry.get(action_name)` 返回一个**统一接口 Executable**：

```python
class Executable(ABC):
    async def run(tracker, domain, **kwargs) -> ActionResult: ...

# 两种实现：
class LocalExecutable(Executable):   # 包装现有 Action 实例
class MCPExecutable(Executable):     # 调 MCPClient.tools/call，失败触发 LocalExecutable 降级
```

这样 action_node 中调用逻辑零变更，仅替换一处工厂。

### 6.3 目录结构变更总览

```
atguigu_ai/
├── mcp/                     # (新增) MCP 协议、Client、Server、ToolRegistry
├── memory/                  # (新增) short_term + long_term + hooks
├── training/flow_generator/ # (新增) Flow LLM 生成工具
└── cli/flow_generate.py     # (新增) CLI 入口
ecs_demo/
└── ecommerce_mcp_server.py  # (新增) 电商 MCP 服务入口
ecs_demo/data/flows/         # (不变) 原有 YAML
ecs_demo/domain/             # (不变) 原有槽位/回复/Action 声明
ecs_demo/actions/            # (不变) 原有 Action 代码，
                             #   e-commerce-mcp-server 直接 import 复用，
                             #   不复制一份
```

### 6.4 向后兼容策略

- 三个模块均通过 `enabled: true/false` 配置开关，默认全部 `false` → 运行时行为等价于基线
- MCP 关闭时：ToolRegistry 全部走本地分支
- 记忆关闭时：hook 函数直接 return，no-op
- 生成工具不进 Agent 主链路，没有开关概念

---

## 7. 实施顺序与里程碑

| 阶段 | 产出 | 预计工作量 | 验证标准 |
|---|---|---|---|
| **M1：MCP 协议层** | mcp/ 模块 + ecommerce_mcp_server + ToolRegistry 集成 + endpoints 配置 | ~3 天 | 原有 ecs_demo 跑通 `modify_order_receive_info` 全程，MCP 日志看到调用，手动停 MCP 服务后降级本地直调成功 |
| **M2：记忆系统** | memory/ 模块（short_term + long_term）+ hooks 注入 + config 配置 | ~3 天 | ① 30 轮对话后 session_summary 槽有内容；② 用户说"记住以后用顺丰"→ Neo4j 中出现偏好节点；③ 新开会话命令生成提示词里包含画像文本 |
| **M3：Flow 生成工具** | flow_generate CLI + 提示词模板 + 校验闭环复用 | ~1 天 | `atguigu flow-generate "取消订单流程"` → 产出 YAML，通过 Trainer._validate，人工对比与现有 cancel_order 结构一致 |
| **M4：集成测试+文档** | 三模块联调、README 启动说明、Git 提交规范合规 | ~1 天 | 三种开关组合（全关/全开/单开）的 e2e 冒烟测试均通过 |

### Git 提交约定

- 每阶段独立 PR（或独立 commit 集），不跨模块混合：`feat(mcp): ...` / `feat(memory): ...` / `feat(flow-gen): ...`
- 全部推送至 `dev` 分支；M4 完成后再 `git checkout main && git merge dev && git push`

---

## 8. 测试计划

### 8.1 MCP 模块

| 用例 | 期望 |
|---|---|
| 正常调用 query_order（goto=before_delivered） | 返回与本地直调完全相同的 responses/buttons |
| MCP 服务未启动 | 本地 Action 兜底，链路不中断，日志记录降级事件 |
| MCP 响应超时（>10s） | 重试 2 次后熔断，后续调用自动走本地；30s 后半开探测 |
| MCP 返回 isError=true | 降级到本地并日志告警 |

### 8.2 记忆模块

| 用例 | 期望 |
|---|---|
| 会话达 22 轮 | 前 12 轮压缩为 session_summary，保留最近 10 轮 |
| 用户说"记住我默认快递公司用顺丰" | Neo4j user_memory 中 User-[:偏好]->Preference 节点存在 |
| 新会话第一句话"我的订单什么时候发" | command_generator 提示词注入"用户长期偏好：快递公司=顺丰" |
| 用户提到"上次那个订单"后紧接"帮我改地址" | order_id 槽从记忆提及中自动消歧 |
| 记忆 Neo4j 连接失败 | 对话正常走，只打一条警告日志，不抛错 |

### 8.3 Flow 生成工具

| 用例 | 期望 |
|---|---|
| 需求："查询订单详情" | 生成的 Flow 结构、step id、goto 槽机制与 flow_order.yml 中 query_order_detail 语义一致 |
| 需求里用到了不存在的 Action | 自动修复循环最多 2 次后报错列出缺失项 |
| 生成的 YAML 跳转目标不存在 | Trainer._validate 捕获并提示 |

### 8.4 基线回归（必须过）

- **全配置关闭（默认）场景**：ecs_demo 三条主链路（查订单详情/改收货地址/取消订单）行为与 main 分支 baseline 逐响应一致

---

## 9. 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| MCP 工具命名空间映射遗漏 → 链路跑不起来 | 中 | M1 结束前做一张映射清单，单测逐条比对 |
| 记忆 LLM 提取幻觉 → 写入错误偏好 | 中 | 置信度字段 + 兜底写入时与已存节点按相似度合并，不直接覆盖；提供用户侧"忘记记忆"指令 Action |
| 短期压缩丢失关键状态（活跃 Flow 栈） | 低 | 压缩提示词明确要求保留"正在执行的流程名和已收集槽位"；压缩前序列化 dialogue_stack 快照存事件 |
| Flow 生成模板依赖框架内置示例，业务贴合度不够 | 低 | 项目 flows 存在时自动优先用项目真实 Flow 做 few-shot |

---

## 10. 面试叙事参考（简历 Q&A 准备）

1. **为什么自研 MCP 而不是接官方 SDK？**  
   → 核心是解耦思想：客服业务和 Agent 平台是不同团队交付的（甚至异构语言），统一协议让业务方按接口暴露工具，Agent 即插即用。协议借用 JSON-RPC 2.0 + tools/list/call 语义已覆盖 MCP 三要素（能力协商/工具发现/调用），够支撑内部平台。

2. **记忆为什么选图谱不选向量？**  
   → 客服记忆是结构化关系（用户→偏好→快递公司、用户→地址），Cypher 精确查询比向量 TOP-K 更可控；且和 GraphRAG 共用 Neo4j 基础设施，运维统一。

3. **声明式 Flow 引擎的好处？**  
   → 正因为流程是闭集声明式数据，①才能被 LLM 可靠生成（有校验）②未来可以做可视化画布拖拽（Dify/Coze 的路径）③ 非程序员产品可直接改 YAML 走流程。

---

## 11. 待确认项（无阻塞，默认值已取）

- [ ] 短期压缩 max_raw_turns 默认值 20、keep_recent_turns 默认 10 → 如要调整在 config.yml 覆盖
- [ ] 长期记忆 idle_timeout_minutes 默认 30 → 影响兜底提取触发时机
- [ ] MCP 服务默认端口 8765 → 冲突则在 endpoints.yml mcp.servers.ecommerce.base_url 修改

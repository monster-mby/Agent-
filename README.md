# Agent 智能助手平台

个人 AI Agent 项目，基于"技能优先（Skill-First）"架构，独立实现 ReAct、Reflection、RAG、GraphRAG、LLM 意图路由等 Agent 核心设计模式。

**技术栈**：Python 3.12 · LangGraph · LangChain · FastAPI · Streamlit · LiteLLM · ChromaDB · GraphRAG · Pydantic · SQLite

---

## 🏗️ 架构图

```
Streamlit 前端 ──→ FastAPI 网关 ──→ 编排器 (Orchestrator)
                                       │
                  ┌────────────────────┼──────────────────────┐
                  ▼                    ▼                      ▼
            LangGraph 图         单技能调用            Reflection
       (DualRetrievalGraph)  (skill_manager)         反思闭环
                  │
   ┌──────────────┼──────────────┐
   ▼              ▼              ▼
 12 活跃技能   Conversation   三层容错
(RAG 全链路)   Memory        (校验/重试/熔断)
```

---

## 1. Agent 核心设计模式

### ReAct Agent 循环

`src/agent/orchestrator.py` → `run_agent()`

LLM 自主 Think→Act→Observe 多轮决策循环：

```
user → assistant(tool_calls) → tool(执行结果) → assistant → ...
```

- `MessageHistory` 管理标准 `user → assistant → tool` 角色流转
- LLM 接收完整对话历史 + 工具列表，自主决策调哪个技能
- 支持 `max_turns` 限制 + 连续失败熔断（默认 3 次）
- 工具执行结果截断注入（单条 ≤ 4000 字符），防止 token 爆炸

### Reflection 反思模式

`src/agent/langgraph/graphs.py` → `build_reflection_graph()`

基于 LangGraph StateGraph 的 4 节点闭环：

```
GeneratorNode → ExternalFeedbackNode → CriticNode → [should_continue?]
                  ↑                                              │
                  │                                  ┌───────────┘
                  │                                  ▼
                  └────────── ReviserNode ←──────────┘ 继续
                                                        │
                                                        ▼ END 不达标终止
```

| 节点 | 职责 | 实现细节 |
|------|------|---------|
| GeneratorNode | 调目标技能生成初始结果 | 写入 `reflection_context.refined_output` |
| ExternalFeedbackNode | 采集外部反馈 | 不调 LLM，按技能类型分发策略：`rag_answer` 验证引用完整性、`code_explainer` 跑 flake8、默认通用质量评估 |
| CriticNode | LLM 低温 0.1 生成批评报告 | 输出结构化 JSON：`{points: [{severity, description, suggested_fix}], overall_score}` |
| should_continue | 条件边路由 | 三条件：质量分 ≥ 0.8 / 收敛检测（两轮 improvement_score < 阈值）/ 最大迭代 2 次 |
| ReviserNode | 基于批评修订 | 调 ReflectionSkill（temperature=0.5），`generation + critique → revised_generation`，自动计算 improvement_score |

### DAG 工作流编排

基于 LangGraph StateGraph，将知识库索引→检索等核心流水线配置化声明，支持顺序边、条件边（`build_should_continue`）、SQLite Checkpoint 持久化断点续跑。

### LLM 意图路由

`src/agent/orchestrator.py` → `_classify_intent_with_llm()`

基于 qwen-turbo 一次调用同时完成流水线分类 + RAG 检索模式判断，取代传统关键词匹配。返回 `{"pipeline": "...", "rag_mode": "vector"|"graphrag"}`，覆盖关键词无法处理的多意图并行场景。

### Function Calling 标准闭环

基于 LiteLLM 统一接口，完整流程：LLM 决策 → `tool_calls` 解析 → `skill_manager.call()` 执行 → 结果注入 `MessageHistory`（`role: tool`）→ 回传 LLM 继续推理。

---

## 2. RAG 能力

### 双检索引擎（DualRetrievalGraph）

`src/agent/langgraph/dual_retrieval_graph.py`

```
RulesInjector → QueryRewrite → Embed → VectorRetrieve → Rerank → ContextMerge → RagAnswer
```

- **RulesInjector**：作为图首个节点，从 RulesEngine 提取会话规则注入 System Prompt 前缀
- **QueryRewrite**：指代消解 / 歧义消除 / 多意图拆分（qwen-turbo）
- **Embed**：文本向量化，缓存去重 + 批处理
- **VectorRetrieve**：ChromaDB 按知识库 namespace 隔离检索
- **Rerank**：阿里云百炼 gte-rerank 精排模型
- **ContextMerge**：拼接规则前缀 + 精排文档 + 对话历史摘要（ConversationMemory 70/30 混合）
- **RagAnswer**：LLM 生成带引用的回答 + 置信度估算
- 支持 `retrieval_sources` 引用溯源和 `applied_rules` 规则可见

### GraphRAG 知识图谱

`src/skills/custom/rag_skills/graphrag_indexer/` + `graphrag_searcher/`

- 实体/关系提取 → 社区聚类 → 三种检索模式：`local`（实体级精确检索）、`global`（社区级全局摘要）、`hybrid`（融合）
- 双路自动路由：日常问题走 vector，架构/关联类问题走 graphrag

---

## 3. 技能体系

4 层分层架构：常量层 → Pydantic 数据模型层 → 核心业务逻辑层 → 技能类层。12 个活跃技能通过 SkillManager 统一注册与调用，新增技能无需修改现有代码。

| 分类 | 技能 | 说明 |
|------|------|------|
| RAG 文档 | `document_loader` | 加载 17 种格式的本地文档 |
| RAG 文档 | `document_chunker` | 语义边界切分（4 种策略） |
| RAG 文档 | `text_embedder` | 文本向量化，缓存去重+批处理 |
| RAG 检索 | `query_rewrite_skill` | 指代消解 / 歧义消除 |
| RAG 检索 | `rerank_skill` | gte-rerank 精排 |
| RAG 检索 | `rag_answer` | LLM 生成带引用+置信度的回答 |
| 知识图谱 | `graphrag_indexer` | 构建知识图谱索引 |
| 知识图谱 | `graphrag_searcher` | local/global/hybrid 检索 |
| 内容处理 | `text_summarizer` | 长文本摘要（启发式/TextRank/LLM） |
| 内容处理 | `translator` | 中英互译，多风格 |
| 代码 | `code_explainer` | 代码逐行解释 + 内嵌 Reflection 反思闭环 |
| 管理 | `kb_manager` | 知识库 CRUD + 文档上传索引（策略模式路由） |

---

## 4. 产品化基础设施

| 模块 | 文件 | 说明 |
|------|------|------|
| SessionManager | `src/infrastructure/session_manager.py` | 会话隔离 + 持久化，复用 LangGraph Checkpoint |
| RulesEngine | `src/infrastructure/rules_engine.py` | 管理会话级 LLM 行为规则，作为图首个节点注入 |
| VectorStoreManager | `src/infrastructure/vector_store.py` | ChromaDB namespace 隔离 + 线程安全单例 |
| ConversationMemory | `src/infrastructure/conversation_memory.py` | 消息 ≥ 20 条异步生成摘要→向量化→混合检索（知识库 70% + 摘要 30%） |
| IndexingPipeline | `src/infrastructure/indexing_pipeline.py` | document_loader → chunker → embedder 自动化索引 |

---

## 5. 三层容错架构

| 层 | 机制 | 
|----|------|
| 第一层 | Pydantic 输入输出校验 |
| 第二层 | 指数退避自动重试装饰器 |
| 第三层 | 熔断与降级（半开试探 / 自动恢复） |

---

## 📂 代码导航

```
src/
├── agent/
│   ├── orchestrator.py           ← 编排器主入口、ReAct 循环、意图路由
│   ├── langgraph/
│   │   ├── dual_retrieval_graph.py ← 生产 RAG 图（DualRetrievalGraph）
│   │   ├── graphs.py             ← Reflection 图 + 流水线
│   │   ├── nodes.py              ← 所有 LangGraph 节点（含 Reflection 4 节点）
│   │   └── state.py              ← GraphState、ReflectionContext、收敛检测
│   ├── llm_client.py             ← LLM 客户端 + tool_calls 格式清洗
│   └── engine.py                 ← Agent 引擎
├── skills/
│   ├── custom/rag_skills/        ← RAG 全链路（loader/chunker/embedder/rerank/...）
│   ├── custom/reflection_skills/ ← ReflectionSkill（Reviser 节点调用）
│   └── preset/                   ← 预设技能储备
├── api/
│   ├── routes/                   ← FastAPI 路由（会话/知识库/规则/研报）
│   ├── sse_stream.py             ← SSE 流式输出 + 中间状态推送
│   └── deps.py                   ← 依赖注入 + SkillManager 单例
├── infrastructure/               ← SessionManager / RulesEngine / VectorStore / ConversationMemory
└── core/                         ← 配置 + LiteLLM 客户端
```



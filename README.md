# Enterprise Learning Agent

个人 AI Agent 智能助手平台 —— 集成 RAG、GraphRAG、ReAct、Reflection 等 Agent 核心设计模式。

## 🏗️ 架构

```
Streamlit 前端  ──→  FastAPI 网关  ──→  编排器 (Orchestrator)
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
              LangGraph 图          单技能调用              API 直调
         (DualRetrievalGraph)   (skill_manager)     (stock/tech-radar)
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
  14 个活跃技能   Reflection    Conversation
  (RAG 全链路 +   反思闭环       Memory
   金融投研)                     对话记忆
```

## 🔑 核心技术

| 模块 | 关键文件 | 亮点 |
|------|---------|------|
| ReAct Agent | `src/agent/orchestrator.py` → `run_agent()` | Think→Act→Observe 多轮循环，MessageHistory 角色流转，连续失败熔断 |
| Reflection | `src/agent/langgraph/graphs.py` → `build_reflection_graph()` | 4 节点 LangGraph 闭环，LLM Critic→Reviser↻，收敛检测 |
| RAG 全链路 | `src/agent/langgraph/dual_retrieval_graph.py` | RulesInject→QueryRewrite→Embed→VectorRetrieve→Rerank→ContextMerge→RagAnswer |
| GraphRAG | `src/skills/custom/rag_skills/graphrag_*/` | 实体/关系提取→社区聚类→local/global/hybrid 三模式检索 |
| LLM 意图路由 | `src/agent/orchestrator.py` → `_classify_intent_with_llm()` | 一次 qwen-turbo 调用同时输出流水线 + RAG 模式 |
| 对话记忆 | `src/infrastructure/conversation_memory.py` | 长对话自动摘要→向量化→混合检索（知识库70%+摘要30%） |
| 技能体系 | `src/skills/` | 14 活跃技能 + 10 扩展储备，4 层分层架构，SkillManager 统一调度 |

## 📂 代码导航

```
src/
├── agent/
│   ├── orchestrator.py    ← Agent 主入口、ReAct 循环、意图路由
│   ├── langgraph/
│   │   ├── dual_retrieval_graph.py  ← 生产 RAG 图
│   │   ├── graphs.py      ← Reflection 图 + 流水线
│   │   ├── nodes.py       ← 所有 LangGraph 节点（含 Reflection 4 节点）
│   │   └── state.py       ← GraphState、ReflectionContext
│   └── engine.py          ← Agent 引擎
├── skills/
│   ├── custom/rag_skills/ ← RAG 全链路（loader/chunker/embedder/rerank/...）
│   └── custom/finance_skills/ ← 投研日报 + 技术雷达
├── api/                   ← FastAPI 路由 + SSE 流式
├── infrastructure/        ← SessionManager / RulesEngine / VectorStore / ConversationMemory
└── core/                  ← 配置 + LiteLLM 客户端
```

## 🚀 运行

```bash
poetry install
cp .env.example .env    # 填入 DASHSCOPE_API_KEY
streamlit run app.py --server.port 8501
```

## 🧪 测试

```bash
poetry run pytest -m "not real"   # 200+ 用例
```

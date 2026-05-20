import os
import sys

import pytest
from unittest.mock import patch, MagicMock

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.orchestrator import SkillOrchestrator


class TestRagPipeline:
    """测试增强型 RAG 全链路"""

    @pytest.fixture
    def orchestrator(self):
        """初始化编排器"""
        return SkillOrchestrator()

    def test_execute_rag_pipeline_success(self, orchestrator):
        """测试 RAG 链路成功执行（含 LLM 生成）"""

        mock_vec_result = {
            "results": [
                {"id": "1", "content": "Python是一种编程语言", "score": 0.9},
                {"id": "2", "content": "Java也是一种编程语言", "score": 0.8},
            ]
        }

        mock_rerank_result = {
            "reranked_docs": [
                {"doc_id": "1", "content": "Python是一种编程语言", "rerank_score": 0.95}
            ]
        }

        mock_rag_answer = {
            "answer": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布。",
            "citations": [{"citation_id": 1, "chunk_id": "1", "text_snippet": "Python是一种编程语言"}],
            "confidence": 0.85
        }

        with patch.object(orchestrator.skill_manager, 'call') as mock_call:
            def side_effect(skill_name, **kwargs):
                if skill_name == "query_rewrite_skill":
                    return {"skill": skill_name, "status": "success",
                            "output": {"rewritten_queries": ["Python是什么？"]}}
                elif skill_name == "vector_search":
                    return {"skill": skill_name, "status": "success", "output": mock_vec_result}
                elif skill_name == "rerank_skill":
                    return {"skill": skill_name, "status": "success", "output": mock_rerank_result}
                elif skill_name == "rag_answer":
                    return {"skill": skill_name, "status": "success", "output": mock_rag_answer}
                return {"skill": skill_name, "status": "error", "output": None}

            mock_call.side_effect = side_effect

            result = orchestrator.execute_rag_pipeline(query="它是什么语言？")

            assert result.success is True
            assert result.pipeline_type == "rag_enhanced"
            assert len(result.final_output["sources"]) > 0
            assert result.final_output["answer"] != ""
            assert "Python" in result.final_output["answer"]
            print(f"✅ RAG 链路测试通过 | 答案: {result.final_output['answer'][:50]}...")

    def test_execute_rag_pipeline_rerank_fallback(self, orchestrator):
        """测试当精排失败时的降级逻辑"""

        mock_vec_result = {
            "results": [
                {"id": "1", "content": "测试内容", "score": 0.5}
            ]
        }

        with patch.object(orchestrator.skill_manager, 'call') as mock_call:
            def side_effect(skill_name, **kwargs):
                if skill_name == "query_rewrite_skill":
                    return {"skill": skill_name, "status": "success",
                            "output": {"rewritten_queries": ["测试问题"]}}
                elif skill_name == "vector_search":
                    return {"skill": skill_name, "status": "success", "output": mock_vec_result}
                elif skill_name == "rerank_skill":
                    raise Exception("Rerank API Error")
                return {"skill": skill_name, "status": "error", "output": None}

            mock_call.side_effect = side_effect

            result = orchestrator.execute_rag_pipeline(query="测试")

            assert result.success is True
            assert len(result.final_output["sources"]) > 0
            print(f"✅ 降级逻辑测试通过 | 摘要: {result.summary}")

    def test_execute_smart_rag_vector_mode(self, orchestrator):
        """测试智能路由：强制 vector 模式"""

        with patch.object(orchestrator, 'execute_rag_pipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock(
                success=True,
                pipeline_type="rag_enhanced",
                final_output={"answer": "测试答案", "sources": []}
            )

            result = orchestrator.execute_smart_rag(query="Python是什么？", mode="vector")

            assert result.success is True
            assert mock_pipeline.called
            print(f"✅ Vector 路由测试通过")

    def test_execute_smart_rag_graphrag_mode(self, orchestrator):
        """测试智能路由：强制 graphrag 模式"""

        with patch.object(orchestrator.skill_manager, 'call') as mock_call:
            mock_call.return_value = {
                "skill": "graphrag_searcher",
                "status": "success",
                "output": {"answer": "GraphRAG 全局答案", "results": []}
            }

            result = orchestrator.execute_smart_rag(query="总结这个项目的技术架构", mode="graphrag")

            assert result.success is True
            assert result.pipeline_type == "macro_rag"
            mock_call.assert_called_once_with("graphrag_searcher", query="总结这个项目的技术架构", mode="global")
            print(f"✅ GraphRAG 路由测试通过")

    def test_execute_smart_rag_auto_mode_macro(self, orchestrator):
        """测试智能路由：auto 模式命中宏观关键词"""

        with patch.object(orchestrator.skill_manager, 'call') as mock_call:
            mock_call.return_value = {
                "skill": "graphrag_searcher",
                "status": "success",
                "output": {"answer": "宏观总结答案"}
            }

            # 【修改】加入冒号作为词边界隔断，满足正则 (?![一-龥]) 的要求
            result = orchestrator.execute_smart_rag(query="总结：这个项目的整体架构", mode="auto")

            assert result.success is True
            assert result.pipeline_type == "macro_rag"
            assert "graphrag" in result.pipeline_name
            print(f"✅ Auto 模式（宏观）路由测试通过")

    def test_execute_smart_rag_auto_mode_detail(self, orchestrator):
        """测试智能路由：auto 模式走细节查询"""

        with patch.object(orchestrator, 'execute_rag_pipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock(
                success=True,
                pipeline_type="rag_enhanced",
                final_output={"answer": "细节答案", "sources": []}
            )

            result = orchestrator.execute_smart_rag(query="Python 的列表推导式怎么用？", mode="auto")

            assert result.success is True
            mock_pipeline.assert_called_once()
            print(f"✅ Auto 模式（细节）路由测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

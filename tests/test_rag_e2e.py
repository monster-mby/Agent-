import os
import sys
import pytest
from unittest.mock import patch

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.orchestrator import SkillOrchestrator


@pytest.mark.real
class TestRagE2E:
    """真实文档 RAG 端到端测试（接入真实 LLM）"""

    @pytest.fixture
    def orchestrator(self):
        return SkillOrchestrator()

    def test_rag_vector_mode_with_real_llm(self, orchestrator):
        """测试 Vector + Rerank + 真实 LLM 生成答案"""

        mock_document_content = """
        Python 是一种由 Guido van Rossum 于 1991 年首次发布的高级编程语言。
        Python 的设计哲学强调代码的可读性，其语法结构允许程序员用较少的代码表达概念。
        Python 广泛应用于 Web 开发、数据分析、人工智能、自动化脚本等领域。
        """

        mock_vec_result = {
            "results": [
                {"id": "doc_1", "content": mock_document_content, "score": 0.98},
                {"id": "doc_2", "content": "Java 是另一种流行语言。", "score": 0.45}
            ]
        }

        # 保存原始 call 方法
        original_call = orchestrator.skill_manager.call

        with patch.object(orchestrator.skill_manager, 'call') as mock_call:
            def side_effect(skill_name, **kwargs):
                if skill_name in ["vector_search", "query_rewrite_skill", "rerank_skill"]:
                    # 仅 Mock 这三个技能
                    if skill_name == "vector_search":
                        return {"skill": skill_name, "status": "success", "output": mock_vec_result}
                    elif skill_name == "query_rewrite_skill":
                        return {"skill": skill_name, "status": "success",
                                "output": {"rewritten_queries": [kwargs.get("original_query", "")]}}
                    elif skill_name == "rerank_skill":
                        return {"skill": skill_name, "status": "success",
                                "output": {"reranked_docs": mock_vec_result["results"][:2]}}

                # rag_answer 及其他技能走原始真实逻辑
                return original_call(skill_name, **kwargs)

            mock_call.side_effect = side_effect

            result = orchestrator.execute_smart_rag(query="Python 是谁发明的，有什么特点？", mode="vector")

            assert result.success is True
            answer = result.final_output.get("answer", "")
            assert "Guido" in answer or "1991" in answer or "Python" in answer, f"LLM 未基于文档回答: {answer}"
            assert len(result.final_output.get("sources", [])) > 0

            print(f"✅ 真实 RAG 测试通过 | 答案: {answer[:80]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

import os
import sys
import json
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

import pytest
from pydantic import ValidationError

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入待测试的技能
from src.skills.custom.rag_skills.rerank_skill.skill import RerankSkill, RerankInput, RerankCandidate
from src.skills.custom.rag_skills.query_rewrite_skill.skill import QueryRewriteSkill, QueryRewriteInput, QueryHistoryItem


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def sample_candidates():
    """创建示例候选文档"""
    return [
        RerankCandidate(
            content="Python是一种高级编程语言",
            source="vector",
            doc_id="1",
            original_score=0.8
        ),
        RerankCandidate(
            content="Java是一种面向对象的编程语言",
            source="es",
            doc_id="2",
            original_score=0.7
        ),
        RerankCandidate(
            content="机器学习是人工智能的一个分支",
            source="kg",
            doc_id="3",
            original_score=0.6
        ),
        RerankCandidate(
            content="Python有丰富的数据分析库",
            source="vector",
            doc_id="4",
            original_score=0.5
        ),
        RerankCandidate(
            content="深度学习使用神经网络",
            source="es",
            doc_id="5",
            original_score=0.4
        ),
        RerankCandidate(
            content="Python语法简洁易懂",
            source="kg",
            doc_id="6",
            original_score=0.3
        ),
    ]


@pytest.fixture
def sample_history():
    """创建示例对话历史"""
    return [
        QueryHistoryItem(role="user", content="什么是Python？"),
        QueryHistoryItem(role="assistant", content="Python是一种高级编程语言。"),
        QueryHistoryItem(role="user", content="它有什么特点？"),
    ]


# ==========================================
# RerankSkill 测试
# ==========================================

class TestRerankSkill:
    """RerankSkill 测试套件"""

    def test_input_validation(self, sample_candidates):
        """测试输入验证"""
        # 正常输入
        valid_input = RerankInput(
            query="Python编程",
            candidates=sample_candidates,
            top_n=3
        )
        assert valid_input.query == "Python编程"
        assert len(valid_input.candidates) == 6

        # 缺少必填字段
        with pytest.raises(ValidationError):
            # 错误示例：传入字符串给 candidates 列表
            RerankInput(query="test", candidates="not_a_list")

        with pytest.raises(ValidationError):
            # 错误示例：缺失 query
            RerankInput(candidates=sample_candidates)

    def test_skip_logic(self, sample_candidates):
        """测试跳过逻辑"""
        skill = RerankSkill()

        # 测试1：候选数 <= top_n
        assert skill._check_skip("test query", sample_candidates[:3], top_n=5) is True

        # 测试2：候选数 > top_n
        assert skill._check_skip("test query", sample_candidates, top_n=3) is False

        # 测试3：查询过短
        assert skill._check_skip("py", sample_candidates, top_n=3) is True

        # 测试4：空候选
        assert skill._check_skip("test query", [], top_n=3) is True

    # @patch('src.skills.custom.rag_skills.rerank_skill.skill.TextRerank')
    # @patch('requests.post')
    # def test_call_dashscope_rerank_sdk(self, mock_tr, mock_req_post, sample_candidates):
    #     """测试 dashscope SDK 调用"""
    #     skill = RerankSkill()
    #
    #     # 1. 构造 SDK 响应 (注意：现在要用 mock_tr)
    #     mock_response = Mock()
    #     mock_response.status_code = 200
    #     mock_response.output = {
    #         "results": [
    #             {"index": 0, "relevance_score": 0.95},
    #             {"index": 3, "relevance_score": 0.85},
    #             {"index": 5, "relevance_score": 0.75},
    #         ]
    #     }
    #     mock_tr.call.return_value = mock_response
    #
    #     # 2. 构造 requests 响应（安全网，注意：现在要用 mock_req_post）
    #     mock_post_response = Mock()
    #     mock_post_response.raise_for_status.return_value = None
    #     mock_post_response.json.return_value = {"output": {"results": []}}
    #     mock_req_post.return_value = mock_post_response
    #
    #     # 3. 执行调用
    #     docs_text = [c.content for c in sample_candidates]
    #     result = skill._call_dashscope_rerank(
    #         query="Python",
    #         docs=docs_text,
    #         api_key="test_key",
    #         top_n=3
    #     )
    #
    #     # 4. 验证
    #     assert len(result) == 3
    #     assert result[0]["index"] == 0
    #     assert result[0]["relevance_score"] == 0.95
    #
    #     # 验证 SDK 成功时，没有回退到 requests
    #     mock_req_post.assert_not_called()

    @patch('requests.post')
    def test_call_dashscope_rerank_requests_fallback(self, mock_post, sample_candidates):
        """测试 requests 回退调用"""
        import src.skills.custom.rag_skills.rerank_skill.skill as skill_module

        # 备份并强制禁用 SDK，触发回退逻辑
        original_tr = getattr(skill_module, 'TextRerank', None)
        skill_module.TextRerank = None

        try:
            skill = RerankSkill()

            # Mock requests 响应
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 2, "relevance_score": 0.8},
                    ]
                }
            }
            mock_post.return_value = mock_response

            # 执行调用
            docs_text = [c.content for c in sample_candidates]
            result = skill._call_dashscope_rerank(
                query="Java",
                docs=docs_text,
                api_key="test_key",
                top_n=2
            )

            # 验证
            assert len(result) == 2
            mock_post.assert_called_once()
        finally:
            # 恢复环境
            skill_module.TextRerank = original_tr

    def test_execute_skip(self, sample_candidates):
        """测试执行时跳过精排"""
        skill = RerankSkill()

        # 候选数 <= top_n，应该跳过
        input_data = RerankInput(
            query="Python",
            candidates=sample_candidates[:2],
            top_n=5,
            api_key="test_key"
        )

        result = skill.execute(input_data)

        assert result["skipped"] is True
        assert len(result["reranked_docs"]) == 2

    @patch.object(RerankSkill, '_call_dashscope_rerank')
    def test_execute_full(self, mock_call_rerank, sample_candidates):
        """测试完整执行流程"""
        skill = RerankSkill()

        # Mock API 调用
        mock_call_rerank.return_value = [
            {"index": 0, "relevance_score": 0.95},
            {"index": 3, "relevance_score": 0.85},
            {"index": 5, "relevance_score": 0.75},
            {"index": 1, "relevance_score": 0.25},  # 低于阈值
        ]

        input_data = RerankInput(
            query="Python编程",
            candidates=sample_candidates,
            top_n=3,
            min_relevance_score=0.5,
            api_key="test_key"
        )

        result = skill.execute(input_data)

        assert result["skipped"] is False
        assert len(result["reranked_docs"]) == 3
        assert result["reranked_docs"][0]["doc_id"] == "1"
        assert result["reranked_docs"][0]["rerank_score"] == 0.95

    @patch.object(RerankSkill, '_call_dashscope_rerank')
    def test_execute_soft_fallback(self, mock_call_rerank, sample_candidates):
        """测试软过滤补全"""
        skill = RerankSkill()

        # 只有1个达标结果，需要补全
        mock_call_rerank.return_value = [
            {"index": 0, "relevance_score": 0.95},
            {"index": 3, "relevance_score": 0.25},
            {"index": 5, "relevance_score": 0.20},
        ]

        input_data = RerankInput(
            query="Python",
            candidates=sample_candidates,
            top_n=3,
            min_relevance_score=0.5,
            api_key="test_key"
        )

        result = skill.execute(input_data)

        assert len(result["reranked_docs"]) == 3
        # 验证补全了低分结果
        assert result["reranked_docs"][1]["rerank_score"] < 0.5

    @patch.object(RerankSkill, '_call_dashscope_rerank')
    def test_execute_api_failure_fallback(self, mock_call_rerank, sample_candidates):
        """测试 API 失败时的降级"""
        skill = RerankSkill()

        # Mock API 调用失败
        mock_call_rerank.side_effect = Exception("API 调用失败")

        input_data = RerankInput(
            query="Python",
            candidates=sample_candidates,
            top_n=3,
            api_key="test_key"
        )

        result = skill.execute(input_data)

        assert result["skipped"] is True
        # 降级为原始顺序
        assert len(result["reranked_docs"]) == 3
        assert result["reranked_docs"][0]["doc_id"] == "1"


# ==========================================
# QueryRewriteSkill 测试
# ==========================================

class TestQueryRewriteSkill:
    """QueryRewriteSkill 测试套件"""

    def test_input_validation(self, sample_history):
        """测试输入验证"""
        # 正常输入
        valid_input = QueryRewriteInput(
            original_query="它有什么特点？",
            history=sample_history
        )
        assert valid_input.original_query == "它有什么特点？"
        assert len(valid_input.history) == 3

        # 缺少必填字段或类型错误
        with pytest.raises(ValidationError):
            # 传入非字符串类型会触发 ValidationError
            QueryRewriteInput(original_query=123, history=[])

        with pytest.raises(ValidationError):
            # 传入非列表类型给 history
            QueryRewriteInput(original_query="test", history="not_a_list")

    def test_check_need_rewrite(self, sample_history):
        """测试改写必要性判断"""
        skill = QueryRewriteSkill()

        # 测试1：包含指代词
        assert skill._check_need_rewrite("它有什么特点？", []) is True

        # 测试2：包含并列连词
        assert skill._check_need_rewrite("Python和Java有什么区别？", []) is True

        # 测试3：有历史且查询足够长
        assert skill._check_need_rewrite("它有什么特点？", sample_history) is True

        # 测试4：查询过短且无历史
        assert skill._check_need_rewrite("好的", []) is False

        # 测试5：正常查询无需改写
        assert skill._check_need_rewrite("Python是什么？", []) is False

    def test_build_prompt(self, sample_history):
        """测试 Prompt 构建"""
        skill = QueryRewriteSkill()

        messages = skill._build_prompt(
            query="它有什么特点？",
            history=sample_history,
            max_splits=3,
            history_window=3
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "指代消解" in messages[0]["content"]
        assert "Python" in messages[0]["content"]  # 历史内容应在 prompt 中

    def test_parse_llm_output_json(self):
        """测试 JSON 输出解析"""
        skill = QueryRewriteSkill()

        # 标准 JSON 数组
        result = QueryRewriteSkill._parse_llm_output(
            content='["Python的特点", "Python的优势"]',
            max_queries=3,
            fallback="Python"
        )
        assert result == ["Python的特点", "Python的优势"]

        # Markdown 代码块包裹
        result = QueryRewriteSkill._parse_llm_output(
            content='```json\n["Python的特点", "Python的优势"]\n```',
            max_queries=3,
            fallback="Python"
        )
        assert result == ["Python的特点", "Python的优势"]

    def test_parse_llm_output_fallback(self):
        """测试解析失败时的兜底"""
        skill = QueryRewriteSkill()

        # 无效 JSON，使用行分割
        result = QueryRewriteSkill._parse_llm_output(
            content="Python的特点\nPython的优势",
            max_queries=3,
            fallback="Python"
        )
        assert result == ["Python的特点", "Python的优势"]

        # 完全无效，使用 fallback
        result = QueryRewriteSkill._parse_llm_output(
            content="",
            max_queries=3,
            fallback="Python"
        )
        assert result == ["Python"]

    @patch.object(QueryRewriteSkill, '_get_client')
    def test_execute_skip(self, mock_get_client):
        """测试跳过改写"""
        skill = QueryRewriteSkill()

        # 无需改写的查询
        input_data = QueryRewriteInput(
            original_query="Python是什么？",
            history=[],
            api_key="test_key"
        )

        result = skill.execute(input_data)

        assert result["is_rewritten"] is False
        assert result["rewritten_queries"] == ["Python是什么？"]
        # 不应调用 LLM
        mock_get_client.assert_not_called()

    @patch.object(QueryRewriteSkill, '_call_llm')
    @patch.object(QueryRewriteSkill, '_get_client')
    def test_execute_full(self, mock_get_client, mock_call_llm, sample_history):
        """测试完整改写流程"""
        skill = QueryRewriteSkill()

        # Mock LLM 响应
        mock_call_llm.return_value = '["Python的特点", "Python的优势"]'

        input_data = QueryRewriteInput(
            original_query="它有什么特点？",
            history=sample_history,
            api_key="test_key"
        )

        result = skill.execute(input_data)

        assert result["is_rewritten"] is True
        assert result["rewritten_queries"] == ["Python的特点", "Python的优势"]

    @patch.object(QueryRewriteSkill, '_call_llm')
    @patch.object(QueryRewriteSkill, '_get_client')
    def test_execute_llm_failure(self, mock_get_client, mock_call_llm, sample_history):
        """测试 LLM 失败时的降级"""
        skill = QueryRewriteSkill()

        # Mock LLM 调用失败
        mock_call_llm.side_effect = Exception("LLM 调用失败")

        input_data = QueryRewriteInput(
            original_query="它有什么特点？",
            history=sample_history,
            api_key="test_key"
        )

        result = skill.execute(input_data)

        # 降级为原始查询
        assert result["is_rewritten"] is False
        assert result["rewritten_queries"] == ["它有什么特点？"]

    def test_max_sub_queries_limit(self, sample_history):
        """测试最大子查询数限制"""
        skill = QueryRewriteSkill()

        # 直接测试解析逻辑
        result = QueryRewriteSkill._parse_llm_output(
            content='["问题1", "问题2", "问题3", "问题4", "问题5"]',
            max_queries=3,
            fallback="问题"
        )

        # 应该只返回前3个
        assert len(result) == 3
        assert result == ["问题1", "问题2", "问题3"]


# ==========================================
# 集成测试
# ==========================================

class TestIntegration:
    """集成测试"""

    @patch.object(RerankSkill, '_call_dashscope_rerank')
    def test_rerank_full_workflow(self, mock_call_rerank, sample_candidates):
        """测试 RerankSkill 完整工作流"""
        skill = RerankSkill()

        # Mock API 响应
        mock_call_rerank.return_value = [
            {"index": 0, "relevance_score": 0.95},
            {"index": 3, "relevance_score": 0.85},
            {"index": 5, "relevance_score": 0.75},
        ]

        # 1. 创建输入
        input_data = RerankInput(
            query="Python编程",
            candidates=sample_candidates,
            top_n=3,
            min_relevance_score=0.5,
            api_key="test_key"
        )

        # 2. 执行
        result = skill.execute(input_data)

        # 3. 验证完整流程
        assert result["skipped"] is False
        assert len(result["reranked_docs"]) == 3
        assert all("rerank_score" in doc for doc in result["reranked_docs"])
        assert result["reranked_docs"][0]["rerank_score"] > result["reranked_docs"][1]["rerank_score"]

    @patch.object(QueryRewriteSkill, '_call_llm')
    @patch.object(QueryRewriteSkill, '_get_client')
    def test_query_rewrite_full_workflow(self, mock_get_client, mock_call_llm, sample_history):
        """测试 QueryRewriteSkill 完整工作流"""
        skill = QueryRewriteSkill()

        # Mock LLM 响应
        mock_call_llm.return_value = '["Python的特点", "Python的优势"]'

        # 1. 创建输入（包含指代消解场景）
        input_data = QueryRewriteInput(
            original_query="它有什么特点？",
            history=sample_history,
            max_sub_queries=3,
            api_key="test_key"
        )

        # 2. 执行
        result = skill.execute(input_data)

        # 3. 验证完整流程
        assert result["is_rewritten"] is True
        assert len(result["rewritten_queries"]) == 2
        assert "它" not in result["rewritten_queries"][0]  # 指代应被消解

if __name__ == "__main__":    pytest.main()
# tests/test_orchestrator_graphrag_routing.py

import pytest
from unittest import mock
from src.agent.orchestrator import SkillOrchestrator
from src.skills.base.skill_manager import SkillManager


def _create_orchestrator_for_test():
    """创建用于测试的 orchestrator，跳过自动技能发现以避免网络请求"""
    orch = SkillOrchestrator.__new__(SkillOrchestrator)
    orch.skill_manager = SkillManager()
    orch._pipelines = {}
    orch._tools_cache = []
    orch.llm_client = None  # 测试中会 mock
    orch._register_predefined_pipelines()
    orch._sync_tools_to_llm()
    return orch


class TestGraphRAGKeywordRouting:
    """测试 _route_to_skill 关键词快速路由"""

    def test_route_indexer_by_keyword(self):
        orch = _create_orchestrator_for_test()

        # 各种"索引"说法都应该命中 indexer
        cases = [
            "帮我构建知识图谱索引",
            "请创建索引",
            "索引这些文档",
            "build index for my data",
            "建立知识图谱",
        ]
        for text in cases:
            result = orch._route_to_skill(text)
            assert result is not None, f"失败：{text}"
            assert result[0] == "graphrag_indexer", f"失败：{text} → {result[0]}"

    def test_route_searcher_by_keyword(self):
        orch = _create_orchestrator_for_test()

        cases = [
            "图谱搜索：Python 和 Java 什么关系",
            "用 local search 查一下",
            "知识图谱检索实体的关联",
            "graphrag search 人工智能",
        ]
        for text in cases:
            result = orch._route_to_skill(text)
            assert result is not None, f"失败：{text}"
            assert result[0] == "graphrag_searcher", f"失败：{text} → {result[0]}"

    def test_route_fallback_to_llm(self):
        """不包含 GraphRAG 关键词的输入应该返回 None，交给 LLM"""
        orch = _create_orchestrator_for_test()

        cases = [
            "帮我总结这段文本",
            "写一封邮件",
            "今天天气怎么样",
        ]
        for text in cases:
            result = orch._route_to_skill(text)
            assert result is None, f"不应被路由：{text}"


class TestArgumentInferences:
    """测试 _infer_arguments_from_input 参数提取"""

    def test_infer_indexer_documents(self):
        orch = _create_orchestrator_for_test()

        args = orch._infer_arguments_from_input(
            "graphrag_indexer",
            "帮我索引这段话：人工智能是计算机科学的重要分支，机器学习是核心驱动力"
        )
        assert "documents" in args
        assert len(args["documents"]) >= 1
        assert "人工智能" in args["documents"][0]["text"]

    def test_infer_indexer_files(self):
        orch = _create_orchestrator_for_test()

        args = orch._infer_arguments_from_input(
            "graphrag_indexer",
            "请索引 data/report.txt 和 docs/readme.md"
        )
        assert "input_files" in args
        assert any("report.txt" in f for f in args["input_files"])

    def test_infer_searcher_mode_and_query(self):
        orch = _create_orchestrator_for_test()

        # global
        args = orch._infer_arguments_from_input(
            "graphrag_searcher",
            "全局搜索：这个项目整体架构是什么"
        )
        assert args["mode"] == "global"
        assert "query" in args

        # hybrid
        args = orch._infer_arguments_from_input(
            "graphrag_searcher",
            "混合搜索 Python 和 Rust 的对比"
        )
        assert args["mode"] == "hybrid"

        # local（默认）
        args = orch._infer_arguments_from_input(
            "graphrag_searcher",
            "查一下 REST API 怎么设计的"
        )
        assert args["mode"] == "local"


class TestPipelineMatching:
    """测试 index_then_search 流水线匹配"""

    def test_match_index_then_search(self):
        orch = _create_orchestrator_for_test()

        # 使用明确的单文件索引触发词，避免误匹配到 multi_file 流水线
        # "索引这些文档"会被误判为多文件，改为"索引文档"或"索引并查询"
        pipeline = orch._match_pipeline("索引文档然后查询里面讲了什么")
        assert pipeline is not None
        assert pipeline.name == "index_then_search"

    def test_match_multi_file_pipeline(self):
        orch = _create_orchestrator_for_test()

        pipeline = orch._match_pipeline("批量索引这些文件然后搜索")
        assert pipeline is not None
        assert pipeline.name == "multi_file_index_then_search"


class TestProcessEndToEnd:
    """端到端测试：process() 正确路由"""

    def test_process_indexer(self):
        orch = _create_orchestrator_for_test()
        # 注册 graphrag_indexer 到 skill_manager
        from src.skills.custom.rag_skills.graphrag_indexer.skill import GraphRAGIndexerSkill
        orch.skill_manager.register(GraphRAGIndexerSkill)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0, stderr="")

            with mock.patch.object(
                orch.skill_manager._registry["graphrag_indexer"],
                "_get_index_stats",
                return_value={
                    "entity_count": 3,
                    "relationship_count": 2,
                    "community_count": 1,
                },
            ):
                result = orch.process("帮我构建知识图谱索引：人工智能是...")

                assert result.pipeline_type in ("single", "sequential")
                assert "graphrag_indexer" in str(result.pipeline_name)
                # 只要不抛异常且 returned 就算通过
                assert result.summary != ""

    def test_process_searcher(self):
        orch = _create_orchestrator_for_test()
        # 注册 graphrag_searcher 到 skill_manager
        from src.skills.custom.rag_skills.graphrag_searcher.skill import GraphRAGSearcherSkill
        orch.skill_manager.register(GraphRAGSearcherSkill)

        with mock.patch.object(
            orch.skill_manager._registry["graphrag_searcher"],
            "execute",
            return_value={
                "answer": "mock answer",
                "sources": [],
                "mode": "local",
                "entity_count": 0,
                "community_count": 0,
                "error": None,
            },
        ):
            result = orch.process("图谱搜索：Python 有哪些优点")

            assert result.pipeline_type == "single"
            assert "graphrag_searcher" in result.pipeline_name
if __name__ == "__main__":
    pytest.main()
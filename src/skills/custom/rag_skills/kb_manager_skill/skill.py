# src/skills/custom/rag_skills/kb_manager_skill/skill.py

"""
知识库管理技能

职责：
    1. 接收并验证用户输入（Discriminated Union 强制必填字段）
    2. 将请求路由到对应的 handler（策略模式，消除 if-elif）
    3. 统一异常处理 & 结构化日志
    4. 返回符合 output_schema 的标准响应
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated, Any, ClassVar, Dict, Union, get_args

from pydantic import BaseModel, Field

from src.skills.base.base_skill import BaseSkill
from src.infrastructure.kb_manager import (
    # 写操作
    create_knowledge_base,
    add_document_to_kb,
    update_kb_status,
    delete_knowledge_base,
    delete_document,
    # 读操作
    get_knowledge_base,
    list_knowledge_bases,
    list_documents,
    # 枚举 & 异常
    IndexingStatus,
    KnowledgeBaseError,
    KnowledgeBaseNotFoundError,
    DocumentNotFoundError,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# Action 枚举 — 所有合法操作的注册表
# ═══════════════════════════════════════════════════════

class KBAction(str, Enum):
    """知识库管理操作类型"""
    CREATE_KB  = "create_kb"
    LIST_KBS   = "list_kbs"
    GET_KB     = "get_kb"
    DELETE_KB  = "delete_kb"
    UPLOAD_DOC = "upload_doc"
    LIST_DOCS  = "list_docs"
    DELETE_DOC = "delete_doc"
    REINDEX    = "reindex"


# ═══════════════════════════════════════════════════════
# Discriminated Union 输入模型
# ═══════════════════════════════════════════════════════

class CreateKBInput(BaseModel):
    """创建知识库"""
    action: KBAction = KBAction.CREATE_KB
    name: str = Field(..., min_length=1, description="知识库名称")
    description: str = Field(default="", description="知识库描述")


class ListKBsInput(BaseModel):
    """列出所有知识库"""
    action: KBAction = KBAction.LIST_KBS


class GetKBInput(BaseModel):
    """查询单个知识库"""
    action: KBAction = KBAction.GET_KB
    kb_id: str = Field(..., min_length=1, description="知识库ID")


class DeleteKBInput(BaseModel):
    """删除知识库"""
    action: KBAction = KBAction.DELETE_KB
    kb_id: str = Field(..., min_length=1, description="知识库ID")


class UploadDocInput(BaseModel):
    """上传文档到知识库"""
    action: KBAction = KBAction.UPLOAD_DOC
    kb_id: str = Field(..., min_length=1, description="目标知识库ID")
    file_name: str = Field(..., min_length=1, description="文件名")
    file_path: str = Field(..., min_length=1, description="文件存储路径")


class ListDocsInput(BaseModel):
    """列出知识库下所有文档"""
    action: KBAction = KBAction.LIST_DOCS
    kb_id: str = Field(..., min_length=1, description="知识库ID")


class DeleteDocInput(BaseModel):
    """删除文档"""
    action: KBAction = KBAction.DELETE_DOC
    doc_id: str = Field(..., min_length=1, description="文档ID")


class ReindexInput(BaseModel):
    """触发知识库重新索引"""
    action: KBAction = KBAction.REINDEX
    kb_id: str = Field(..., min_length=1, description="知识库ID")


# 联合类型 — Pydantic 根据 action 字段自动分发到对应子模型
KBManagerInput = Annotated[
    Union[
        CreateKBInput,
        ListKBsInput,
        GetKBInput,
        DeleteKBInput,
        UploadDocInput,
        ListDocsInput,
        DeleteDocInput,
        ReindexInput,
    ],
    Field(discriminator="action"),
]


# ═══════════════════════════════════════════════════════
# 输出模型
# ═══════════════════════════════════════════════════════

class KBManagerOutput(BaseModel):
    """知识库管理统一输出模型"""
    success: bool
    action: str = Field(default="", description="回显操作类型")
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    error: str = ""


# ═══════════════════════════════════════════════════════
# 技能主体
# ═══════════════════════════════════════════════════════

class KnowledgeBaseManagerSkill(BaseSkill):
    """知识库管理技能 — 策略模式路由"""

    name = "kb_manager"
    description = "管理知识库的生命周期：创建、查询、删除、上传文档等"
    version = "1.1.0"
    author = "EnterpriseLearningAgent"
    triggers = [
        "知识库", "创建知识库", "上传文档", "删除知识库",
        "kb_manager", "重新索引",
    ]

    input_schema = KBManagerInput
    output_schema = KBManagerOutput

    # ═══════════════════════════════════════════════════
    # Handler 注册表（策略模式核心）
    # ═══════════════════════════════════════════════════

    HANDLER_MAP: ClassVar[Dict[KBAction, str]] = {
        KBAction.CREATE_KB:  "_handle_create_kb",
        KBAction.LIST_KBS:   "_handle_list_kbs",
        KBAction.GET_KB:     "_handle_get_kb",
        KBAction.DELETE_KB:  "_handle_delete_kb",
        KBAction.UPLOAD_DOC: "_handle_upload_doc",
        KBAction.LIST_DOCS:  "_handle_list_docs",
        KBAction.DELETE_DOC: "_handle_delete_doc",
        KBAction.REINDEX:    "_handle_reindex",
    }

    # ═══════════════════════════════════════════════════
    # 核心路由
    # ═══════════════════════════════════════════════════

    def execute(self, input_data: KBManagerInput) -> Dict[str, Any]:
        """
        根据 input_data.action 自动路由到对应 handler

        Args:
            input_data: 已通过 Pydantic discriminated union 验证的输入

        Returns:
            符合 output_schema 的字典
        """
        action: KBAction = input_data.action

        logger.info(
            "收到知识库管理请求 | action=%s | input_preview=%s",
            action.value,
            input_data.model_dump(exclude={"action"}),
        )

        handler_name = self.HANDLER_MAP.get(action)
        if handler_name is None:
            logger.warning("不支持的 action | action=%s", action.value)
            return KBManagerOutput(
                success=False,
                action=action.value,
                message="不支持的操作",
                error=f"未知 action: {action.value}",
            ).model_dump()

        handler = getattr(self, handler_name, None)
        if handler is None:
            logger.error(
                "Handler 未实现 | action=%s handler_name=%s",
                action.value, handler_name,
            )
            return KBManagerOutput(
                success=False,
                action=action.value,
                message="系统错误",
                error=f"Handler {handler_name} 未实现",
            ).model_dump()

        try:
            result = handler(input_data)
            logger.info("操作成功 | action=%s", action.value)
            return result

        except KnowledgeBaseNotFoundError as e:
            logger.warning("知识库不存在 | action=%s kb_id=%s", action.value, e.kb_id)
            return KBManagerOutput(
                success=False,
                action=action.value,
                message="操作失败：知识库不存在",
                error=str(e),
            ).model_dump()

        except DocumentNotFoundError as e:
            logger.warning("文档不存在 | action=%s doc_id=%s", action.value, e.doc_id)
            return KBManagerOutput(
                success=False,
                action=action.value,
                message="操作失败：文档不存在",
                error=str(e),
            ).model_dump()

        except KnowledgeBaseError as e:
            logger.error(
                "业务异常 | action=%s error=%s", action.value, str(e)
            )
            return KBManagerOutput(
                success=False,
                action=action.value,
                message="操作失败",
                error=str(e),
            ).model_dump()

        except Exception:
            logger.exception("系统异常 | action=%s", action.value)
            return KBManagerOutput(
                success=False,
                action=action.value,
                message="系统异常",
                error="内部错误，请查看日志",
            ).model_dump()

    # ═══════════════════════════════════════════════════
    # Handler 实现
    # ═══════════════════════════════════════════════════

    def _handle_create_kb(self, inp: CreateKBInput) -> Dict[str, Any]:
        """创建知识库"""
        kb = create_knowledge_base(inp.name, inp.description)
        return KBManagerOutput(
            success=True,
            action=KBAction.CREATE_KB.value,
            message="知识库创建成功",
            data=kb.model_dump(),
        ).model_dump()

    def _handle_list_kbs(self, inp: ListKBsInput) -> Dict[str, Any]:
        """列出所有知识库"""
        kbs = list_knowledge_bases()
        return KBManagerOutput(
            success=True,
            action=KBAction.LIST_KBS.value,
            message=f"共 {len(kbs)} 个知识库",
            data={"knowledge_bases": [kb.model_dump() for kb in kbs]},
        ).model_dump()

    def _handle_get_kb(self, inp: GetKBInput) -> Dict[str, Any]:
        """查询单个知识库"""
        kb = get_knowledge_base(inp.kb_id)
        return KBManagerOutput(
            success=True,
            action=KBAction.GET_KB.value,
            message="查询成功",
            data=kb.model_dump(),
        ).model_dump()

    def _handle_delete_kb(self, inp: DeleteKBInput) -> Dict[str, Any]:
        """删除知识库"""
        delete_knowledge_base(inp.kb_id)
        return KBManagerOutput(
            success=True,
            action=KBAction.DELETE_KB.value,
            message="知识库已删除",
            data={"kb_id": inp.kb_id},
        ).model_dump()

    def _handle_upload_doc(self, inp: UploadDocInput) -> Dict[str, Any]:
        """上传文档并触发索引"""
        # 1. 获取知识库信息（用于获取 vector_namespace）
        kb = get_knowledge_base(inp.kb_id)

        # 2. 添加文档元数据
        doc = add_document_to_kb(inp.kb_id, inp.file_name, inp.file_path)

        # 3. ✅ 新增：触发索引流水线
        from src.infrastructure.indexing_pipeline import run_indexing_pipeline
        index_result = run_indexing_pipeline(
            kb_id=inp.kb_id,
            doc_id=doc.doc_id,
            file_path=inp.file_path,
            vector_namespace=kb.vector_namespace
        )

        if not index_result["success"]:
            return KBManagerOutput(
                success=False,
                action=KBAction.UPLOAD_DOC.value,
                message="文档上传成功，但索引失败",
                data=doc.model_dump(),
                error=index_result.get("error", "未知索引错误")
            ).model_dump()

        # 4. 返回包含索引统计的响应
        return KBManagerOutput(
            success=True,
            action=KBAction.UPLOAD_DOC.value,
            message="文档上传并索引成功",
            data={
                **doc.model_dump(),
                "indexing": {
                    "chunk_count": index_result["chunk_count"],
                    "elapsed_ms": index_result["total_elapsed_ms"]
                }
            }
        ).model_dump()

    def _handle_list_docs(self, inp: ListDocsInput) -> Dict[str, Any]:
        """列出知识库下所有文档"""
        docs = list_documents(inp.kb_id)
        return KBManagerOutput(
            success=True,
            action=KBAction.LIST_DOCS.value,
            message=f"共 {len(docs)} 个文档",
            data={"documents": [d.model_dump() for d in docs]},
        ).model_dump()

    def _handle_delete_doc(self, inp: DeleteDocInput) -> Dict[str, Any]:
        """删除文档"""
        delete_document(inp.doc_id)
        return KBManagerOutput(
            success=True,
            action=KBAction.DELETE_DOC.value,
            message="文档已删除",
            data={"doc_id": inp.doc_id},
        ).model_dump()

    def _handle_reindex(self, inp: ReindexInput) -> Dict[str, Any]:
        """触发重新索引"""
        kb = get_knowledge_base(inp.kb_id)
        docs = list_documents(inp.kb_id)

        if not docs:
            return KBManagerOutput(
                success=False,
                action=KBAction.REINDEX.value,
                message="知识库下没有文档",
                error="该知识库下不存在可索引的文档"
            ).model_dump()

        # 更新知识库状态为处理中
        update_kb_status(inp.kb_id, IndexingStatus.PROCESSING)

        success_count = 0
        failed_count = 0

        for doc in docs:
            try:
                from src.infrastructure.indexing_pipeline import run_indexing_pipeline
                result = run_indexing_pipeline(
                    kb_id=inp.kb_id,
                    doc_id=doc.doc_id,
                    file_path=doc.file_path,
                    vector_namespace=kb.vector_namespace
                )

                if result["success"]:
                    success_count += 1
                    logger.info("文档重新索引成功 | doc_id=%s", doc.doc_id)
                else:
                    failed_count += 1
                    logger.warning("文档重新索引失败 | doc_id=%s error=%s", doc.doc_id, result.get("error"))
            except Exception as exc:
                failed_count += 1
                logger.exception("文档重新索引异常 | doc_id=%s", doc.doc_id)

        # 更新最终状态
        final_status = IndexingStatus.DONE if failed_count == 0 else IndexingStatus.ERROR
        update_kb_status(inp.kb_id, final_status)

        return KBManagerOutput(
            success=True,
            action=KBAction.REINDEX.value,
            message=f"重新索引完成：{success_count} 成功，{failed_count} 失败",
            data={
                "kb_id": inp.kb_id,
                "total_docs": len(docs),
                "success_count": success_count,
                "failed_count": failed_count,
                "final_status": final_status.value
            }
        ).model_dump()

    # ═══════════════════════════════════════════════════
    # 自省工具
    # ═══════════════════════════════════════════════════

    @classmethod
    def get_registered_actions(cls) -> list[str]:
        """返回所有已注册的 action 值（用于文档/调试）"""
        return [a.value for a in cls.HANDLER_MAP.keys()]
"""知识库相关端点（优化版）"""
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import (
    APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
)
from src.api.auth import verify_api_key
from src.api.deps import get_kb_manager, get_owned_kb  # 自定义依赖
from src.api.schemas.knowledge_base import (
    CreateKBRequest,
    UpdateKBRequest,
    UpdateDocumentRequest,
    ReindexRequest,
    KBDetailResponse,
    KBListResponse,
    DocumentResponse,
    DocumentUploadMeta,
    ReindexTaskResponse,
    KBStatus,
    PaginatedResponse,
)
from src.infrastructure.kb_manager import KBManager
from src.infrastructure.kb_manager import KnowledgeBase, Document  # 假设的 ORM 模型

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])

# 嵌套文档路由（前缀会自动合并）
documents_router = APIRouter(prefix="/{kb_id}/documents", tags=["documents"])


# ==================== 知识库 CRUD ====================

@router.post("", response_model=KBDetailResponse, status_code=201)
async def create_kb(
    request: CreateKBRequest,
    user_id: str = Depends(verify_api_key),
    kb_manager: KBManager = Depends(get_kb_manager),
):
    """创建知识库"""
    kb = kb_manager.create_kb(
        name=request.name,
        description=request.description,
        embedding_model=request.embedding_model,
        created_by=user_id,
    )
    logger.info(f"知识库创建成功: {kb.kb_id}")
    return KBDetailResponse.model_validate(kb)


@router.get("", response_model=PaginatedResponse[KBListResponse])
async def list_kbs(
    user_id: str = Depends(verify_api_key),
    kb_manager: KBManager = Depends(get_kb_manager),
    status: Optional[KBStatus] = Query(None, description="按状态筛选"),
    search: Optional[str] = Query(None, description="按名称模糊搜索"),
    sort_by: str = Query("created_at", regex="^(created_at|updated_at|name)$"),
    order: str = Query("desc", regex="^(asc|desc)$"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    列出知识库（分页、过滤、搜索、排序）。
    默认排除已删除的知识库。
    """
    # 默认排除已删除状态
    status_filter = status.value if status else None
    exclude_deleted = True if not status_filter else (status_filter != KBStatus.DELETED)

    kbs, total = kb_manager.list_kbs(
        created_by=user_id,
        status=status_filter,
        search=search,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset,
        exclude_deleted=exclude_deleted,
    )
    return PaginatedResponse(
        items=[KBListResponse.model_validate(kb) for kb in kbs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{kb_id}", response_model=KBDetailResponse)
async def get_kb(
    kb: KnowledgeBase = Depends(get_owned_kb),  # 直接注入已校验的对象
):
    """获取知识库详情"""
    return KBDetailResponse.model_validate(kb)


@router.put("/{kb_id}", response_model=KBDetailResponse)
async def update_kb(
    request: UpdateKBRequest,
    kb: KnowledgeBase = Depends(get_owned_kb),
    kb_manager: KBManager = Depends(get_kb_manager),
):
    """更新知识库信息"""
    updates = request.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated_kb = kb_manager.update_kb(kb.kb_id, **updates)
    if not updated_kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found after update")
    return KBDetailResponse.model_validate(updated_kb)


@router.delete("/{kb_id}", status_code=204)
async def delete_kb(
    kb: KnowledgeBase = Depends(get_owned_kb),
    kb_manager: KBManager = Depends(get_kb_manager),
):
    """
    软删除知识库（标记状态为 DELETED，记录删除时间）。
    知识库及相关文档可定期清理或恢复。
    """
    success = kb_manager.soft_delete_kb(
        kb.kb_id,
        deleted_at=datetime.now(timezone.utc)
    )
    if not success:
        raise HTTPException(status_code=404, detail="Knowledge base not found or already deleted")
    logger.info(f"知识库已软删除: {kb.kb_id}")


# ==================== 文档操作（独立路由） ====================

# @documents_router.post("", status_code=201, response_model=DocumentResponse)
# async def upload_document(
#     kb_id: str,
#     file: UploadFile = File(..., description="上传文件"),
#     metadata: Optional[str] = Form(None, description="元数据（JSON字符串）"),
#     user_id: str = Depends(verify_api_key),
#     kb_manager: KBManager = Depends(get_kb_manager),
#     kb: KnowledgeBase = Depends(get_owned_kb),  # 保证知识库存在且属于当前用户
# ):
#     """
#     通过文件上传添加文档，支持流式处理。
#     也可选择传递 metadata 的 JSON 字符串。
#     """
#     # 解析元数据
#     doc_meta = None
#     if metadata:
#         try:
#             import json
#             meta_dict = json.loads(metadata)
#             doc_meta = DocumentUploadMeta.model_validate(meta_dict)
#         except Exception:
#             raise HTTPException(status_code=400, detail="Invalid metadata format, must be JSON")
#
#     # 流式读取文件内容（限制最大大小，例如 50MB）
#     max_size = 50 * 1024 * 1024
#     content_bytes = await file.read()
#     if len(content_bytes) > max_size:
#         raise HTTPException(status_code=413, detail="File too large (max 50MB)")
#
#     content = content_bytes.decode("utf-8", errors="ignore")  # 根据文件类型可调整编码
#     doc = kb_manager.add_document(
#         kb_id=kb_id,
#         file_name=file.filename or "uploaded_file",
#         content=content,
#         metadata=doc_meta.model_dump() if doc_meta else None,
#     )
#     logger.info(f"文档上传成功: {doc.doc_id}")
#     return DocumentResponse.model_validate(doc)


@documents_router.get("", response_model=PaginatedResponse[DocumentResponse])
async def list_documents(
    kb_id: str,
    user_id: str = Depends(verify_api_key),
    kb_manager: KBManager = Depends(get_kb_manager),
    kb: KnowledgeBase = Depends(get_owned_kb),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """列出知识库下的文档（分页）"""
    docs, total = kb_manager.list_documents(kb_id, limit=limit, offset=offset)
    return PaginatedResponse(
        items=[DocumentResponse.model_validate(d) for d in docs],
        total=total,
        limit=limit,
        offset=offset,
    )


@documents_router.post("", status_code=201, response_model=DocumentResponse)
async def upload_document(
        kb_id: str,
        file: UploadFile = File(..., description="上传文件"),
        metadata: Optional[str] = Form(None, description="元数据（JSON字符串）"),
        user_id: str = Depends(verify_api_key),
        kb_manager: KBManager = Depends(get_kb_manager),
        kb: KnowledgeBase = Depends(get_owned_kb),
):
    """
    上传文档并自动触发索引管道。
    业务逻辑统一走 kb_manager Skill。
    """
    import tempfile
    import os

    from src.api.deps import get_skill_manager

    # 解析元数据
    doc_meta = None
    if metadata:
        try:
            import json
            meta_dict = json.loads(metadata)
            doc_meta = DocumentUploadMeta.model_validate(meta_dict)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid metadata format, must be JSON")

    # 流式读取文件内容（限制最大 50MB）
    max_size = 50 * 1024 * 1024
    content_bytes = await file.read()
    if len(content_bytes) > max_size:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    # 保存到临时路径供索引管道使用
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=f"_{file.filename}") as tmp_file:
        tmp_file.write(content_bytes)
        tmp_file_path = tmp_file.name

    try:
        # ✅ 统一走 SkillManager → kb_manager Skill（不再直接 import 3 个原子 Skill）
        skill_manager = get_skill_manager()
        result = skill_manager.call(
            "kb_manager",
            action="upload_doc",
            kb_id=kb_id,
            file_name=file.filename or "uploaded_file",
            file_path=tmp_file_path,
        )

        if not result.get("success"):
            logger.warning(f"kb_manager 上传失败: {result.get('error')}")
            # 文档元数据可能已创建，尝试返回已有信息
            doc = kb_manager.get_document(doc_id=result.get("data", {}).get("doc_id", ""))
            if doc:
                return DocumentResponse.model_validate(doc)
            raise HTTPException(status_code=500, detail=result.get("error", "文档上传失败"))

        # 从 Skill 返回的 data 中提取文档信息
        doc_data = result.get("data", {})
        doc = kb_manager.get_document(doc_id=doc_data.get("doc_id", ""))
        if doc:
            return DocumentResponse.model_validate(doc)

        raise HTTPException(status_code=500, detail="文档上传后无法获取信息")

    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_file_path)
        except Exception:
            pass


@documents_router.delete("/{doc_id}", status_code=204)
async def delete_document(
    kb_id: str,
    doc_id: str,
    user_id: str = Depends(verify_api_key),
    kb_manager: KBManager = Depends(get_kb_manager),
    kb: KnowledgeBase = Depends(get_owned_kb),
):
    """删除文档"""
    doc = kb_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.kb_id != kb_id:
        raise HTTPException(status_code=404, detail="Document not found in this knowledge base")

    success = kb_manager.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or already deleted")
    logger.info(f"文档已删除: {doc_id}")


# ==================== 重新索引（异步任务） ====================

@router.post("/{kb_id}/reindex", response_model=ReindexTaskResponse)
async def reindex_kb(
    kb_id: str,
    request: ReindexRequest,
    user_id: str = Depends(verify_api_key),
    kb_manager: KBManager = Depends(get_kb_manager),
    kb: KnowledgeBase = Depends(get_owned_kb),
):
    """
    提交重新索引任务，立即返回 task_id。
    实际任务由后台队列（如 arq）异步执行。
    """
    # 生成任务ID，并将任务交给队列
    task_id = str(uuid.uuid4())
    kb_manager.submit_reindex_task(kb_id, force=request.force, task_id=task_id)
    logger.info(f"重新索引任务已提交: kb={kb_id}, task={task_id}")
    return ReindexTaskResponse(task_id=task_id)


# 将文档子路由挂载到主路由器
router.include_router(documents_router)
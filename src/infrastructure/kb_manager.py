"""
知识库管理基础设施 — 负责元数据存储 (SQLite)

提供知识库和文档的元数据持久化，以及数据库引擎、
会话管理的统一封装。

表结构:
    knowledge_bases  — 知识库元信息
    documents        — 文档元信息（外键关联 knowledge_bases）

使用示例:
    from src.infrastructure.kb_manager import get_session

    with get_session() as session:
        kb = KnowledgeBase(name="技术文档", vector_namespace="tech_docs")
        session.add(kb)
        session.commit()
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings
from sqlalchemy import (
    Column,
    Engine,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TIMESTAMP,
    create_engine,
    event,
    func,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    relationship,
    sessionmaker,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# 配置（pydantic-settings，支持环境变量覆盖）
# ═══════════════════════════════════════════════════════

class KBSettings(BaseSettings):
    """知识库模块配置 — 可通过环境变量 KB_DB_PATH 覆盖"""

    db_path: str = str(Path(__file__).resolve().parent.parent.parent / "data" / "metrics.db")

    model_config = {"env_prefix": "KB_", "env_file": ".env", "extra": "ignore"}


_settings = KBSettings()

# ✅ 确保数据库目录存在
Path(_settings.db_path).parent.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════
# 状态枚举
# ═══════════════════════════════════════════════════════

class IndexingStatus(str, Enum):
    """知识库索引状态"""
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    ERROR      = "error"
    DELETED    = "deleted"  # ✅ 新增，用于软删除


class DocumentStatus(str, Enum):
    """文档处理状态"""
    UPLOADED  = "uploaded"
    INDEXED   = "indexed"
    FAILED    = "failed"
    PROCESSING = "processing"  # <--- 新增这一行

# ═══════════════════════════════════════════════════════
# ORM 基类
# ═══════════════════════════════════════════════════════

class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类"""
    pass


# ═══════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════

class KnowledgeBase(Base):
    """
    知识库元数据表

    字段说明:
        kb_id            — 主键，UUID 自动生成
        name             — 知识库名称（必填）
        description      — 描述文本
        vector_namespace — 对应 ChromaDB 的 Collection 名称（唯一）
        indexing_status  — 索引状态: pending | processing | done | error
        created_at       — 创建时间（UTC，数据库自动填充）
        updated_at       — 更新时间（UTC，应用层自动更新）
    """
    __tablename__ = "knowledge_bases"

    kb_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    vector_namespace = Column(
        String, unique=True, nullable=False,
        comment="对应 ChromaDB 的 Collection 名称",
    )
    indexing_status = Column(
        SAEnum(IndexingStatus, name="indexing_status_enum", create_constraint=True),
        default=IndexingStatus.PENDING,
        index=True,
    )
    created_by = Column(String, nullable=False, default="unknown")  # ✅ 新增
    embedding_model = Column(String, nullable=False, default="text-embedding-v3")  # ✅ 新增
    deleted_at = Column(TIMESTAMP, nullable=True)  # ✅ 新增
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(
        TIMESTAMP,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    documents = relationship(
        "Document", back_populates="kb",
        cascade="all, delete-orphan",
    )

    # ✅ 加这个属性，让 Pydantic schema 的 status 字段能匹配到
    @property
    def status(self):
        return self.indexing_status

    def __repr__(self) -> str:
        return f"<KnowledgeBase(kb_id={self.kb_id!r}, name={self.name!r})>"


class Document(Base):
    """
    文档元数据表

    字段说明:
        doc_id      — 主键，UUID 自动生成
        kb_id       — 所属知识库 ID（外键）
        file_name   — 原始文件名（必填）
        file_path   — 文件存储路径（必填）
        chunk_count — 分块数量，默认 0
        status      — 处理状态: uploaded | indexed | failed
    """
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    kb_id = Column(String, ForeignKey("knowledge_bases.kb_id"), nullable=False, index=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    chunk_count = Column(Integer, default=0)
    status = Column(
        SAEnum(DocumentStatus, name="document_status_enum", create_constraint=True),
        default=DocumentStatus.UPLOADED,
        index=True,
    )
    content = Column(Text, default="")  # ✅ 新增
    meta_json = Column(Text, nullable=True, default=None)

    @property
    def doc_metadata(self) -> dict | None:
        if self.meta_json:
            import json
            try:
                return json.loads(self.meta_json)
            except Exception:
                return None
        return None
    kb = relationship("KnowledgeBase", back_populates="documents")



    def __repr__(self) -> str:
        return f"<Document(doc_id={self.doc_id!r}, file_name={self.file_name!r})>"


# ═══════════════════════════════════════════════════════
# 引擎 & 会话（单例模式）
# ═══════════════════════════════════════════════════════

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def _ensure_columns(engine, table_name, required_columns):
    """自动添加表中缺失的列（仅 SQLite）"""
    import sqlalchemy as sa
    insp = sa.inspect(engine)
    if not insp.has_table(table_name):
        return False
    existing_cols = {col['name'] for col in insp.get_columns(table_name)}
    added = False
    with engine.connect() as conn:
        for col_name, col_type in required_columns.items():
            if col_name not in existing_cols:
                col_type_sql = str(col_type.compile(dialect=sa.dialects.sqlite.dialect()))
                # 为简单，这里只处理字符串和文本类型，实际可根据类型扩展
                if isinstance(col_type, sa.TIMESTAMP):
                    sql = f'ALTER TABLE {table_name} ADD COLUMN {col_name} TIMESTAMP'
                else:
                    sql = f'ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type_sql}'
                conn.execute(sa.text(sql))
                added = True
        conn.commit()
    return added


def _build_engine() -> Engine:
    """创建 SQLite 引擎并配置 PRAGMA（仅调用一次）"""
    engine = create_engine(
        f"sqlite:///{_settings.db_path}",
        connect_args={"check_same_thread": False},
        echo=False,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _):
        """每个连接建立时启用 WAL 模式 + 外键约束"""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine




def get_kb_engine() -> Engine:
    """
    获取知识库专用数据库引擎（单例）

    Returns:
        sqlalchemy.engine.Engine
    """
    global _engine
    if _engine is None:
        _engine = _build_engine()
        # 自动建表（create_all 只负责新建表，不改已有的列）
        Base.metadata.create_all(_engine)
        # 自动添加缺失列
        from sqlalchemy import String, Text, TIMESTAMP
        _ensure_columns(_engine, "knowledge_bases", {
            "created_by": String(),
            "embedding_model": String(),
            "deleted_at": TIMESTAMP(),
        })
        _ensure_columns(_engine, "documents", {
            "content": Text(),
            "meta_json": Text(),
        })
    return _engine


def get_session() -> Session:
    """
    获取数据库会话（上下文管理器用法）

    使用方式:
        with get_session() as session:
            kb = session.query(KnowledgeBase).first()

    Returns:
        sqlalchemy.orm.Session
    """
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_kb_engine(),
            expire_on_commit=False,  # ← 关键：commit 后不使属性过期
        )
    return _SessionFactory()

# ═══════════════════════════════════════════════════════
# 自定义业务异常
# ═══════════════════════════════════════════════════════

class KnowledgeBaseError(Exception):
    """知识库模块通用业务异常基类"""
    pass


class KnowledgeBaseNotFoundError(KnowledgeBaseError):
    """指定知识库不存在"""

    def __init__(self, kb_id: str) -> None:
        super().__init__(f"知识库不存在: kb_id={kb_id!r}")
        self.kb_id = kb_id


class DocumentNotFoundError(KnowledgeBaseError):
    """指定文档不存在"""

    def __init__(self, doc_id: str) -> None:
        super().__init__(f"文档不存在: doc_id={doc_id!r}")
        self.doc_id = doc_id


class DuplicateNamespaceError(KnowledgeBaseError):
    """向量命名空间（Collection Name）重复"""

    def __init__(self, vector_namespace: str) -> None:
        super().__init__(f"向量命名空间已被占用: vector_namespace={vector_namespace!r}")
        self.vector_namespace = vector_namespace


class KnowledgeBaseIntegrityError(KnowledgeBaseError):
    """数据库完整性约束冲突（通用）"""
    pass


# ═══════════════════════════════════════════════════════
# Pydantic 响应模型
# ═══════════════════════════════════════════════════════

class KnowledgeBaseResponse(BaseModel):
    """知识库查询/创建响应"""
    model_config = ConfigDict(from_attributes=True)

    kb_id: str
    name: str
    description: str
    vector_namespace: str
    indexing_status: IndexingStatus
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @staticmethod
    def from_orm_obj(kb: KnowledgeBase) -> "KnowledgeBaseResponse":
        """从 ORM 对象安全构建响应（必须在 session 内调用）"""
        return KnowledgeBaseResponse(
            kb_id=kb.kb_id,
            name=kb.name,
            description=kb.description or "",
            vector_namespace=kb.vector_namespace,
            indexing_status=kb.indexing_status,
            created_at=kb.created_at.isoformat() if kb.created_at else None,
            updated_at=kb.updated_at.isoformat() if kb.updated_at else None,
        )


class DocumentResponse(BaseModel):
    """文档查询/创建响应"""
    model_config = ConfigDict(from_attributes=True)

    doc_id: str
    kb_id: str
    file_name: str
    file_path: str
    chunk_count: int
    status: DocumentStatus

    @staticmethod
    def from_orm_obj(doc: Document) -> "DocumentResponse":
        """从 ORM 对象安全构建响应（必须在 session 内调用）"""
        return DocumentResponse(
            doc_id=doc.doc_id,
            kb_id=doc.kb_id,
            file_name=doc.file_name,
            file_path=doc.file_path,
            chunk_count=doc.chunk_count or 0,
            status=doc.status,
            metadata=doc.doc_metadata,  # ← 加这行
        )


# ═══════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════

def generate_kb_id() -> str:
    """生成唯一知识库 ID"""
    return str(uuid.uuid4())


def generate_vector_namespace(kb_id: str) -> str:
    """根据 kb_id 生成隔离的向量命名空间"""
    return f"kb_{kb_id}"


def _check_kb_exists(session: Session, kb_id: str) -> KnowledgeBase:
    """查询并验证知识库存在性"""
    kb = session.scalar(select(KnowledgeBase).where(KnowledgeBase.kb_id == kb_id))
    if kb is None:
        raise KnowledgeBaseNotFoundError(kb_id)
    return kb


@contextmanager
def _wrap_db_error(func_name: str, kb_id: str = "", doc_id: str = ""):
    """将 SQLAlchemy 异常转换为业务异常的上下文管理器"""
    try:
        yield
    except IntegrityError as exc:
        logger.error("数据库完整性约束冲突 | func=%s kb_id=%s doc_id=%s", func_name, kb_id, doc_id)
        orig_msg = str(exc.orig).lower() if exc.orig else ""
        if "vector_namespace" in orig_msg or "unique" in orig_msg:
            raise DuplicateNamespaceError(f"命名空间冲突: {func_name}") from exc
        raise KnowledgeBaseIntegrityError(str(exc.orig or exc)) from exc
    except OperationalError as exc:
        logger.error("数据库操作异常 | func=%s kb_id=%s doc_id=%s", func_name, kb_id, doc_id)
        raise KnowledgeBaseError(f"数据库操作失败: {exc}") from exc
    except SQLAlchemyError as exc:
        logger.error("未预期的数据库异常 | func=%s kb_id=%s doc_id=%s", func_name, kb_id, doc_id)
        raise KnowledgeBaseError(f"数据库异常: {exc}") from exc


# ═══════════════════════════════════════════════════════
# 读操作
# ═══════════════════════════════════════════════════════

def get_knowledge_base(kb_id: str) -> KnowledgeBaseResponse:
    """
    按 ID 查询知识库

    Raises:
        KnowledgeBaseNotFoundError: 知识库不存在
    """
    logger.info("查询知识库 | kb_id=%s", kb_id)
    with get_session() as session:
        kb = _check_kb_exists(session, kb_id)
        return KnowledgeBaseResponse.from_orm_obj(kb)


def list_knowledge_bases(
    status: Optional[IndexingStatus] = None,
) -> List[KnowledgeBaseResponse]:
    """
    列出所有知识库，可按索引状态筛选
    """
    logger.info("列出知识库 | filter_status=%s", status)
    with get_session() as session:
        stmt = select(KnowledgeBase)
        if status is not None:
            stmt = stmt.where(KnowledgeBase.indexing_status == status)
        stmt = stmt.order_by(KnowledgeBase.created_at.desc())
        results = session.scalars(stmt).all()
        return [KnowledgeBaseResponse.from_orm_obj(kb) for kb in results]


def get_document(doc_id: str) -> DocumentResponse:
    """
    按 ID 查询文档

    Raises:
        DocumentNotFoundError: 文档不存在
    """
    logger.info("查询文档 | doc_id=%s", doc_id)
    with get_session() as session:
        doc = session.scalar(select(Document).where(Document.doc_id == doc_id))
        if doc is None:
            raise DocumentNotFoundError(doc_id)
        return DocumentResponse.from_orm_obj(doc)


def list_documents(kb_id: str) -> List[DocumentResponse]:
    """
    列出指定知识库下的所有文档

    Raises:
        KnowledgeBaseNotFoundError: 知识库不存在
    """
    logger.info("列出文档 | kb_id=%s", kb_id)
    with get_session() as session:
        _check_kb_exists(session, kb_id)
        stmt = select(Document).where(Document.kb_id == kb_id).order_by(Document.doc_id)
        docs = session.scalars(stmt).all()
        return [DocumentResponse.from_orm_obj(d) for d in docs]


# ═══════════════════════════════════════════════════════
# 写操作
# ══════════════════════════════════════════════════════

def create_knowledge_base(name: str, description: str = "") -> KnowledgeBaseResponse:
    """
    创建一个新的知识库
    """
    kb_id = generate_kb_id()
    vector_namespace = generate_vector_namespace(kb_id)

    logger.info("创建知识库 | kb_id=%s name=%s namespace=%s", kb_id, name, vector_namespace)

    with get_session() as session:
        kb = KnowledgeBase(
            kb_id=kb_id,
            name=name,
            description=description,
            vector_namespace=vector_namespace,
            indexing_status=IndexingStatus.PENDING,
        )
        session.add(kb)

        with _wrap_db_error("create_knowledge_base", kb_id=kb_id):
            session.commit()

        response = KnowledgeBaseResponse.from_orm_obj(kb)

        # ✅ 新增：同步创建 ChromaDB Collection
        try:
            from src.infrastructure.vector_store import vector_store
            vector_store.create_collection(vector_namespace, metadata={"kb_id": kb_id, "kb_name": name})
            logger.info("ChromaDB Collection 已创建 | namespace=%s", vector_namespace)
        except Exception as exc:
            logger.warning("ChromaDB Collection 创建失败（可后续重试）| namespace=%s error=%s", vector_namespace, exc)

        logger.info("知识库创建成功 | kb_id=%s name=%s", kb_id, name)
    return response


def add_document_to_kb(
    kb_id: str, file_name: str, file_path: str
) -> DocumentResponse:
    """
    向指定知识库添加一条文档元数据
    """
    logger.info("添加文档 | kb_id=%s file_name=%s", kb_id, file_name)

    with get_session() as session:
        kb = _check_kb_exists(session, kb_id)

        doc = Document(
            kb_id=kb_id,
            file_name=file_name,
            file_path=file_path,
            chunk_count=0,
            status=DocumentStatus.UPLOADED,
        )
        session.add(doc)

        kb.updated_at = datetime.now(timezone.utc)

        with _wrap_db_error("add_document_to_kb", kb_id=kb_id):
            session.commit()

        response = DocumentResponse.from_orm_obj(doc)

    logger.info("文档添加成功 | doc_id=%s kb_id=%s file_name=%s", doc.doc_id, kb_id, file_name)
    return response


def update_kb_status(kb_id: str, status: IndexingStatus) -> KnowledgeBaseResponse:
    """
    （原子操作）更新知识库的索引状态
    """
    logger.info("更新索引状态 | kb_id=%s new_status=%s", kb_id, status.value)

    with get_session() as session:
        stmt = (
            update(KnowledgeBase)
            .where(KnowledgeBase.kb_id == kb_id)
            .values(
                indexing_status=status,
                updated_at=datetime.now(timezone.utc),
            )
        )
        result = session.execute(stmt)

        if result.rowcount == 0:
            raise KnowledgeBaseNotFoundError(kb_id)

        with _wrap_db_error("update_kb_status", kb_id=kb_id):
            session.commit()

        kb = _check_kb_exists(session, kb_id)
        response = KnowledgeBaseResponse.from_orm_obj(kb)

    logger.info("索引状态更新成功 | kb_id=%s new_status=%s", kb_id, status.value)
    return response


def update_document_status(doc_id: str, status: DocumentStatus) -> DocumentResponse:
    """
    （原子操作）更新文档处理状态
    """
    logger.info("更新文档状态 | doc_id=%s new_status=%s", doc_id, status.value)

    with get_session() as session:
        stmt = update(Document).where(Document.doc_id == doc_id).values(status=status)
        result = session.execute(stmt)

        if result.rowcount == 0:
            raise DocumentNotFoundError(doc_id)

        with _wrap_db_error("update_document_status", doc_id=doc_id):
            session.commit()

        doc = session.scalar(select(Document).where(Document.doc_id == doc_id))
        response = DocumentResponse.from_orm_obj(doc)

    logger.info("文档状态更新成功 | doc_id=%s new_status=%s", doc_id, status.value)
    return response


def delete_knowledge_base(kb_id: str) -> bool:
    logger.info("删除知识库 | kb_id=%s", kb_id)

    with get_session() as session:
        kb = _check_kb_exists(session, kb_id)
        vector_namespace = kb.vector_namespace  # ✅ 修改：先获取命名空间

        session.delete(kb)

        with _wrap_db_error("delete_knowledge_base", kb_id=kb_id):
            session.commit()

    # ✅ 新增：删除对应的 ChromaDB 集合
    try:
        from src.infrastructure.vector_store import vector_store
        vector_store.delete_collection(vector_namespace)
        logger.info("ChromaDB Collection 已删除 | namespace=%s", vector_namespace)
    except Exception as exc:
        # 警告而非报错：即使向量库删除失败，SQLite 记录也已清除
        logger.warning("ChromaDB Collection 删除失败（可能需要手动清理） | namespace=%s error=%s", vector_namespace, exc)

    logger.info("知识库已删除 | kb_id=%s", kb_id)
    return True


def delete_document(doc_id: str) -> bool:
    """
    删除指定文档
    """
    logger.info("删除文档 | doc_id=%s", doc_id)

    with get_session() as session:
        doc = session.scalar(select(Document).where(Document.doc_id == doc_id))
        if doc is None:
            raise DocumentNotFoundError(doc_id)

        session.delete(doc)

        with _wrap_db_error("delete_document", doc_id=doc_id):
            session.commit()

    logger.info("文档已删除 | doc_id=%s", doc_id)
    return True


# ═══════════════════════════════════════════════════════
# KBManager 业务封装类（为 API 层提供统一入口）
# ═══════════════════════════════════════════════════════

class KBManager:
    """知识库业务管理器，封装所有读写操作，并对外提供与 routes 一致的接口"""
    _instance = None

    @classmethod
    def default(cls) -> "KBManager":
        """返回 KBManager 单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ---------- 知识库 ----------
    def get_kb(self, kb_id: str) -> Optional[KnowledgeBase]:
        """返回 ORM 对象，未找到返回 None（而非抛异常，便于上层判断）"""
        try:
            with get_session() as session:
                return session.get(KnowledgeBase, kb_id)
        except Exception:
            return None

    def create_kb(self, name: str, description: str = "",
                  embedding_model: str = "text-embedding-v3",
                  created_by: str = "unknown") -> KnowledgeBase:
        """
        创建知识库，并存入 API 所需的额外字段。
        底层 create_knowledge_base 已生成 kb_id 和 namespace，这里通过直接构建 ORM 对象来实现。
        """
        # 手动构建 ORM 对象，因为底层函数返回的是响应模型，缺少自定义字段
        kb_id = generate_kb_id()
        vector_namespace = generate_vector_namespace(kb_id)

        with get_session() as session:
            kb = KnowledgeBase(
                kb_id=kb_id,
                name=name,
                description=description,
                vector_namespace=vector_namespace,
                embedding_model=embedding_model,
                created_by=created_by,
                indexing_status=IndexingStatus.PENDING,
            )
            session.add(kb)
            with _wrap_db_error("create_kb", kb_id=kb_id):
                session.commit()
            # ✅ 修复 DetachedInstanceError：在 session 关闭前脱离对象
            session.expunge(kb)
            # 同步创建 ChromaDB collection
            try:
                from src.infrastructure.vector_store import vector_store
                vector_store.create_collection(vector_namespace, metadata={"kb_id": kb_id, "kb_name": name})
                logger.info("ChromaDB Collection 已创建 | namespace=%s", vector_namespace)
            except Exception as exc:
                logger.warning("ChromaDB Collection 创建失败 | %s", exc)
            logger.info("知识库创建成功 | kb_id=%s", kb_id)
            return kb

    def list_kbs(self, created_by: str = None, status: str = None,
                 search: str = None, sort_by: str = "created_at",
                 order: str = "desc", limit: int = 50, offset: int = 0,
                 exclude_deleted: bool = True) -> tuple[list, int]:
        """
        分页、过滤、搜索、排序列表。
        返回 (ORM 对象列表, 总数)
        """
        with get_session() as session:
            stmt = select(KnowledgeBase)
            # 所有权过滤
            if created_by:
                stmt = stmt.where(KnowledgeBase.created_by == created_by)
            # 状态过滤
            if status:
                stmt = stmt.where(KnowledgeBase.indexing_status == status)
            # 排除已删除
            if exclude_deleted:
                stmt = stmt.where(KnowledgeBase.indexing_status != IndexingStatus.DELETED)
            # 模糊搜索
            if search:
                stmt = stmt.where(KnowledgeBase.name.ilike(f"%{search}%"))
            # 排序
            sort_col = getattr(KnowledgeBase, sort_by, KnowledgeBase.created_at)
            if order == "asc":
                stmt = stmt.order_by(sort_col.asc())
            else:
                stmt = stmt.order_by(sort_col.desc())
            # 总数
            count_stmt = stmt.with_only_columns(func.count()).order_by(None)
            total = session.scalar(count_stmt)
            # 分页
            stmt = stmt.limit(limit).offset(offset)
            items = session.scalars(stmt).all()
            return items, total

    def update_kb(self, kb_id: str, **kwargs) -> Optional[KnowledgeBase]:
        """部分更新知识库字段，返回更新后的 ORM 对象"""
        allowed_fields = {"name", "description", "embedding_model"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return self.get_kb(kb_id)

        updates["updated_at"] = datetime.now(timezone.utc)
        with get_session() as session:
            stmt = update(KnowledgeBase).where(KnowledgeBase.kb_id == kb_id).values(**updates)
            result = session.execute(stmt)
            with _wrap_db_error("update_kb", kb_id=kb_id):
                session.commit()
            if result.rowcount == 0:
                return None
            return session.get(KnowledgeBase, kb_id)

    def soft_delete_kb(self, kb_id: str, deleted_at: datetime = None) -> bool:
        """将知识库状态标记为 DELETED，设置删除时间"""
        with get_session() as session:
            stmt = update(KnowledgeBase).where(KnowledgeBase.kb_id == kb_id).values(
                indexing_status=IndexingStatus.DELETED,
                deleted_at=deleted_at or datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            result = session.execute(stmt)
            with _wrap_db_error("soft_delete_kb", kb_id=kb_id):
                session.commit()
            return result.rowcount > 0

    # ---------- 文档 ----------
    def add_document(self, kb_id: str, file_name: str, content: str = "",
                     metadata: dict = None, file_path: str = "") -> Document:
        """添加文档，支持存储内容和元数据"""
        with get_session() as session:
            _check_kb_exists(session, kb_id)

            # 元数据序列化为 JSON 字符串
            meta_str = ""
            if metadata:
                import json
                meta_str = json.dumps(metadata, ensure_ascii=False)

            doc = Document(
                kb_id=kb_id,
                file_name=file_name,
                file_path=file_path or file_name,  # 若无独立路径，用文件名占位
                content=content or "",
                meta_json=meta_str,
                status=DocumentStatus.UPLOADED,
            )
            session.add(doc)
            kb = session.get(KnowledgeBase, kb_id)
            kb.updated_at = datetime.now(timezone.utc)

            with _wrap_db_error("add_document", kb_id=kb_id):
                session.commit()
            logger.info("文档添加成功 | doc_id=%s kb_id=%s", doc.doc_id, kb_id)
            return doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        with get_session() as session:
            return session.get(Document, doc_id)

    def list_documents(self, kb_id: str, limit: int = 50, offset: int = 0) -> tuple[list, int]:
        with get_session() as session:
            _check_kb_exists(session, kb_id)
            stmt = select(Document).where(Document.kb_id == kb_id)
            count_stmt = stmt.with_only_columns(func.count()).order_by(None)
            total = session.scalar(count_stmt)
            stmt = stmt.order_by(Document.doc_id).limit(limit).offset(offset)
            items = session.scalars(stmt).all()
            return items, total

    def update_document(self, doc_id: str, **kwargs) -> Optional[Document]:
        allowed = {"content", "meta_json", "file_name", "file_path"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if "meta_json" in updates and isinstance(updates["meta_json"], dict):
            import json
            updates["meta_json"] = json.dumps(updates["meta_json"])
        if not updates:
            return self.get_document(doc_id)
        with get_session() as session:
            stmt = update(Document).where(Document.doc_id == doc_id).values(**updates)
            result = session.execute(stmt)
            with _wrap_db_error("update_document", doc_id=doc_id):
                session.commit()
            if result.rowcount == 0:
                return None
            return session.get(Document, doc_id)

    def delete_document(self, doc_id: str) -> bool:
        with get_session() as session:
            doc = session.get(Document, doc_id)
            if not doc:
                return False
            session.delete(doc)
            with _wrap_db_error("delete_document", doc_id=doc_id):
                session.commit()
            return True

    # ---------- 重新索引提交 ----------
    def submit_reindex_task(self, kb_id: str, force: bool, task_id: str):
        """
        将重新索引任务提交到后台队列。
        这里仅做状态更新并记录日志，实际任务需由任务队列（如 arq）触发。
        """
        # 至少更新状态为 PROCESSING
        update_kb_status(kb_id, IndexingStatus.PROCESSING)
        logger.info("重新索引任务已注册 | kb_id=%s task_id=%s force=%s", kb_id, task_id, force)
        # 真实场景下应推送任务到队列，例如:
        # await arq_pool.enqueue_job("reindex_kb", kb_id=kb_id, force=force, task_id=task_id)
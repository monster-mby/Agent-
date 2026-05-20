"""
会话管理模块 - 基于 SQLAlchemy 的会话 CRUD + LangGraph Checkpoint 桥接

核心设计：
- 展示层（sessions 表）：面向用户的会话元数据 + 消息历史
- 执行层（checkpoints 表）：LangGraph 内部状态快照（由 langgraph-checkpoint-sqlite 管理）
- 关联键：langgraph_thread_id（UUID v4），两个表通过此字段桥接

依赖：
    pip install sqlalchemy pydantic pydantic-settings langgraph-checkpoint-sqlite
"""

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Generator

from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from sqlalchemy import (
    Column, String, Text, TIMESTAMP, Index, create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session as OrmSession, sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool
from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger("session_manager")


# ═══════════════════════════════════════════════════════
# 配置模型
# ═══════════════════════════════════════════════════════

class SessionConfig(BaseSettings):
    """会话管理配置（支持环境变量 + 构造函数覆盖）"""
    model_config = ConfigDict(env_prefix="SESSION_")

    db_path: str = "data/checkpoints.sqlite"
    """SQLite 数据库文件路径，环境变量: SESSION_DB_PATH"""
    db_url: str = ""
    """完整数据库 URL；为空时自动从 db_path 拼接 sqlite:///... 支持 :memory: 用于测试"""
    wal_mode: bool = True
    """是否启用 WAL 模式（生产环境推荐开启）"""
    synchronous: str = "NORMAL"
    """SQLite synchronous 设置（NORMAL 兼顾性能与安全）"""

    @property
    def database_url(self) -> str:
        if self.db_url:
            return self.db_url
        # 支持内存数据库简写
        if self.db_path == ":memory:":
            return "sqlite:///:memory:"
        return f"sqlite:///{self.db_path}"


# ═══════════════════════════════════════════════════════
# Pydantic 数据模型 (v2)
# ═══════════════════════════════════════════════════════

class Message(BaseModel):
    """单条对话消息"""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: str  # "user" | "assistant" | "system"
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)


class Session(BaseModel):
    """会话对象（展示层）"""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str = "未命名会话"
    knowledge_base_ids: list[str] = Field(default_factory=list)
    langgraph_thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_history: list[Message] = Field(default_factory=list)
    status: str = "active"  # "active" | "archived" | "deleted"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)
    rules: list[str] = Field(default_factory=list)  # ✅ 新增：规则 ID 列表


# ═══════════════════════════════════════════════════════
# SQLAlchemy ORM 模型
# ═══════════════════════════════════════════════════════

class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    """sessions 表的 ORM 映射"""
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False, default="未命名会话")
    knowledge_base_ids = Column(Text, nullable=False, default="[]")       # JSON 数组
    langgraph_thread_id = Column(String, nullable=False, unique=True, index=True)
    message_history = Column(Text, nullable=False, default="[]")          # JSON 数组
    status = Column(String, nullable=False, default="active", index=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    metadata_ = Column("metadata", Text, default="{}")
    rules = Column(Text, nullable=False, default="[]")  # ✅ 新增：JSON 数组（规则 ID 列表）# JSON 对象


# ═══════════════════════════════════════════════════════
# 会话管理器
# ═══════════════════════════════════════════════════════

class SessionManager:
    """
    会话管理器 - 提供会话 CRUD + LangGraph Checkpoint 桥接

    使用示例::

        # 生产环境
        sm = SessionManager()

        # 测试环境（内存数据库）
        sm = SessionManager(config=SessionConfig(db_path=":memory:"))

        # 上下文管理器
        with SessionManager() as sm:
            session = sm.create_session(user_id="u1", name="测试")

        # 获取 LangGraph checkpointer
        checkpointer = sm.get_checkpointer()
    """

    # ── 构造 & 初始化 ──────────────────────────────

    def __init__(self, config: Optional[SessionConfig] = None, checkpointer: SqliteSaver = None):
        """
        初始化 SessionManager

        Args:
            config: 会话配置（可选，默认使用 SessionConfig()）
            checkpointer: LangGraph Checkpointer 实例（必传，避免与 checkpointer.py 单例冲突）

        Example:
            >>> from src.agent.langgraph.checkpointer import get_checkpointer
            >>> sm = SessionManager(
            ...     config=SessionConfig(db_path=":memory:"),
            ...     checkpointer=get_checkpointer()
            ... )
        """
        if checkpointer is None:
            raise ValueError(
                "checkpointer 参数必须提供。请从 src.agent.langgraph.checkpointer "
                "导入 get_checkpointer() 并传入，避免与全局单例冲突。\n"
                "Example:\n"
                "    from src.agent.langgraph.checkpointer import get_checkpointer\n"
                "    sm = SessionManager(checkpointer=get_checkpointer())"
            )

        self.config = config or SessionConfig()
        self._checkpointer = checkpointer  # ✅ 只接受注入，不自建
        self._closed = False  # ✅ 新增：标记是否已关闭

        # 连接池
        connect_args = {"check_same_thread": False}
        poolclass = None
        if self.config.db_path == ":memory:":
            # 内存数据库必须是单连接 StaticPool，否则多连接看不到彼此数据
            poolclass = StaticPool

        self._engine = create_engine(
            self.config.database_url,
            connect_args=connect_args,
            poolclass=poolclass,
            echo=False,
        )
        # ✅ 修复：使用 scoped_session 按线程隔离 session
        self._SessionFactory = sessionmaker(bind=self._engine)
        self._SessionLocal = scoped_session(self._SessionFactory)

        # 初始化表
        Base.metadata.create_all(self._engine)

        # WAL 模式（仅文件数据库）
        if self.config.wal_mode and self.config.db_path != ":memory:":
            with self._engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text(f"PRAGMA synchronous={self.config.synchronous}"))
                conn.commit()

        logger.info("SessionManager initialized (db=%s)", self.config.db_path)

    # ── 上下文管理器 ────────────────────────────────

    def __enter__(self) -> "SessionManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # ✅ 修改：不销毁连接池，只做日志（长生命周期对象不应在 __exit__ 中 dispose）
        logger.debug("SessionManager context exited (db=%s)", self.config.db_path)
        return False

    def close(self):
        """
        显式释放资源（仅在应用关闭时调用）

        Warning:
            调用后所有数据库操作将失败，请确保在应用生命周期结束时调用。
        """
        self._closed = True  # ✅ 修复：标记已关闭
        self._engine.dispose()
        logger.info("SessionManager resources released (db=%s)", self.config.db_path)

    # ── 内部工具 ────────────────────────────────────

    @contextmanager
    def _session(self) -> Generator[OrmSession, None, None]:
        """获取 ORM Session（自动关闭，防泄漏）"""
        if self._closed:  # ✅ 修复：检查是否已关闭
            raise RuntimeError("SessionManager 已关闭，不可再操作")
        db = self._SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _row_to_session(row: SessionRow) -> Session:
        """将 ORM 行转回 Pydantic Session"""
        return Session(
            session_id=row.session_id,
            user_id=row.user_id,
            name=row.name,
            knowledge_base_ids=json.loads(row.knowledge_base_ids),
            langgraph_thread_id=row.langgraph_thread_id,
            message_history=[Message(**m) for m in json.loads(row.message_history)],
            status=row.status,
            created_at=row.created_at.replace(tzinfo=timezone.utc) if row.created_at else datetime.now(timezone.utc),
            updated_at=row.updated_at.replace(tzinfo=timezone.utc) if row.updated_at else datetime.now(timezone.utc),
            metadata=json.loads(row.metadata_),
            rules=json.loads(row.rules),  # ✅ 新增：反序列化 rules
        )

    def get_checkpointer(self) -> SqliteSaver:
        """获取 LangGraph checkpointer 实例"""
        return self._checkpointer

    # ── CRUD 操作 ───────────────────────────────────

    def create_session(
        self,
        user_id: str,
        name: str = "未命名会话",
        knowledge_base_ids: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        rules: Optional[list[str]] = None,  # ✅ 新增：rules 参数
    ) -> Session:
        """创建新会话"""
        # ✅ 修复：校验 user_id 非空
        if not user_id or not isinstance(user_id, str):
            raise ValueError(f"user_id 必须是非空字符串，实际为: {repr(user_id)}")
        session = Session(
            user_id=user_id,
            name=name,
            knowledge_base_ids=knowledge_base_ids or [],
            metadata=metadata or {},
            rules=rules or [],  # ✅ 新增：初始化 rules
        )
        with self._session() as db:
            row = SessionRow(
                session_id=session.session_id,
                user_id=session.user_id,
                name=session.name,
                knowledge_base_ids=json.dumps(session.knowledge_base_ids, ensure_ascii=False),
                langgraph_thread_id=session.langgraph_thread_id,
                message_history=json.dumps([], ensure_ascii=False),
                status=session.status,
                metadata_=json.dumps(session.metadata, ensure_ascii=False),
                rules=json.dumps(session.rules, ensure_ascii=False),  # ✅ 新增：序列化 rules
            )
            db.add(row)
        logger.info("Session created: %s", session.session_id)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """根据 session_id 获取会话"""
        with self._session() as db:
            row = db.query(SessionRow).filter_by(session_id=session_id).first()
            return self._row_to_session(row) if row else None

    def get_session_by_thread_id(self, langgraph_thread_id: str) -> Optional[Session]:
        """根据 LangGraph thread_id 反查会话"""
        with self._session() as db:
            row = db.query(SessionRow).filter_by(langgraph_thread_id=langgraph_thread_id).first()
            return self._row_to_session(row) if row else None

    def list_sessions(
            self,
            user_id: str,
            status: str = "active",
            limit: int = 50,
            offset: int = 0,
    ) -> list[Session]:
        """列出用户的会话（分页，按更新时间倒序）"""
        with self._session() as db:
            query = db.query(SessionRow).filter_by(user_id=user_id)

            # ✅ 修复：支持 status='all' 查询所有状态
            if status != "all":
                query = query.filter_by(status=status)

            # ✅ 修复：使用已构建的 query 变量，而非重新查询
            rows = (
                query
                .order_by(SessionRow.updated_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [self._row_to_session(r) for r in rows]

    def update_session(self, session_id: str, **kwargs) -> Optional[Session]:
        """更新会话元数据（name / status / metadata / knowledge_base_ids）"""
        allowed = {"name", "status", "metadata", "knowledge_base_ids", "rules"}  # ✅ 新增：rules
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return self.get_session(session_id)

        # JSON 字段序列化
        if "metadata" in updates:
            updates["metadata_"] = json.dumps(updates.pop("metadata"), ensure_ascii=False)
        if "knowledge_base_ids" in updates:
            updates["knowledge_base_ids"] = json.dumps(updates.pop("knowledge_base_ids"), ensure_ascii=False)
        if "rules" in updates:  # ✅ 新增：rules 序列化
            updates["rules"] = json.dumps(updates.pop("rules"), ensure_ascii=False)

        updates["updated_at"] = datetime.now(timezone.utc)

        with self._session() as db:
            db.query(SessionRow).filter_by(session_id=session_id).update(updates)
        logger.info("Session updated: %s fields=%s", session_id, list(kwargs.keys()))
        return self.get_session(session_id)

    def archive_session(self, session_id: str) -> Optional[Session]:
        """归档会话"""
        return self.update_session(session_id, status="archived")

    def delete_session(self, session_id: str) -> bool:
        """软删除会话"""
        return self.update_session(session_id, status="deleted") is not None

    # ── 消息操作 ────────────────────────────────────

    def add_message(self, session_id: str, message: Message) -> Optional[Session]:
        """追加一条消息到会话"""
        with self._session() as db:
            row = db.query(SessionRow).filter_by(session_id=session_id).with_for_update().first()
            if not row:
                logger.warning("Session not found: %s", session_id)
                return None

            history = json.loads(row.message_history)
            history.append(message.model_dump(mode="json"))
            row.message_history = json.dumps(history, ensure_ascii=False)
            row.updated_at = datetime.now(timezone.utc)
        logger.info("Message added to session %s, total=%d", session_id, len(history))
        return self.get_session(session_id)

    def add_messages_batch(self, session_id: str, messages: list[Message]) -> Optional[Session]:
        """批量追加消息（减少写事务次数）"""
        if not messages:
            return self.get_session(session_id)

        with self._session() as db:
            row = db.query(SessionRow).filter_by(session_id=session_id).with_for_update().first()
            if not row:
                logger.warning("Session not found: %s", session_id)
                return None

            history = json.loads(row.message_history)
            history.extend(m.model_dump(mode="json") for m in messages)
            row.message_history = json.dumps(history, ensure_ascii=False)
            row.updated_at = datetime.now(timezone.utc)
        logger.info("Batch: %d messages added to session %s, total=%d", len(messages), session_id, len(history))
        return self.get_session(session_id)

    # 文件：session_manager.py 第 375-382 行
    def get_message_history(self, session_id: str, limit: Optional[int] = None) -> Optional[list[Message]]:
        """获取会话消息历史（支持截取最近 N 条）"""
        session = self.get_session(session_id)
        if not session:
            return None  # ✅ 修复：不存在时返回 None 而非 []
        if limit is not None:  # ✅ 修复：limit=0 时也执行截取
            return session.message_history[-limit:] if limit > 0 else []
        return session.message_history

    # ── 统计 ────────────────────────────────────────

    def count_sessions(self, user_id: str, status: str = "active") -> int:
        """统计用户活跃会话数"""
        with self._session() as db:
            return db.query(SessionRow).filter_by(user_id=user_id, status=status).count()
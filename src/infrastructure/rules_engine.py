"""
规则引擎模块 - 管理会话级别的 LLM 行为规则

核心设计：
- Rule 模型：rule_id, session_id, content, priority, category, enabled, created_by
- 存储：SQLite rules 表，复合索引 (session_id, enabled, priority DESC)
- 检索：按 session_id + enabled 过滤，按 priority 降序排列
- 注入：RulesInjectorNode 将规则拼接为 System Prompt 前缀
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List
from contextlib import contextmanager
from enum import StrEnum

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Boolean,
    Index,
    TIMESTAMP,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session as OrmSession, sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool

# ✅ 新增：从 rule_templates 导入
from src.infrastructure.rule_templates import RuleCategory


logger = logging.getLogger("rules_engine")


# ═══════════════════════════════════════════════════════
# 枚举
# ═══════════════════════════════════════════════════════

class RuleCategory(StrEnum):
    """规则分类枚举，约束合法取值"""
    CODE_REVIEW = "code_review"
    TECH_WRITING = "tech_writing"
    COMPLIANCE = "compliance"
    TEACHING = "teaching"
    GENERAL = "general"


# ═══════════════════════════════════════════════════════
# Pydantic 数据模型
# ═══════════════════════════════════════════════════════

class Rule(BaseModel):
    """规则对象"""
    model_config = {"from_attributes": True}

    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    content: str  # 规则内容（如"始终用中文回答"）
    priority: int = Field(default=3, ge=1, le=5)  # 优先级 1-5，5 最高
    category: RuleCategory = RuleCategory.GENERAL
    enabled: bool = True
    created_by: str = "system"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════
# SQLAlchemy ORM 模型
# ═══════════════════════════════════════════════════════

class Base(DeclarativeBase):
    pass


class RuleRow(Base):
    """rules 表的 ORM 映射"""
    __tablename__ = "rules"

    rule_id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    priority = Column(Integer, nullable=False, default=3)
    category = Column(String, nullable=False, default=RuleCategory.GENERAL)
    enabled = Column(Boolean, nullable=False, default=True)
    created_by = Column(String, nullable=False, default="system")
    created_at = Column(TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    # ── 复合索引：覆盖 get_enabled_rules 的核心查询 ──
    __table_args__ = (
        Index("idx_session_enabled_priority", "session_id", "enabled", "priority"),
    )


# ═══════════════════════════════════════════════════════
# 规则引擎
# ═══════════════════════════════════════════════════════

class RulesEngine:
    """
    规则引擎 - 管理会话级别的 LLM 行为规则

    使用示例::

        engine = RulesEngine(db_path="data/checkpoints.sqlite")
        engine.add_rule(session_id="s1", content="始终用中文回答", priority=5)
        prefix = engine.build_system_prefix("s1")
        engine.close()
    """

    # ── 内容校验常量 ──
    _MAX_CONTENT_LENGTH = 4000  # 单条规则最大字符数

    def __init__(
        self,
        db_path: str = "data/checkpoints.sqlite",
        pool_size: int = 5,
        echo: bool = False,
    ):
        """
        初始化 RulesEngine

        Args:
            db_path: SQLite 数据库路径，支持 :memory:
            pool_size: 连接池大小
            echo: 是否输出 SQL 日志
        """
        self.db_path = db_path

        # WAL 模式 + 多线程访问
        connect_args = {"check_same_thread": False}
        poolclass = None
        if db_path == ":memory:":
            poolclass = StaticPool
            connect_args = {}  # :memory: 必须单线程
        # Deleted:else:
        # Deleted:    # 文件数据库启用 WAL，提升并发读性能
        # Deleted:    connect_args["pragma"] = "journal_mode=WAL"

        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args=connect_args,
            poolclass=poolclass,
            pool_size=pool_size,
            max_overflow=10,
            echo=echo,
        )
        self._SessionLocal = scoped_session(sessionmaker(bind=self._engine))

        # 初始化表
        Base.metadata.create_all(self._engine)
        # ✅ 修复：通过 text() 执行 PRAGMA 语句（仅文件数据库）
        if db_path != ":memory:":
            with self._engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.commit()

        logger.info("RulesEngine initialized (db=%s, pool_size=%d)", db_path, pool_size)
    # ── 资源释放 ────────────────────────────────────

    def close(self):
        """释放引擎资源：逐出所有 session 并销毁连接池"""
        self._SessionLocal.remove()
        self._engine.dispose()
        logger.info("RulesEngine closed (db=%s)", self.db_path)

    # ── 内部工具 ────────────────────────────────────

    @contextmanager
    def _session(self) -> OrmSession:
        """获取 ORM Session，自动提交/回滚/关闭"""
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
    def _row_to_rule(row: RuleRow) -> Rule:
        """ORM 行 → Pydantic 模型（通过 from_attributes 自动映射）"""
        return Rule.model_validate(row)

    @staticmethod
    def _validate_content(content: str) -> str:
        """校验规则内容非空 & 长度合理，抛出 ValueError"""
        if not content or not content.strip():
            raise ValueError("规则内容不能为空")
        if len(content) > RulesEngine._MAX_CONTENT_LENGTH:
            raise ValueError(
                f"规则内容不能超过 {RulesEngine._MAX_CONTENT_LENGTH} 字符，当前 {len(content)}"
            )
        return content.strip()

    # ── 核心操作 ────────────────────────────────────

    def add_rule(
        self,
        session_id: str,
        content: str,
        priority: int = 3,
        category: RuleCategory = RuleCategory.GENERAL,
        created_by: str = "system",
    ) -> Rule:
        """添加新规则"""
        content = self._validate_content(content)

        rule = Rule(
            session_id=session_id,
            content=content,
            priority=priority,
            category=category,
            enabled=True,
            created_by=created_by,
        )
        with self._session() as db:
            row = RuleRow(
                rule_id=rule.rule_id,
                session_id=rule.session_id,
                content=rule.content,
                priority=rule.priority,
                category=rule.category,
                enabled=rule.enabled,
                created_by=rule.created_by,
            )
            db.add(row)
        logger.info("Rule added: %s session=%s priority=%d category=%s",
                    rule.rule_id, session_id, priority, category)
        return rule

    def add_rules_batch(self, rules_data: List[dict]) -> List[Rule]:
        """
        批量添加规则

        Args:
            rules_data: 字典列表，每项含 session_id/content/priority/category/created_by
        Returns:
            创建的 Rule 对象列表
        """
        rules: List[Rule] = []
        with self._session() as db:
            for data in rules_data:
                content = self._validate_content(data.get("content", ""))
                rule = Rule(
                    session_id=data["session_id"],
                    content=content,
                    priority=data.get("priority", 3),
                    category=data.get("category", RuleCategory.GENERAL),
                    enabled=True,
                    created_by=data.get("created_by", "system"),
                )
                db.add(RuleRow(
                    rule_id=rule.rule_id,
                    session_id=rule.session_id,
                    content=rule.content,
                    priority=rule.priority,
                    category=rule.category,
                    enabled=rule.enabled,
                    created_by=rule.created_by,
                ))
                rules.append(rule)
        logger.info("Batch: %d rules added", len(rules))
        return rules

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """根据 rule_id 获取规则"""
        with self._session() as db:
            row = db.query(RuleRow).filter_by(rule_id=rule_id).first()
            return self._row_to_rule(row) if row else None

    def get_enabled_rules(self, session_id: str) -> List[Rule]:
        """
        获取会话的启用规则（按 priority 降序）

        走复合索引 idx_session_enabled_priority 覆盖查询
        """
        with self._session() as db:
            rows = (
                db.query(RuleRow)
                .filter_by(session_id=session_id, enabled=True)
                .order_by(RuleRow.priority.desc())
                .all()
            )
            return [self._row_to_rule(r) for r in rows]

    def update_rule(self, rule_id: str, **kwargs) -> Optional[Rule]:
        """更新规则（仅允许 content/priority/category/enabled 字段）"""
        allowed = {"content", "priority", "category", "enabled"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_rule(rule_id)

        # 若有 content 更新，先校验
        if "content" in updates:
            updates["content"] = self._validate_content(updates["content"])

        with self._session() as db:
            count = db.query(RuleRow).filter_by(rule_id=rule_id).update(updates)
        if count == 0:
            logger.warning("Rule update failed: %s not found", rule_id)
            return None
        logger.info("Rule updated: %s fields=%s", rule_id, list(updates.keys()))
        return self.get_rule(rule_id)

    def toggle_rule(self, rule_id: str) -> Optional[Rule]:
        """
        切换规则的启用状态

        使用单次 UPDATE enabled = NOT enabled，避免多次查询
        """
        with self._session() as db:
            result = (
                db.query(RuleRow)
                .filter_by(rule_id=rule_id)
                .update({"enabled": text("NOT enabled")}, synchronize_session="fetch")
            )
        if result == 0:
            logger.warning("Rule toggle failed: %s not found", rule_id)
            return None
        rule = self.get_rule(rule_id)
        logger.info("Rule toggled: %s → enabled=%s", rule_id, rule.enabled)
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        """软删除（逻辑删除）规则"""
        result = self.update_rule(rule_id, enabled=False)
        if result:
            logger.info("Rule soft-deleted: %s", rule_id)
        else:
            logger.warning("Rule delete failed: %s not found", rule_id)
        return result is not None

    # ── System Prompt 生成 ──────────────────────────

    def build_system_prefix(self, session_id: str) -> str:
        """
        构建 System Prompt 前缀

        将所有启用的规则按 priority 降序拼接，
        返回格式：
            [code_review] (优先级 5) 使用类型提示
            [general]   (优先级 3) 始终用中文回答
        """
        rules = self.get_enabled_rules(session_id)
        if not rules:
            return ""

        lines = []
        for r in rules:
            lines.append(f"[{r.category}] (优先级 {r.priority}) {r.content}")

        prefix = "\n".join(lines)
        logger.debug("Built system prefix for %s: %d rules, %d chars",
                     session_id, len(rules), len(prefix))
        return prefix

    # ── 扩展查询与模板支持 ──────────────────────────

    def list_rules(
            self,
            session_id: str,
            category: Optional[str] = None,
            enabled: Optional[bool] = None,
            sort_by: str = "priority",
            order: str = "desc",
            limit: int = 50,
            offset: int = 0,
    ) -> List[Rule]:
        """
        分页查询规则，支持过滤和排序

        Args:
            session_id: 会话 ID
            category: 规则分类过滤
            enabled: 启用状态过滤
            sort_by: 排序字段 (priority/created_at/category)
            order: 排序方向 (asc/desc)
            limit: 每页数量
            offset: 偏移量

        Returns:
            Rule 对象列表
        """
        with self._session() as db:
            query = db.query(RuleRow).filter_by(session_id=session_id)

            # 应用过滤条件
            if category is not None:
                query = query.filter_by(category=category)
            if enabled is not None:
                query = query.filter_by(enabled=enabled)

            # 应用排序
            if sort_by == "created_at":
                column = RuleRow.created_at
            elif sort_by == "category":
                column = RuleRow.category
            else:  # 默认按 priority
                column = RuleRow.priority

            if order == "asc":
                query = query.order_by(column.asc())
            else:
                query = query.order_by(column.desc())

            # 应用分页
            rows = query.offset(offset).limit(limit).all()
            return [self._row_to_rule(r) for r in rows]

    def apply_template(
            self,
            session_id: str,
            template_name: str,
            created_by: str,
            overrides=None,
    ) -> List[Rule]:
        """
        应用规则模板到会话

        Args:
            session_id: 会话 ID
            template_name: 模板名称
            created_by: 创建者
            overrides: 覆盖字段（CreateRuleRequest 对象或字典）

        Returns:
            创建的 Rule 对象列表
        """
        from src.infrastructure.rule_templates import get_template, apply_template as apply_tmpl

        # 获取模板
        template = get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        # 应用模板（使用 rule_templates 模块的函数）
        rules_data = apply_tmpl(
            template_name=template_name,
            session_id=session_id,
            created_by=created_by,
            overrides=overrides.model_dump() if hasattr(overrides, 'model_dump') else overrides,
        )

        # 批量创建规则
        return self.add_rules_batch(rules_data)

    def list_templates(self):
        """
        返回所有规则模板

        Returns:
            模板信息列表（字典）
        """
        from src.infrastructure.rule_templates import list_templates

        return list_templates()
"""
向量存储基础设施 — ChromaDB 集成

提供知识库级别的向量命名空间隔离管理。
支持：
    - 线程安全单例 + 依赖注入（可测试性）
    - 延迟初始化
    - namespace 合法性校验
    - 自定义异常体系
    - 结构化日志
"""

from __future__ import annotations

import logging
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import InvalidCollectionException  # ← 新增

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 自定义异常
# ═══════════════════════════════════════════════════════

class VectorStoreError(Exception):
    """向量存储通用业务异常基类"""
    pass


class CollectionNotFoundError(VectorStoreError):
    """指定集合不存在"""

    def __init__(self, namespace: str) -> None:
        super().__init__(f"向量集合不存在: namespace={namespace!r}")
        self.namespace = namespace


class CollectionExistsError(VectorStoreError):
    """集合已存在（重复创建）"""

    def __init__(self, namespace: str) -> None:
        super().__init__(f"向量集合已存在: namespace={namespace!r}")
        self.namespace = namespace


class InvalidNamespaceError(VectorStoreError):
    """命名空间不符合 ChromaDB 命名规则"""

    def __init__(self, namespace: str, reason: str) -> None:
        super().__init__(
            f"非法命名空间: namespace={namespace!r}, reason={reason}"
        )
        self.namespace = namespace
        self.reason = reason


class VectorStoreConnectionError(VectorStoreError):
    """无法连接到 ChromaDB 后端"""
    pass


# ═══════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════

class VectorStoreSettings(BaseModel):
    """ChromaDB 向量存储配置（可通过环境变量覆盖）"""
    persist_path: str = Field(
        default="data/vector_store",
        description="持久化存储路径",
    )
    anonymized_telemetry: bool = Field(
        default=False,
        description="是否允许 Chroma 匿名遥测",
    )
    allow_reset: bool = Field(
        default=False,
        description="是否允许 reset() 操作（生产环境必须 False）",
    )
    is_persistent: bool = Field(
        default=True,
        description="是否持久化（False 则为纯内存模式）",
    )

    def to_chroma_settings(self) -> ChromaSettings:
        """转换为 ChromaDB Settings 对象"""
        return ChromaSettings(
            anonymized_telemetry=self.anonymized_telemetry,
            allow_reset=self.allow_reset,
            is_persistent=self.is_persistent,
        )


# ═══════════════════════════════════════════════════════
# 响应模型
# ═══════════════════════════════════════════════════════

@dataclass
class CollectionInfo:
    """集合信息（用于 list_collections）"""
    name: str
    metadata: Dict[str, str] = field(default_factory=dict)
    count: Optional[int] = None  # 文档数量（需调用 .count()）


# ═══════════════════════════════════════════════════════
# 向量存储管理器
# ═══════════════════════════════════════════════════════

class VectorStoreManager:
    """
    ChromaDB 向量存储管理器（线程安全单例 + 依赖注入）

    使用方式:

        # 生产：默认单例
        store = VectorStoreManager.default()

        # 测试：注入 Mock 客户端
        store = VectorStoreManager(client=mock_client)

        # 显式释放资源
        store.close()
    """

    # --- 单例基础设施 ---
    _instance: Optional["VectorStoreManager"] = None
    _lock: threading.Lock = threading.Lock()
    _default_settings: VectorStoreSettings = VectorStoreSettings()

    # --- namespace 合法性校验 ---
    # ChromaDB 要求: 3-63 字符，字母数字/下划线/连字符，首尾必须是字母或数字
    _NAMESPACE_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$")

    # --- 用于延迟初始化的哨兵 ---
    _UNINITIALIZED = object()

    def __init__(
        self,
        client: Optional[ClientAPI] = None,
        settings: Optional[VectorStoreSettings] = None,
    ):
        """
        Args:
            client: 可选，注入 ChromaDB 客户端（测试用）
            settings: 可选，向量存储配置
        """
        self._settings = settings or self._default_settings

        if client is not None:
            # 依赖注入路径：直接使用外部客户端
            self._client: ClientAPI = client
            self._initialized = True
            logger.info("ChromaDB 使用注入客户端")
        else:
            # 延迟初始化路径：标记未初始化，首次访问时创建
            self._client = self._UNINITIALIZED  # type: ignore[assignment]
            self._initialized = False

        # 每个实例自己的锁（用于延迟初始化的线程安全）
        self._init_lock = threading.Lock()

    @classmethod
    def default(cls) -> "VectorStoreManager":
        """
        获取默认单例实例（线程安全）

        Returns:
            共享的 VectorStoreManager 实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # double-checked locking
                    cls._instance = cls(settings=cls._default_settings)
                    logger.info("VectorStoreManager 单例已创建")
        return cls._instance

    @classmethod
    def configure_default(
        cls,
        settings: VectorStoreSettings,
    ) -> None:
        """
        配置默认单例的参数（必须在首次调用 default() 之前调用）

        Raises:
            RuntimeError: 单例已创建后才调用
        """
        with cls._lock:
            if cls._instance is not None:
                raise RuntimeError(
                    "VectorStoreManager 单例已创建，无法重新配置。"
                    "请在首次访问 default() 之前调用 configure_default()。"
                )
            cls._default_settings = settings
            logger.info("VectorStoreManager 默认配置已更新")

    # --- 延迟初始化的 property ---

    @property
    def _db(self) -> ClientAPI:
        """
        获取 ChromaDB 客户端（延迟初始化，线程安全）

        首次访问时创建 PersistentClient，后续直接返回缓存实例。
        """
        if self._initialized:
            return self._client

        with self._init_lock:
            if self._initialized:  # double-checked locking
                return self._client

            logger.info(
                "初始化 ChromaDB PersistentClient | path=%s",
                self._settings.persist_path,
            )
            try:
                # 构建 settings，显式禁用遥测（避免 PostHog 兼容性错误）
                chroma_settings = self._settings.to_chroma_settings()
                # 强制禁用匿名遥测
                chroma_settings.anonymized_telemetry = False
                self._client = chromadb.PersistentClient(
                    path=self._settings.persist_path,
                    settings=chroma_settings,
                )

            except Exception as exc:
                logger.exception("ChromaDB 客户端初始化失败")
                raise VectorStoreConnectionError(
                    f"无法连接到 ChromaDB: {exc}"
                ) from exc

            return self._client

    # --- 上下文管理器 ---

    def close(self) -> None:
        """显式释放 ChromaDB 连接资源"""
        with self._init_lock:
            if self._initialized and self._client is not self._UNINITIALIZED:
                # ChromaDB PersistentClient 没有显式 close 方法，
                # 但我们可以清空引用让 GC 处理
                self._client = self._UNINITIALIZED  # type: ignore[assignment]
                self._initialized = False
                logger.info("ChromaDB 客户端已关闭")

    def __enter__(self) -> "VectorStoreManager":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # --- namespace 校验 ---

    @classmethod
    def validate_namespace(cls, namespace: str) -> None:
        """
        校验 namespace 是否符合 ChromaDB 命名规则

        Raises:
            InvalidNamespaceError: namespace 不合法
        """
        if not namespace:
            raise InvalidNamespaceError(namespace, "namespace 不能为空")
        if len(namespace) < 3:
            raise InvalidNamespaceError(
                namespace, f"长度不足（当前 {len(namespace)}，最小 3）"
            )
        if len(namespace) > 63:
            raise InvalidNamespaceError(
                namespace, f"长度超限（当前 {len(namespace)}，最大 63）"
            )
        if not cls._NAMESPACE_PATTERN.match(namespace):
            raise InvalidNamespaceError(
                namespace,
                "只能包含字母、数字、下划线和连字符，且首尾必须是字母或数字",
            )

    # ═══════════════════════════════════════════════════
    # 核心操作
    # ═══════════════════════════════════════════════════

    def health_check(self) -> bool:
        """
        检查 ChromaDB 连接是否正常

        Returns:
            True 表示连接正常
        """
        try:
            _ = self._db.heartbeat()
            logger.debug("ChromaDB 心跳正常")
            return True
        except Exception as exc:
            logger.warning("ChromaDB 心跳失败: %s", exc)
            return False

    def collection_exists(self, namespace: str) -> bool:
        """
        检查指定命名空间的集合是否存在

        Args:
            namespace: 向量集合命名空间

        Returns:
            True 表示集合存在
        """
        self.validate_namespace(namespace)
        try:
            self._db.get_collection(name=namespace)
            return True
        except (chromadb.errors.InvalidCollectionException, ValueError):
            return False

    def create_collection(
        self,
        namespace: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        为知识库创建独立的向量集合

        Args:
            namespace: 向量集合命名空间（符合 ChromaDB 命名规则）
            metadata: 可选元数据（如 kb_id、kb_name、created_at）

        Raises:
            InvalidNamespaceError: 命名空间不合法
            CollectionExistsError: 集合已存在
            VectorStoreConnectionError: 数据库连接异常
            VectorStoreError: 其他 ChromaDB 错误
        """
        self.validate_namespace(namespace)

        logger.info("创建向量集合 | namespace=%s metadata=%s", namespace, metadata)
        try:
            self._db.create_collection(
                name=namespace,
                metadata=metadata,
            )
            logger.info("向量集合已创建 | namespace=%s", namespace)
        except chromadb.errors.UniqueConstraintError as exc:
            logger.warning("向量集合已存在 | namespace=%s", namespace)
            raise CollectionExistsError(namespace) from exc
        except (chromadb.errors.ChromaError, RuntimeError) as exc:
            logger.error("创建向量集合失败 | namespace=%s error=%s", namespace, exc)
            raise VectorStoreError(
                f"创建向量集合失败: namespace={namespace!r}, error={exc}"
            ) from exc

    def get_or_create_collection(
        self,
        namespace: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> chromadb.Collection:
        """
        获取或创建向量集合（无并发窗口）

        Args:
            namespace: 向量集合命名空间
            metadata: 仅在创建时使用的元数据

        Returns:
            ChromaDB Collection 对象

        Raises:
            InvalidNamespaceError: 命名空间不合法
            VectorStoreConnectionError: 数据库连接异常
        """
        self.validate_namespace(namespace)

        logger.info("获取或创建向量集合 | namespace=%s", namespace)
        try:
            collection = self._db.get_or_create_collection(
                name=namespace,
                metadata=metadata,
            )
            logger.info("向量集合就绪 | namespace=%s", namespace)
            return collection
        except (chromadb.errors.ChromaError, RuntimeError) as exc:
            logger.error(
                "获取或创建向量集合失败 | namespace=%s error=%s", namespace, exc
            )
            raise VectorStoreError(
                f"获取或创建向量集合失败: namespace={namespace!r}, error={exc}"
            ) from exc

    def get_collection(self, namespace: str) -> chromadb.Collection:
        """
        获取指定命名空间的集合

        Args:
            namespace: 向量集合命名空间

        Returns:
            ChromaDB Collection 对象

        Raises:
            InvalidNamespaceError: 命名空间不合法
            CollectionNotFoundError: 集合不存在
            VectorStoreConnectionError: 数据库连接异常
        """
        self.validate_namespace(namespace)

        try:
            collection = self._db.get_collection(name=namespace)
            logger.debug("获取向量集合 | namespace=%s", namespace)
            return collection
        except (chromadb.errors.InvalidCollectionException, ValueError) as exc:
            logger.warning("向量集合不存在 | namespace=%s", namespace)
            raise CollectionNotFoundError(namespace) from exc
        except (chromadb.errors.ChromaError, RuntimeError) as exc:
            logger.error("获取向量集合失败 | namespace=%s error=%s", namespace, exc)
            raise VectorStoreError(
                f"获取向量集合失败: namespace={namespace!r}, error={exc}"
            ) from exc

    def delete_collection(self, namespace: str) -> None:
        """
        删除知识库对应的向量集合

        Args:
            namespace: 向量集合命名空间

        Raises:
            InvalidNamespaceError: 命名空间不合法
            CollectionNotFoundError: 集合不存在
            VectorStoreConnectionError: 数据库连接异常
        """
        self.validate_namespace(namespace)

        logger.info("删除向量集合 | namespace=%s", namespace)
        try:
            self._db.delete_collection(name=namespace)
            logger.info("向量集合已删除 | namespace=%s", namespace)
        except (chromadb.errors.InvalidCollectionException, ValueError) as exc:
            logger.warning("删除失败：集合不存在 | namespace=%s", namespace)
            raise CollectionNotFoundError(namespace) from exc
        except (chromadb.errors.ChromaError, RuntimeError) as exc:
            logger.error("删除向量集合失败 | namespace=%s error=%s", namespace, exc)
            raise VectorStoreError(
                f"删除向量集合失败: namespace={namespace!r}, error={exc}"
            ) from exc

    def list_collections(self) -> List[CollectionInfo]:
        """
        列出所有向量集合（含元数据与文档计数）

        Returns:
            CollectionInfo 列表

        Raises:
            VectorStoreConnectionError: 数据库连接异常
            VectorStoreError: 其他 ChromaDB 错误
        """
        logger.debug("列出所有向量集合")
        try:
            raw_collections = self._db.list_collections()
            result: List[CollectionInfo] = []
            for col in raw_collections:
                try:
                    count = col.count()
                except Exception:
                    count = None

                # 安全提取 metadata（不同版本 API 差异）
                try:
                    meta = col.metadata or {}
                except Exception:
                    meta = {}

                result.append(
                    CollectionInfo(
                        name=col.name,
                        metadata=dict(meta) if meta else {},
                        count=count,
                    )
                )

            logger.info("列出向量集合 | count=%d", len(result))
            return result
        except (chromadb.errors.ChromaError, RuntimeError) as exc:
            logger.error("列出向量集合失败 | error=%s", exc)
            raise VectorStoreError(f"列出向量集合失败: {exc}") from exc

    # 放在 VectorStoreManager 类的内部（与 list_collections 等方法同级）
    def search(
            self,
            collection_name: str,
            query_embedding: list[float],
            top_k: int,
    ) -> list[dict]:
        """
        在指定集合中执行向量相似度检索。

        Args:
            collection_name: 目标集合名称（对应知识库 ID）
            query_embedding: 查询向量（浮点数列表）
            top_k: 返回的最大文档数

        Returns:
            list[dict]: 每个字典包含：
                - id: str            # 文档唯一 ID
                - text: str          # 文档文本
                - embedding: list   # 文档对应的向量（若库中已存储）
                - metadata: dict    # 附加元数据

        Raises:
            VectorStoreConnectionError: 数据库连接异常或集合不存在
            VectorStoreError: 其他 ChromaDB 错误
        """
        try:
            # 获取集合（只读，不创建）
            collection = self._db.get_collection(collection_name)
        except (InvalidCollectionException, ValueError):
            # ChromaDB 新版抛出 InvalidCollectionException，旧版抛出 ValueError
            logger.error("向量集合不存在: %s", collection_name)
            raise VectorStoreConnectionError(
                f"集合 '{collection_name}' 不存在，请先创建知识库"
            )
        except Exception as e:
            logger.error("获取集合失败: %s", e, exc_info=True)
            raise VectorStoreConnectionError(f"无法连接集合 '{collection_name}': {e}")

        try:
            # 执行查询，请求返回文档、元数据和向量
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "embeddings"]
            )
        except Exception as e:
            logger.error("向量查询失败: %s", e, exc_info=True)
            raise VectorStoreError(f"检索失败: {e}")

        # 标准化输出格式
        documents = []
        # 兼容不同 ChromaDB 版本返回结构（有时是二维数组）
        ids = results.get("ids", [[]])[0] if results.get("ids") else []
        texts = results.get("documents", [[]])[0] if results.get("documents") else []
        embeddings = results.get("embeddings", [[]])[0] if results.get("embeddings") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

        for i in range(len(ids)):
            doc = {
                "id": ids[i] if i < len(ids) else f"result_{i}",
                "text": texts[i] if i < len(texts) else "",
                "embedding": embeddings[i] if i < len(embeddings) else [],
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }
            documents.append(doc)

        logger.debug("检索完成，集合=%s，返回 %d 条", collection_name, len(documents))
        return documents
    # ═══════════════════════════════════════════════════
    # 重置（仅测试环境）
    # ═══════════════════════════════════════════════════

    def reset(self) -> None:
        """
        重置整个向量数据库（删除所有集合）

        ⚠️  仅在 allow_reset=True 时可用，生产环境会直接抛异常。

        Raises:
            RuntimeError: allow_reset 未启用
            VectorStoreConnectionError: 数据库连接异常
        """
        if not self._settings.allow_reset:
            raise RuntimeError(
                "reset() 被禁用。如需使用，请在 VectorStoreSettings 中设置 "
                "allow_reset=True（生产环境严禁开启）。"
            )
        logger.warning("⚠️  正在重置整个向量数据库...")
        try:
            self._db.reset()
            logger.warning("向量数据库已重置")
        except (chromadb.errors.ChromaError, RuntimeError) as exc:
            logger.error("重置向量数据库失败 | error=%s", exc)
            raise VectorStoreError(f"重置失败: {exc}") from exc


# ═══════════════════════════════════════════════════════
# 全局默认实例
# ═══════════════════════════════════════════════════════

# 模块级变量，天然单例。首次使用时通过 VectorStoreManager.default() 获取。
# 如需注入 Mock，使用 VectorStoreManager(client=mock_client) 创建独立实例。
vector_store = VectorStoreManager.default()

# ✅ 新增：获取全局单例（供 conversation_memory 等模块使用）
def get_vector_store() -> VectorStoreManager:
    return vector_store
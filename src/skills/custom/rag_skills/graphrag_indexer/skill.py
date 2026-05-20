"""
GraphRAG 索引构建技能（生产优化版）

优化点清单：
  P0: pyarrow 读取 metadata 统计行数（替代 Pandas 全量加载）
  P0: _SUCCESS 标记文件 → 完整性校验
  P0: force_reindex 前清理旧产物
  P1: 文件名 sanitize（自动去除特殊字符）
  P1: overwrite 策略（覆盖/跳过/报错）
  P1: cleanup_temp → 自动清理临时文件
  P1: timeout 可配置
  P1: documents + input_files 同时生效（不再 elif 互斥）
  P2: Pydantic 输出模型
  P2: 日志级进度反馈
  P3: 环境变量优先级文档化

完全匹配 BaseSkill 接口：
  - 无 setup()，配置从环境变量读取
  - execute(**kwargs) → Dict[str, Any]，同步方法
  - 通过 SkillManager.call("graphrag_indexer", ...) 调用
"""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger(__name__)

# ============================================================
# Pydantic 输出模型（可选依赖）
# ============================================================
try:
    from pydantic import BaseModel, Field


    class GraphRAGIndexerInput(BaseModel):
        """GraphRAG 索引构建的输入模型"""
        documents: Optional[List[Dict[str, str]]] = Field(
            default=None,
            description='内存文档列表，格式为 [{"id": "doc1", "text": "文档内容..."}, ...]'
        )
        input_files: Optional[List[str]] = Field(
            default=None,
            description='外部文件路径列表，格式为 ["/path/to/doc.txt", ...]'
        )
        force_reindex: bool = Field(
            default=False,
            description="是否强制重建索引（会先清理旧产物）"
        )
        overwrite: str = Field(
            default="overwrite",
            description='同名文件处理策略："overwrite"（覆盖）/ "skip"（跳过）/ "error"（报错）'
        )
        cleanup_temp: bool = Field(
            default=False,
            description="是否在索引成功后删除通过 documents 写入的临时文件"
        )
        timeout: int = Field(
            default=3600,
            description="CLI 超时秒数"
        )

    class GraphRAGIndexerOutput(BaseModel):
        """GraphRAG 索引构建的输出模型"""
        success: bool = Field(..., description="是否成功")
        message: str = Field(default="", description="结果消息")
        output_dir: str = Field(default="", description="索引产物目录")
        entity_count: int = Field(default=0, description="实体数量")
        relationship_count: int = Field(default=0, description="关系数量")
        community_count: int = Field(default=0, description="社区数量")
        error: Optional[str] = Field(default=None, description="错误信息")
        index_fresh: bool = Field(default=True, description="索引是否是最新构建的")
        files_processed: int = Field(default=0, description="处理的文件数")

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # 降级：用普通 dict，下游代码不受影响


# ============================================================
# 工具函数
# ============================================================

def sanitize_filename(name: str, max_length: int = 100) -> str:
    """
    将任意字符串转为安全的文件名。

    策略：
      1. 优先用 python-slugify（如果已安装）
      2. 否则用正则去除/替换特殊字符
    """
    try:
        from slugify import slugify
        sanitized = slugify(name, max_length=max_length, word_boundary=True)
        return sanitized or "unnamed_document"
    except ImportError:
        # 降级方案：手动清理
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f ]', '_', name)  # 添加了空格匹配
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('._')
        if not sanitized:
            sanitized = "unnamed_document"
        return sanitized[:max_length]


def _get_parquet_row_count(file_path: Path) -> int:
    """用 pyarrow 读取 Parquet metadata 获取行数（不加载数据）"""
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(str(file_path)).num_rows
    except ImportError:
        # 降级：Pandas 只读 schema 不读数据
        try:
            import pandas as pd
            return len(pd.read_parquet(file_path, columns=[]))
        except Exception:
            return 0
    except Exception:
        return 0


# ============================================================
# 技能类
# ============================================================

class GraphRAGIndexerSkill(BaseSkill):
    """
    构建知识图谱索引。

    依赖：
      - pip install graphrag
      - graphrag init --root ./graphrag_data（执行一次初始化）

    配置优先级（从高到低）：
      1. execute() 参数
      2. 环境变量 GRAPHRAG_API_KEY / GRAPHRAG_ROOT
      3. graphrag_data/.env 文件
      4. graphrag_data/settings.yaml 中的默认值
    """

    # ============ 元数据 ============
    name: str = "graphrag_indexer"
    description: str = (
        "对文档集执行实体/关系提取，构建知识图谱索引。"
        "生成 entities.parquet / relationships.parquet / communities.parquet 等产物。"
        "支持内存文本和文件路径两种输入方式。"
    )
    version: str = "0.2.0"
    author: str = "dev-team"
    triggers: list[str] = [
        "构建图谱索引", "知识图谱索引", "graphrag index",
        "实体提取", "建立知识图谱", "索引文档"
    ]
    changelog: list[Dict[str, str]] = [
        {"version": "0.2.0", "change": (
            "P0: pyarrow 统计行数 / _SUCCESS 完整性校验 / force_reindex 清理旧产物; "
            "P1: 文件名 sanitize / overwrite 策略 / cleanup_temp / timeout 可配置; "
            "P2: Pydantic 输出模型 / 日志级进度反馈"
        )},
        {"version": "0.1.0", "change": "初始版本"},
    ]

    # 输入输出模型（供 SkillManager 校验用）
    input_schema = GraphRAGIndexerInput if HAS_PYDANTIC else None
    output_schema = GraphRAGIndexerOutput if HAS_PYDANTIC else None

    # ============ 初始化 ============
    def __init__(self):
        super().__init__()
        graphrag_root = os.getenv("GRAPHRAG_ROOT", "./graphrag_data")
        self.graphrag_root = Path(graphrag_root).resolve()
        self.input_dir = self.graphrag_root / "input"
        self.output_dir = self.graphrag_root / "output"
        self.artifacts_dir = self.output_dir / "artifacts"
        self._success_marker = self.artifacts_dir / "_SUCCESS"

        # 确保必要的目录存在
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            "GraphRAGIndexerSkill 初始化: root=%s, input=%s, output=%s",
            self.graphrag_root, self.input_dir, self.output_dir,
        )

    # ============ 核心执行 ============
    def execute(
            self,
            input_data: GraphRAGIndexerInput,
    ) -> Dict[str, Any]:
        """
        执行 GraphRAG 索引构建。

        Args:
            input_data: GraphRAG 索引构建输入 Pydantic 对象

        Returns:
            dict 包含 success / message / entity_count / relationship_count /
                 community_count / error / index_fresh / files_processed
        """
        documents = input_data.documents or []
        input_files = input_data.input_files or []
        force_reindex = input_data.force_reindex
        overwrite = input_data.overwrite
        cleanup_temp = input_data.cleanup_temp
        timeout = input_data.timeout
        temp_files_written: List[Path] = []
        files_processed = 0

        # ── 0. 参数校验 ──────────────────────────────────────
        # ── 0. 参数校验 ──
        if overwrite not in ("overwrite", "skip", "error"):
            return self._result(False, error=f"无效的 overwrite 策略...")

        # ── 0.5. 索引完整性检查（提前） ──
        if not force_reindex and self._is_index_valid():
            stats = self._get_index_stats()
            logger.info("索引已存在且完整，跳过构建...")
            return self._result(True, message=f"索引已存在...", **stats)

        # ── 1. 参数校验（需要构建时才检查） ──
        if not documents and not input_files and not list(self.input_dir.glob("*.txt")):
            return self._result(False, error="未提供文档...")

        # ── 1. force_reindex → 清理旧产物 ─────────────────────
        if force_reindex and self.artifacts_dir.exists():
            logger.info("force_reindex=True，清理旧索引产物: %s", self.artifacts_dir)
            shutil.rmtree(self.artifacts_dir)
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        # ── 1.5. 如果需要 skip，预先收集已存在文件 ────────────────────
        existing_files = set()
        if overwrite == "skip":
            existing_files = {f.stem for f in self.input_dir.glob("*.txt")}
            logger.debug("skip 模式：已存在文件 %s", existing_files)

        # ─ 2. 写入 documents（内存文本） ────────────────────
        if documents:
            logger.info("准备写入 %d 个内存文档到 input 目录", len(documents))
            seen_ids: Dict[str, int] = {}  # 用于去重后缀

            for doc in documents:
                doc_id = doc.get("id", "unnamed")
                text = doc.get("text", "")
                safe_name = sanitize_filename(doc_id)

                # 处理同名：加数字后缀
                if safe_name in seen_ids:
                    seen_ids[safe_name] += 1
                    stem, ext = os.path.splitext(safe_name)
                    safe_name = f"{stem}_{seen_ids[safe_name]}{ext or '.txt'}"
                else:
                    seen_ids[safe_name] = 0

                # skip 策略：如果文件已存在，跳过
                if overwrite == "skip" and safe_name in existing_files:
                    logger.debug("跳过已存在文件: %s", safe_name)
                    continue

                file_path = self.input_dir / f"{safe_name}.txt" if not safe_name.endswith(
                    ".txt") else self.input_dir / safe_name

                # overwrite 策略（error 检查）
                if file_path.exists():
                    if overwrite == "error":
                        return self._result(
                            False,
                            error=f"文件已存在且 overwrite='error': {file_path.name}"
                        )

                file_path.write_text(text, encoding="utf-8")
                temp_files_written.append(file_path)
                files_processed += 1
                logger.debug("写入临时文件: %s (%d 字符)", file_path.name, len(text))

        # ── 3. 复制 input_files（文件路径） ───────────────────
        if input_files:
            logger.info("准备复制 %d 个外部文件到 input 目录", len(input_files))
            for f in input_files:
                src = Path(f)
                if not src.exists():
                    return self._result(False, error=f"文件不存在: {f}")

                dst = self.input_dir / src.name
                dst_stem = dst.stem  # 用于 skip 检查

                # skip 策略：如果文件已存在，跳过
                if overwrite == "skip" and dst_stem in existing_files:
                    logger.debug("跳过已存在文件: %s", dst.name)
                    continue

                if dst.exists():
                    if overwrite == "error":
                        return self._result(
                            False,
                            error=f"文件已存在且 overwrite='error': {dst.name}"
                        )

                shutil.copy2(src, dst)
                files_processed += 1
                logger.debug("复制文件: %s → %s", src.name, dst.name)

        # ── 4. 索引完整性检查（优先检查，避免不必要的文件检查） ──
        if not force_reindex and self._is_index_valid():
            stats = self._get_index_stats()
            logger.info("索引已存在且完整，跳过构建。使用 force_reindex=true 强制重建")
            return self._result(
                True,
                message=f"索引已存在（{stats['entity_count']} 实体, "
                        f"{stats['relationship_count']} 关系, "
                        f"{stats['community_count']} 社区）",
                index_fresh=False,
                files_processed=files_processed,
                **stats,
            )

        # ── 5. 再次确认 input 目录有文件 ──────────────────────
        txt_files = list(self.input_dir.glob("*.txt"))
        if not txt_files:
            return self._result(False, error="写入/复制后 input 目录仍为空，无法构建索引")

        logger.info("input 目录共 %d 个 .txt 文件待索引", len(txt_files))

        # ── 6. 执行 graphrag index CLI ────────────────────────
        logger.info("开始执行 graphrag index (timeout=%ds)...", timeout)
        try:
            result = subprocess.run(
                ["graphrag", "index", "--root", str(self.graphrag_root)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    **os.environ,
                    "GRAPHRAG_API_KEY": os.getenv(
                        "GRAPHRAG_API_KEY",
                        os.getenv("OPENAI_API_KEY", ""),
                    ),
                },
            )

            # 输出 CLI 日志（最后 20 行），方便调试
            stderr_tail = result.stderr.strip().split("\n")[-20:]
            for line in stderr_tail:
                logger.debug("[graphrag CLI] %s", line)

            if result.returncode != 0:
                return self._result(
                    False,
                    error=f"索引失败 (exit={result.returncode}): "
                          f"{result.stderr[-500:]}",
                    files_processed=files_processed,
                )

        except subprocess.TimeoutExpired:
            return self._result(
                False,
                error=f"索引超时（>{timeout}s），请减少文档量或增大 timeout 参数",
                files_processed=files_processed,
            )
        except FileNotFoundError:
            return self._result(
                False,
                error="未找到 graphrag 命令。请先执行: pip install graphrag && graphrag init --root ./graphrag_data",
                files_processed=files_processed,
            )

        # ── 7. 写入 _SUCCESS 标记 ─────────────────────────────
        self._success_marker.write_text("ok")
        logger.info("索引构建完成，已写入 _SUCCESS 标记")

        # ── 8. 清理临时文件 ──────────────────────────────────
        if cleanup_temp and temp_files_written:
            for fp in temp_files_written:
                try:
                    fp.unlink()
                    logger.debug("已清理临时文件: %s", fp.name)
                except Exception as e:
                    logger.warning("清理临时文件失败: %s (%s)", fp.name, e)

        # ── 9. 统计并返回 ─────────────────────────────────────
        stats = self._get_index_stats()
        logger.info(
            "索引统计: %d 实体, %d 关系, %d 社区",
            stats["entity_count"], stats["relationship_count"], stats["community_count"],
        )

        return self._result(
            True,
            message=f"索引完成：{stats['entity_count']} 实体, "
                    f"{stats['relationship_count']} 关系, "
                    f"{stats['community_count']} 社区",
            index_fresh=True,
            files_processed=files_processed,
            **stats,
        )

    # ============ 内部方法 ============

    def _is_index_valid(self) -> bool:
        """
        检查索引是否完整。

        规则：
          1. _SUCCESS 标记文件存在
          2. 三个核心 parquet 文件都存在且非空
        """
        if not self._success_marker.exists():
            return False

        required = ["entities.parquet", "relationships.parquet", "communities.parquet"]
        for fname in required:
            fp = self.artifacts_dir / fname
            if not fp.exists() or fp.stat().st_size == 0:
                logger.debug("索引不完整: 缺少或为空 %s", fname)
                return False
        return True

    def _get_index_stats(self) -> dict:
        """
        用 pyarrow 读取 Parquet metadata 获取行数。

        逐个文件 try/except，部分缺失也能返回已有数据。
        """
        stats = {"entity_count": 0, "relationship_count": 0, "community_count": 0}
        mapping = [
            ("entity_count", "entities.parquet"),
            ("relationship_count", "relationships.parquet"),
            ("community_count", "communities.parquet"),
        ]
        for key, filename in mapping:
            fp = self.artifacts_dir / filename
            if fp.exists():
                try:
                    stats[key] = _get_parquet_row_count(fp)
                except Exception as e:
                    logger.warning("读取 %s 失败: %s，计为 0", filename, e)
                    stats[key] = 0
        return stats

    def _result(
        self,
        success: bool,
        message: str = "",
        error: Optional[str] = None,
        index_fresh: bool = False,
        files_processed: int = 0,
        entity_count: int = 0,
        relationship_count: int = 0,
        community_count: int = 0,
    ) -> Dict[str, Any]:
        """构建统一的返回结构"""
        result = {
            "success": success,
            "message": message,
            "output_dir": str(self.artifacts_dir),
            "entity_count": entity_count,
            "relationship_count": relationship_count,
            "community_count": community_count,
            "error": error,
            "index_fresh": index_fresh,
            "files_processed": files_processed,
        }
        # 如果安装了 Pydantic，做输出校验
        if HAS_PYDANTIC and self.output_schema:
            try:
                validated = self.output_schema(**result)
                return validated.model_dump()
            except Exception as e:
                logger.warning("输出校验失败: %s", e)
        return result
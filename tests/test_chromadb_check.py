"""
检查 ChromaDB 向量存储状态（诊断工具脚本）
支持：python -m tests.test_chromadb_check  或  pytest tests/test_chromadb_check.py
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（支持独立运行）
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.infrastructure.vector_store import VectorStoreManager


def check_chromadb():
    vs = VectorStoreManager.default()
    try:
        collections = vs.client.list_collections()
        print(f"现有集合: {[c.name for c in collections]}")
        if any(c.name == "default" for c in collections):
            collection = vs.client.get_collection("default")
            count = collection.count()
            print(f"'default' 集合文档数: {count}")
        else:
            print("'default' 集合不存在")
    except Exception as e:
        print(f"检查失败: {e}")


def test_chromadb_runs():
    """pytest 入口：确保不抛异常"""
    check_chromadb()


if __name__ == "__main__":
    check_chromadb()

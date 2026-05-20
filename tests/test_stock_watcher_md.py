"""
快速测试 StockWatcherSkill 的 Markdown 报告输出（诊断工具脚本）
支持：python -m tests.test_stock_watcher_md  或  pytest tests/test_stock_watcher_md.py
"""
import sys
import io
from pathlib import Path

# 确保项目根目录在 sys.path 中（支持独立运行）
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.skills.custom.finance_skills.stock_watcher.skill import StockWatcherSkill


def generate_report():
    s = StockWatcherSkill()
    data = s.execute()
    md = s.build_markdown_report(data)
    return md


def test_report_generates():
    """pytest 入口：确保报告能正常生成"""
    md = generate_report()
    assert md, "Markdown 报告不应为空"
    assert len(md) > 50, "报告内容过短，可能有异常"


if __name__ == "__main__":
    md = generate_report()
    print(md)

"""投研日报路由 — StockWatcherSkill 的 REST 接口"""
from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, Depends

from src.api.auth import verify_api_key
from src.skills.custom.finance_skills.stock_watcher.skill import StockWatcherSkill

logger = logging.getLogger(__name__)

router = APIRouter(tags=["投研日报"])

# 技能单例，避免每次请求重新初始化
_skill: StockWatcherSkill | None = None


def _get_skill() -> StockWatcherSkill:
    global _skill
    if _skill is None:
        _skill = StockWatcherSkill()
    return _skill


@router.get("/stock-report")
async def get_stock_report(_: str = Depends(verify_api_key)):
    """
    生成投研日报（JSON）
    
    返回核心持仓行情、大宗商品价格、观察池异动、预警信号
    """
    skill = _get_skill()
    data = skill.execute()
    return data


@router.get("/stock-report/markdown")
async def get_stock_report_markdown(_: str = Depends(verify_api_key)):
    """
    生成投研日报（Markdown 格式）
    
    可直接在前端渲染为富文本日报
    """
    skill = _get_skill()
    data = skill.execute()
    return {
        "date": data.get("date", str(date.today())),
        "markdown": skill.build_markdown_report(data),
        "json": data,
    }

"""技术雷达路由 — TechnologyRadarSkill 的 REST 接口"""
from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, Depends, Query

from src.api.auth import verify_api_key
from src.skills.custom.finance_skills.technology_radar.skill import TechnologyRadarSkill

logger = logging.getLogger(__name__)

router = APIRouter(tags=["技术雷达"])

_skill: TechnologyRadarSkill | None = None


def _get_skill() -> TechnologyRadarSkill:
    global _skill
    if _skill is None:
        _skill = TechnologyRadarSkill()
    return _skill


@router.get("/tech-radar")
async def get_tech_radar(
    area_ids: str | None = Query(None, description="按技术ID筛选，逗号分隔，如 energy_solid_state,energy_perovskite"),
    _: str = Depends(verify_api_key),
):
    """
    生成技术雷达扫描报告（JSON）

    扫描14个核心技术领域的最新突破新闻，
    自动映射到收益股票和用户持仓，返回结构化数据
    """
    skill = _get_skill()
    data = skill.execute(area_ids=area_ids)
    return data


@router.get("/tech-radar/markdown")
async def get_tech_radar_markdown(
    area_ids: str | None = Query(None, description="按技术ID筛选"),
    _: str = Depends(verify_api_key),
):
    """
    生成技术雷达日报（Markdown 格式）

    可直接在前端渲染为富文本技术雷达日报
    """
    skill = _get_skill()
    data = skill.execute(area_ids=area_ids)
    return {
        "date": data.get("date", str(date.today())),
        "markdown": skill.build_markdown_report(data),
        "json": data,
    }

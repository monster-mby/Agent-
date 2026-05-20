"""
TechnologyRadarSkill v1 — 硬核技术突破追踪雷达

采集流程：
  1. 14个技术领域×多个关键词×东方财富+百度搜索
  2. 持仓个股新闻采集
  3. 相关性评分（含等级评定）
  4. 持仓催化影响交叉分析
  5. 生成详细Markdown报告
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from typing import Any, Dict, List, Tuple

import requests

from src.skills.base.base_skill import BaseSkill
from src.skills.custom.finance_skills.technology_radar.config import (
    HOLDINGS, TECH_RADAR_CONFIG, TECH_ID_MAP,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════
#  配置
# ═══════════════════════════════════════
REQUEST_TIMEOUT = 15
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/json,*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
_SESSION = requests.Session()
_SESSION.headers.update(HEADERS)


# ═══════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════

def _http_get(url: str, timeout: int = REQUEST_TIMEOUT, encoding: str = "utf-8") -> str | None:
    try:
        r = _SESSION.get(url, timeout=timeout)
        r.encoding = encoding
        return r.text
    except Exception as e:
        logger.debug(f"HTTP GET {url[:80]}: {e}")
        return None

def _clean_html(t: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', t)).strip()

def _today() -> str: return str(date.today())
def _now() -> str: return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ═══════════════════════════════════════
#  新闻采集器
# ═══════════════════════════════════════

class NewsFetcher:
    """多源新闻采集"""

    @staticmethod
    def eastmoney_keyword(keyword: str, limit: int = 5) -> List[Dict]:
        results = []
        try:
            text = _http_get(
                f"https://search-api-web.eastmoney.com/search/jsonp?cb=jQuery&param=%7B%22uid%22%3A%22%22%2C%22keyword%22%3A%22{keyword}%22%2C%22type%22%3A%5B%22cmsArticleWebOld%22%5D%2C%22client%22%3A%22web%22%2C%22clientType%22%3A%22web%22%2C%22clientVersion%22%3A%22curr%22%2C%22param%22%3A%7B%22cmsArticleWebOld%22%3A%7B%22searchScope%22%3A%22default%22%2C%22sort%22%3A%22default%22%2C%22pageIndex%22%3A1%2C%22pageSize%22%3A{limit}%2C%22preTag%22%3A%22%22%2C%22postTag%22%3A%22%22%7D%7D%7D",
                timeout=10)
            if text:
                m = re.search(r'jQuery\([^)]+\)\(({.*})\)', text, re.DOTALL)
                if m:
                    data = json.loads(m.group(1))
                    arts = (data.get("result",{}).get("cmsArticleWebOld",{}).get("list",[])
                            or data.get("result",{}).get("articles",[]))
                    for a in arts[:limit]:
                        results.append({"title": a.get("title",a.get("Title","")),
                                        "summary": _clean_html(a.get("summary",a.get("Summary",""))),
                                        "time": str(a.get("date",a.get("Date","")))[:10],
                                        "source":"东方财富","url":a.get("url",a.get("Url",""))})
        except Exception:
            pass
        return results

    @staticmethod
    def baidu_news(keyword: str, limit: int = 5) -> List[Dict]:
        results = []
        try:
            text = _http_get(f"https://news.baidu.com/ns?word={keyword}&pn=0&rn={limit}&cl=2&ct=1&tn=newstitle&ie=utf-8", timeout=10)
            if text:
                for url_m, title_raw in re.findall(r'<h3[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',text,re.DOTALL)[:limit]:
                    t = _clean_html(title_raw)
                    if t:
                        results.append({"title":t,"summary":"","time":"","source":"百度新闻","url":url_m})
        except Exception:
            pass
        return results

    @staticmethod
    def stock_news(code: str, limit: int = 3) -> List[Dict]:
        results = []
        for url in [f"https://np-announcement.eastmoney.com/api/content/announcement?pageIndex=1&pageSize={limit}&stock={code}",
                    f"https://np-news.eastmoney.com/api/content/news?pageIndex=1&pageSize={limit}&stock={code}"]:
            try:
                r = _SESSION.get(url, timeout=8)
                if r.status_code == 200:
                    for item in r.json().get("data",{}).get("list",[]):
                        results.append({"title":item.get("title",item.get("Title","")),
                                        "summary":"","time":str(item.get("notice_date",item.get("NoticeDate","")))[:10],
                                        "source":"东方财富·个股","url":item.get("url","")})
                    if results: break
            except Exception:
                continue
        return results[:limit]


# ═══════════════════════════════════════
#  相关性评分引擎
# ═══════════════════════════════════════

STRONG_SIGNALS = ["突破","量产","商用","落地","首发","成功","重大进展","里程碑",
                  "全球首","获批","取证","临床","入轨","效率突破","能量密度",
                  "正式发布","标准发布","试点","上线","交付","批量"]
MEDIUM_SIGNALS = ["加速","推进","投资","研发","测试","实验","迭代","升级",
                  "融资","合作","签约","规划","目标","计划","布局"]

def score_news(title: str, summary: str, area_kws: List[str], progress: int) -> Tuple[int, str, str]:
    """返回 (分数 0-100, 等级:high/medium/low, 标签)"""
    text = (title + " " + summary).lower()
    s = sum(10 for kw in area_kws if kw.lower() in text)
    s += sum(15 for sig in STRONG_SIGNALS if sig in text)
    s += sum(5 for sig in MEDIUM_SIGNALS if sig in text)
    s += (progress // 10) * 2
    if any(kw in title.lower() for kw in area_kws if len(kw) > 3): s += 15
    s = min(s, 100)
    lv = "high" if s >= 30 else ("medium" if s >= 15 else "low")
    labels = {"high":"重大突破","medium":"进展跟踪","low":"常规资讯"}
    emojis = {"high":"🔴","medium":"🟡","low":"🟢"}
    return s, lv, f"{emojis[lv]} {labels[lv]}"


# ═══════════════════════════════════════
#  TechnologyRadarSkill
# ═══════════════════════════════════════

class TechnologyRadarSkill(BaseSkill):
    name = "technology_radar"
    description = "14个硬核技术方向全网突破追踪，自动映射受益股票，生成技术雷达日报+持仓催化热力图"
    version = "1.0.0"
    author = "monster"
    triggers = ["技术雷达","技术突破","科技催化","硬核技术","前沿科技","technology_radar"]
    input_schema = None

    def execute(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
        """全量扫描入口。area_ids可选，逗号分隔。"""
        area_filter = kwargs.get("area_ids")
        errors = []
        configs = TECH_RADAR_CONFIG
        if area_filter:
            allowed = set(a.strip() for a in area_filter.split(","))
            configs = [t for t in configs if t["id"] in allowed]

        # ── 并发采集 ──
        raw_news: Dict[str, List[Dict]] = {t["id"]: [] for t in configs}
        holding_news: Dict[str, List[Dict]] = {}
        futures = {}

        with ThreadPoolExecutor(max_workers=20) as pool:
            for tc in configs:
                for kw in tc["keywords"][:3]:
                    td_id = tc["id"]
                    futures[f"kw_{td_id}_{kw}"] = pool.submit(
                        self._multi_search, kw, 3)

            for h in HOLDINGS:
                futures[f"stock_{h['code']}"] = pool.submit(
                    NewsFetcher.stock_news, h["code"], 2)

            # 收集
            for key, fut in futures.items():
                try:
                    result = fut.result()
                    if key.startswith("kw_"):
                        _, td_id = key.split("_", 2)[:2]
                        # td_id might contain underscores from keywords
                        # fix: match td_id properly
                        raw_key = key[3:]  # strip "kw_"
                        for tid in raw_news:
                            if key.startswith(f"kw_{tid}_"):
                                raw_news[tid].extend(result)
                                break
                        else:
                            # fallback: try to find matching td_id
                            for tid in raw_news:
                                if tid in key:
                                    raw_news[tid].extend(result)
                                    break
                    elif key.startswith("stock_"):
                        code = key[6:]
                        holding_news[code] = result or []
                except Exception as e:
                    errors.append(f"{key}:{e}")

        # ── 去重+评分 ──
        tech_results = {}
        for tc in configs:
            tid = tc["id"]
            seen = set()
            scored = []
            for n in raw_news.get(tid, []):
                dedup_key = n.get("title","")[:40]
                if dedup_key in seen: continue
                seen.add(dedup_key)
                s, lv, label = score_news(n["title"], n["summary"], tc["keywords"], tc["progress_pct"])
                scored.append({**n, "score": s, "level": lv, "level_label": label})
            scored.sort(key=lambda x: x["score"], reverse=True)

            counts = {"high":0,"medium":0,"low":0}
            for n in scored: counts[n["level"]] = counts.get(n["level"],0)+1

            tech_results[tid] = {
                "config": {
                    "id": tc["id"], "area": tc["area"], "tech_name": tc["tech_name"],
                    "breakthrough_year": tc["breakthrough_year"],
                    "progress_pct": tc["progress_pct"], "progress_desc": tc["progress_desc"],
                    "principle": tc["principle"], "bottleneck": tc["bottleneck"], "rnd_orgs": tc["rnd_orgs"],
                    "stocks": tc["stocks"], "your_holdings_mapped": tc["your_holdings_mapped"],
                    "keywords": tc["keywords"],
                },
                "total_news": len(scored),
                "counts": counts,
                "high_news": [n for n in scored if n["level"]=="high"][:4],
                "medium_news": [n for n in scored if n["level"]=="medium"][:6],
                "all_news": scored[:20],
            }

        # ── 持仓催化分析 ──
        impact = self._calc_holding_impact(configs, tech_results)
        total_high = sum(r["counts"]["high"] for r in tech_results.values())
        total_medium = sum(r["counts"]["medium"] for r in tech_results.values())
        total_low = sum(r["counts"]["low"] for r in tech_results.values())

        return {
            "date": _today(),
            "generated_at": _now(),
            "summary": self._summary_line(configs, impact, total_high, total_medium),
            "total_news_all": total_high+total_medium+total_low,
            "stats": {"high":total_high, "medium":total_medium, "low":total_low},
            "technologies": tech_results,
            "holding_impact": impact,
            "holding_news": {h["name"]: holding_news.get(h["code"],[]) for h in HOLDINGS},
            "holding_stocks": HOLDINGS,
            "errors": errors,
        }

    def _multi_search(self, keyword: str, limit: int = 3) -> List[Dict]:
        r = []
        r.extend(NewsFetcher.eastmoney_keyword(keyword, limit))
        r.extend(NewsFetcher.baidu_news(keyword, limit))
        return r

    def _calc_holding_impact(self, configs: List[Dict], results: Dict) -> List[Dict]:
        """计算每个持仓受哪些技术催化"""
        impact_map = {}
        for h in HOLDINGS:
            impact_map[h["code"]] = {
                "code": h["code"], "name": h["name"], "tag": h["tag"],
                "tech_count": 0, "high_techs": [], "total_impact_score": 0,
                "tech_details": [],
            }
        for tc in configs:
            tid = tc["id"]
            r = results.get(tid, {})
            for hm in tc["your_holdings_mapped"]:
                # hm 格式如 "长电科技(600584)"
                code_match = hm.split("(")[-1].rstrip(")")
                for hold in HOLDINGS:
                    if hold["code"] == code_match or hold["name"] in hm:
                        cnt = impact_map[hold["code"]]
                        cnt["tech_count"] += 1
                        cnt["total_impact_score"] += sum(n["score"] for n in r.get("high_news",[]))
                        has_high = len(r.get("high_news",[])) > 0
                        cnt["high_techs"].append({
                            "tech_id": tid,
                            "tech_name": tc["tech_name"],
                            "area": tc["area"],
                            "has_high_news": has_high,
                            "high_count": r.get("counts",{}).get("high",0),
                            "progress_pct": tc["progress_pct"],
                            "breakthrough_year": tc["breakthrough_year"],
                        })
                        cnt["tech_details"].append({
                            "area": tc["area"],
                            "tech_name": tc["tech_name"],
                            "progress_pct": tc["progress_pct"],
                            "breakthrough_year": tc["breakthrough_year"],
                        })
                        break
        result_list = sorted(impact_map.values(), key=lambda x: x["total_impact_score"], reverse=True)
        for r in result_list:
            r["impact_level"] = self._impact_level(r["total_impact_score"])
        return result_list

    def _impact_level(self, score: int) -> str:
        if score >= 50: return "🔴 高催化"
        if score >= 20: return "🟡 中催化"
        if score > 0: return "🟢 低催化"
        return "⚪ 暂无催化"

    def _summary_line(self, configs: List[Dict], impact: List[Dict], high: int, med: int) -> str:
        active_holdings = [h for h in impact if h["tech_count"] > 0]
        hold_part = "、".join(f"{h['name']}({h['tech_count']}项技术)" for h in active_holdings[:5]) if active_holdings else "暂无"
        return f"今日扫描{len(configs)}个技术领域 | 🔴重大突破{high}条 🟡进展跟踪{med}条 | 持仓催化: {hold_part}"

    # ═══════════════════════════════════════
    #  Markdown 报告生成
    # ═══════════════════════════════════════

    def build_markdown_report(self, data: Dict) -> str:
        lines = [f"# 🔬 技术雷达日报 — {data.get('date','')}", "",
                 f"*生成时间：{data.get('generated_at','')}*", "",
                 f"> 📡 {data.get('summary','')}", "",
                 f"**数据统计：** 🔴重大突破 {data.get('stats',{}).get('high',0)} 条 "
                 f"| 🟡进展跟踪 {data.get('stats',{}).get('medium',0)} 条 "
                 f"| 🟢常规资讯 {data.get('stats',{}).get('low',0)} 条", ""]

        # ── 持仓催化热力图 ──
        lines.extend(["---", "## 📊 持仓×技术催化热力图", ""])
        lines.append("| 持仓 | 状态 | 关联技术数 | 催化等级 | 受益技术 |")
        lines.append("|------|------|-----------|---------|---------|")
        for h in data.get("holding_impact", []):
            techs = "、".join(t["tech_name"].split("：")[0] for t in h.get("high_techs",[])) or "暂无"
            lines.append(f"| {h['name']} | {h['tag']} | {h['tech_count']} | {h.get('impact_level','')} | {techs} |")
        lines.append("")

        # ── 持仓个股最新新闻 ──
        h_news = data.get("holding_news", {})
        if any(v for v in h_news.values()):
            lines.extend(["---", "## 📰 持仓个股最新消息", ""])
            for h in data.get("holding_stocks", []):
                ns = h_news.get(h["name"], [])
                if ns:
                    lines.append(f"**{h['name']}（{h['tag']}）：**")
                    for n in ns[:3]:
                        lines.append(f"- [{n.get('time','')}] {n.get('title','')}")
                    lines.append("")

        # ── 各技术领域详情 ──
        lines.extend(["---", "## 🧬 核心技术领域追踪", ""])
        techs = data.get("technologies", {})
        for tid, r in techs.items():
            cfg = r.get("config", {})
            area = cfg.get("area", tid)
            tech_name = cfg.get("tech_name", "")
            counts = r.get("counts", {})
            high_n = counts.get("high", 0)
            med_n = counts.get("medium", 0)

            # 进度条
            prog = cfg.get("progress_pct", 0)
            bar = "█" * (prog // 10) + "░" * (10 - prog // 10)
            yr = cfg.get("breakthrough_year", "未知")

            lines.extend([
                f"### {area}：{tech_name}", "",
                f"**预计突破：** {yr}　**成熟度：** {bar} {prog}%",
                f"**今日消息：** 🔴{high_n} / 🟡{med_n} 条",
                f"**底层原理：** {cfg.get('principle','')}",
                f"**当前瓶颈：** {cfg.get('bottleneck','')}",
                f"**研发主体：** {cfg.get('rnd_orgs','')}",
                "",
            ])

            # 收益股票
            stocks = cfg.get("stocks", [])
            if stocks:
                lines.append("**受益公司：**")
                for s in stocks:
                    mark = " ⭐" if any(s["code"] in h.get("code","") or s["name"] in h.get("name","")
                                       for h in data.get("holding_stocks",[])) else ""
                    lines.append(f"- {s['name']}({s['code']})—{s['tag']}{mark}")
                lines.append("")

            # 你的持仓映射
            holds = cfg.get("your_holdings_mapped", [])
            if holds:
                lines.append(f"**→ 你的持仓关联：** {', '.join(holds)}")
                lines.append("")

            # 重大突破新闻
            high_news = r.get("high_news", [])
            if high_news:
                lines.append("**🔴 重大突破：**")
                for n in high_news:
                    src = n.get("source","")
                    lines.append(f"- [{n.get('time','')}] {n.get('title','')}（{src}）")
                lines.append("")

            # 进展跟踪新闻
            med_news = r.get("medium_news", [])
            if med_news:
                lines.append("**🟡 进展跟踪：**")
                for n in med_news[:4]:
                    src = n.get("source","")
                    lines.append(f"- [{n.get('time','')}] {n.get('title','')}（{src}）")
                lines.append("")

            lines.append("---")
            lines.append("")

        # ── 年度进度总览 ──
        lines.extend(["## 🎯 年度技术突破进度总览", ""])
        lines.append("| 技术领域 | 突破年份 | 进度 | 今日动态 |")
        lines.append("|---------|---------|------|---------|")
        for tid, r in sorted(techs.items()):
            cfg = r.get("config", {})
            prog = cfg.get("progress_pct", 0)
            bar = "█" * (prog // 10) + "░" * (10 - prog // 10)
            counts = r.get("counts", {})
            status = f"🔴{counts.get('high',0)} 🟡{counts.get('medium',0)}" if counts.get("high",0) else f"🟢{counts.get('low',0)}" if counts.get("medium",0) else "⚪无消息"
            lines.append(f"| {cfg.get('area','')} | {cfg.get('breakthrough_year','')} | {bar} {prog}% | {status} |")
        lines.append("")

        # ── 附录 ──
        lines.extend([
            "---", "",
            "### 💡 使用说明",
            "",
            "- 每日自动扫描 14 个核心技术方向",
            "- 🔴 = 强信号匹配（含突破、量产、商业化等关键词）",
            "- 🟡 = 中等信号匹配（含研发、推进、合作等关键词）",
            "- 🟢 = 常规资讯",
            "- ⭐ = 此技术直接关联你的持仓",
            "",
            "---",
            f"*报告生成：{_now()}*",
            "",
            "> ⚠️ 数据来源：东方财富、百度新闻。AI自动评分，仅供参考，不构成投资建议。",
        ])

        return "\n".join(lines)

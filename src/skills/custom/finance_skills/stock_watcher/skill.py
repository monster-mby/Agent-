"""
StockWatcherSkill v2 — 详细投研日报
实时行情 + 24h新闻 + 机构评级 + 大宗商品趋势 + 观察池异动 + 预警信号
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import requests
from pydantic import BaseModel, Field

from src.skills.base.base_skill import BaseSkill

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════
REQUEST_TIMEOUT = 12
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://finance.sina.com.cn",
}
_SESSION = requests.Session()
_SESSION.headers.update(HEADERS)

CORE_STOCKS = [
    {"code": "601899", "name": "紫金矿业", "tag": "资源周期",
     "key_metrics": "金价/铜价/碳酸锂", "watch_items": ["金价>3000", "铜价>9500", "碳酸锂企稳"]},
    {"code": "600089", "name": "特变电工", "tag": "电网基建",
     "key_metrics": "国网招标/硅料价格/海外订单", "watch_items": ["迎峰度夏催化", "硅料筑底", "可转债进度"]},
    {"code": "603596", "name": "伯特利",   "tag": "智能底盘",
     "key_metrics": "毛利率/豫北交割/L3法规", "watch_items": ["毛利率持续改善", "豫北转向并表", "新项目定点"]},
]

# 重点观察（买不起 & 危险但可赌）
EXTENDED_STOCKS = [
    {"code": "301018", "name": "申菱环境", "tag": "AI液冷温控", "pool": "买不起想买",
     "key_metrics": "液冷订单/数据中心资本开支", "watch_items": ["Q2利润降幅收窄", "液冷出海进展", "回调到100以下再看"]},
    {"code": "688981", "name": "中芯国际", "tag": "半导体制造", "pool": "买不起想买",
     "key_metrics": "先进制程良率/产能利用率/ASP", "watch_items": ["N+2突破", "成熟制程价格企稳", "回调到90-100区间"]},
    {"code": "600584", "name": "长电科技", "tag": "先进封装", "pool": "危险但可赌",
     "key_metrics": "Chiplet订单/AMD出货量/大基金减持", "watch_items": ["大基金减持消化完毕", "先进封装产能利用率", "Q2利润增速"]},
]

WATCH_POOL = [
    {"code": "688295", "name": "中复神鹰", "pool": "等回调", "price_low": 50, "price_high": 55},
    {"code": "601126", "name": "四方股份", "pool": "等回调", "price_low": 55, "price_high": 60},
    {"code": "300699", "name": "光威复材", "pool": "埋伏",   "price_low": None, "price_high": None},
    {"code": "000960", "name": "锡业股份", "pool": "埋伏",   "price_low": 35,  "price_high": 38},
    {"code": "600111", "name": "北方稀土", "pool": "埋伏",   "price_low": 35,  "price_high": 40},
    {"code": "603259", "name": "药明康德", "pool": "买不起", "price_low": 90,  "price_high": 100},
    {"code": "002837", "name": "英维克",   "pool": "关注",   "price_low": None, "price_high": None},
    {"code": "002335", "name": "科华数据", "pool": "关注",   "price_low": None, "price_high": None},
]


# ═══════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════
def _sf(v: Any) -> float:
    try: return float(v)
    except: return 0.0

def _sina_symbol(code: str) -> str:
    return f"sh{code}" if code.startswith(("6","9")) else f"sz{code}"

def _http_get(url: str, timeout: int = REQUEST_TIMEOUT, encoding: str = "gbk") -> Optional[str]:
    try:
        r = _SESSION.get(url, timeout=timeout)
        r.encoding = encoding
        return r.text
    except Exception as e:
        logger.debug(f"HTTP GET {url[:60]}: {e}")
        return None

# ═══════════════════════════════════════════
#  StockWatcherSkill
# ═══════════════════════════════════════════
class StockWatcherSkill(BaseSkill):
    name = "stock_watcher"
    description = "拉取实时行情、24h新闻、机构评级、大宗商品趋势、观察池异动，生成详细投研日报"
    version = "2.0.0"
    author = "monster"
    triggers = ["投研日报", "持仓检查", "股票预警", "今日行情", "stock_watcher"]
    input_schema = None

    ALERT_GOLD_LOW   = 2800
    ALERT_COPPER_LOW = 9000

    def execute(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
        today = str(date.today())
        errors = []

        # ── 所有股票列表：核心 + 重点观察 + 观察池 ──
        all_stocks = CORE_STOCKS + EXTENDED_STOCKS + WATCH_POOL

        # ── 并发获取数据 ──
        with ThreadPoolExecutor(max_workers=10) as pool:
            f_sina  = pool.submit(self._fetch_sina_batch, all_stocks)
            f_news  = pool.submit(self._fetch_news_batch)
            f_enews = pool.submit(self._fetch_extended_news)
            f_index = pool.submit(self._fetch_industry_news)
            f_gold  = pool.submit(self._fetch_commodity_sina, "AU0", "沪金", "元/克")
            f_cu    = pool.submit(self._fetch_commodity_sina, "CU0", "沪铜", "元/吨")
            f_lc    = pool.submit(self._fetch_commodity_sina, "LC0", "碳酸锂", "元/吨")
            f_si    = pool.submit(self._fetch_commodity_sina, "SI0", "多晶硅", "元/吨")
            f_usd   = pool.submit(self._fetch_usd_index)

            sina_data  = f_sina.result()  or {}
            news_data  = f_news.result()  or {}
            extended_news = f_enews.result() or {}
            industry   = f_index.result() or {}
            commodities = {
                "gold": f_gold.result() or {"name":"沪金","price":"—","change_pct":"—","unit":"元/克"},
                "copper": f_cu.result() or {"name":"沪铜","price":"—","change_pct":"—","unit":"元/吨"},
                "lithium": f_lc.result() or {"name":"碳酸锂","price":"—","change_pct":"—","unit":"元/吨"},
                "polysilicon": f_si.result() or {"name":"多晶硅","price":"—","change_pct":"—","unit":"元/吨"},
                "usd_index": f_usd.result() or {"name":"美元指数","price":"—","change_pct":"—","unit":""},
            }

        # ── 组装结果 ──
        core_detail = self._build_core_detail(CORE_STOCKS, sina_data, news_data)
        extended_detail = self._build_core_detail(EXTENDED_STOCKS, sina_data, extended_news)
        watch_detail = self._build_watch_detail(WATCH_POOL, sina_data)
        alerts = self._build_alerts(core_detail, commodities)
        summary = self._build_summary(core_detail, watch_detail, alerts)

        return {
            "date": today,
            "core_holdings": core_detail,
            "extended_stocks": extended_detail,
            "commodities": commodities,
            "watch_pool": watch_detail,
            "industry_news": industry,
            "alerts": alerts,
            "summary": summary,
            "errors": errors,
        }

    # ═══════════════════════════════════════════
    #  行情（新浪）
    # ═══════════════════════════════════════════
    def _fetch_sina_batch(self, stock_list: list) -> Dict[str, Dict]:
        all_codes = [s["code"] for s in stock_list]
        symbols = [_sina_symbol(c) for c in all_codes]
        url = "http://hq.sinajs.cn/list=" + ",".join(symbols)
        text = _http_get(url)
        if not text: return {}
        data = {}
        for line in text.strip().split("\n"):
            if "hq_str_" not in line or "=" not in line:
                continue
            try:
                sym = line.split("=")[0].replace("var hq_str_","").strip()
                code = sym[2:]
                val = line.split("=",1)[1].strip().strip('";')
                fields = val.split(",") if val else []
                if len(fields) < 10: continue
                prev = _sf(fields[2])
                price = _sf(fields[3])
                data[code] = {
                    "code": code, "name": fields[0],
                    "open": _sf(fields[1]), "prev_close": prev, "price": price,
                    "high": _sf(fields[4]), "low": _sf(fields[5]),
                    "volume": _sf(fields[8]), "turnover": _sf(fields[9]),
                    "change_pct": round((price-prev)/prev*100,2) if prev > 0 else 0,
                }
            except Exception: continue
        return data

    # ═══════════════════════════════════════════
    #  大宗商品（新浪期货）
    # ═══════════════════════════════════════════
    def _fetch_commodity_sina(self, symbol: str, name: str, unit: str) -> Dict:
        text = _http_get(f"http://hq.sinajs.cn/list=nf_{symbol}")
        if not text: return {"name": name, "price": "获取失败", "change_pct": "—", "unit": unit}
        try:
            val = text.split("=",1)[1].strip().strip('";')
            fields = val.split(",")
            price = _sf(fields[8]) if len(fields) > 8 and fields[8] else _sf(fields[3])
            prev  = _sf(fields[4]) if len(fields) > 4 else 0
            pct   = round((price-prev)/prev*100,2) if prev > 0 else 0
            trend = self._guess_trend(symbol, price, prev)
            return {
                "name": name, "symbol": symbol,
                "price": f"{price:.2f}" if price else "—",
                "change_pct": f"{pct:+.2f}%",
                "unit": unit,
                "trend_5d": trend,
                "raw_price": price,
                "raw_prev": prev,
            }
        except Exception:
            return {"name": name, "price": "获取失败", "change_pct": "—", "unit": unit, "trend_5d": "—"}

    def _guess_trend(self, symbol: str, price: float, prev: float) -> str:
        """粗略5日趋势：用 daily kline 最后5根"""
        try:
            text = _http_get(f"http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesDailyKLine?symbol={symbol}")
            if not text: return "—"
            data = json.loads(text)
            if not data: return "—"
            closes = [_sf(d[3]) for d in data[-5:]]
            if len(closes) < 5: return "—"
            if closes[-1] > closes[0]: return "↑ 近5日上行"
            if closes[-1] < closes[0]: return "↓ 近5日下行"
            return "→ 近5日横盘"
        except Exception:
            return "—"

    def _fetch_usd_index(self) -> Dict:
        """美元指数"""
        text = _http_get("http://hq.sinajs.cn/list=fx_susdindex")
        if text and "=" in text:
            try:
                val = text.split("=",1)[1].strip().strip('";')
                fields = val.split(",")
                if len(fields) >= 3 and fields[1]:
                    return {"name":"美元指数","price":fields[1],"change_pct":fields[2] if len(fields)>2 else "—","unit":""}
            except: pass
        return {"name":"美元指数","price":"获取失败","change_pct":"—","unit":""}

    # ═══════════════════════════════════════════
    #  24h 新闻（东方财富 - 公告/资讯）
    # ═══════════════════════════════════════════
    def _fetch_news_batch(self) -> Dict[str, List[Dict]]:
        result = {}
        for stock in CORE_STOCKS:
            result[stock["code"]] = self._fetch_single_stock_news(stock["code"])
        return result

    def _fetch_extended_news(self) -> Dict[str, List[Dict]]:
        """重点观察股票的新闻"""
        result = {}
        for stock in EXTENDED_STOCKS:
            result[stock["code"]] = self._fetch_single_stock_news(stock["code"])
        return result

    def _fetch_single_stock_news(self, code: str, limit: int = 5) -> List[Dict]:
        news = []
        # 东方财富个股资讯 JSON 接口
        url = f"https://np-new stock.eastmoney.com/api/content/announcement?pageIndex=1&pageSize={limit}&stock={code}"
        # 换个更可靠的接口
        urls = [
            f"https://np-announcement.eastmoney.com/api/content/announcement?pageIndex=1&pageSize={limit}&stock={code}",
            f"https://np-news.eastmoney.com/api/content/news?pageIndex=1&pageSize={limit}&stock={code}",
        ]
        for url in urls:
            try:
                r = _SESSION.get(url, timeout=8)
                if r.status_code == 200:
                    data = r.json()
                    items = data.get("data", {}).get("list", [])
                    for item in items:
                        news.append({
                            "title": item.get("title", item.get("Title","")),
                            "time": str(item.get("notice_date", item.get("NoticeDate","")))[:10],
                            "url": item.get("url", ""),
                        })
                    if news: break
            except Exception:
                continue
        return news[:limit]

    def _fetch_industry_news(self) -> Dict[str, List[str]]:
        """行业催化新闻"""
        result = {}
        keywords_map = {
            "电网": "电网",
            "新能源车": "新能源车",
            "半导体": "半导体",
            "液冷/算力": "液冷 算力",
        }
        for category, kw in keywords_map.items():
            try:
                url = f"https://np-news.eastmoney.com/api/content/news?pageIndex=1&pageSize=3&keyword={kw}"
                r = _SESSION.get(url, timeout=8)
                if r.status_code == 200:
                    items = r.json().get("data",{}).get("list",[])
                    if items:
                        result[category] = [i.get("title","") for i in items[:3]]
            except:
                pass
        return result

    # ═══════════════════════════════════════════
    #  组装详细报告
    # ═══════════════════════════════════════════
    def _build_core_detail(self, stocks: list, sina: Dict, news: Dict) -> List[Dict]:
        result = []
        for s in stocks:
            code = s["code"]
            snap = sina.get(code, {})
            price = snap.get("price", 0)
            change = snap.get("change_pct", 0)
            raw_price = snap.get("raw_price", price)

            direction = "🟢 上涨" if change > 0 else ("🔴 下跌" if change < 0 else "⚪ 平盘")
            if change > 3: direction = "🟢 大涨"
            elif change > 1: direction = "🟢 上涨"
            elif change < -3: direction = "🔴 大跌"
            elif change < -1: direction = "🔴 下跌"

            status = "✅ 正常"
            if change < -5: status = "⚠️ 关注"
            elif change < -7: status = "🔴 预警"

            # 金价映射
            gold_mapped = ""
            if code == "601899":
                gold = sina.get("gold", {})
                if gold.get("price"):
                    try:
                        au = float(gold["price"])
                        au_usd = au * 31.1 / 7.15
                        gold_mapped = f"≈{au_usd:.0f} USD/oz"
                    except: pass

            result.append({
                "code": code,
                "name": s["name"],
                "tag": s["tag"],
                "price": price,
                "change_pct": change,
                "direction": direction,
                "status": status,
                "volume": snap.get("volume", 0),
                "turnover": snap.get("turnover", 0),
                "high": snap.get("high", 0),
                "low": snap.get("low", 0),
                "open": snap.get("open", 0),
                "prev_close": snap.get("prev_close", 0),
                "key_metrics": s["key_metrics"],
                "watch_items": s["watch_items"],
                "gold_mapped": gold_mapped,
                "news_24h": news.get(code, []),
            })
        return result

    def _build_watch_detail(self, pool: list, sina: Dict) -> Dict[str, List[Dict]]:
        categories = {"等回调": [], "埋伏": [], "买不起": [], "关注": []}
        for w in pool:
            code = w["code"]
            snap = sina.get(code, {})
            price = snap.get("price", 0)
            change = snap.get("change_pct", 0)
            in_range = False
            if w["price_low"] and w["price_high"]:
                in_range = w["price_low"] <= price <= w["price_high"]

            trigger = ""
            if abs(change) >= 5:
                trigger = f"异动 {change:+.1f}%"
            if in_range:
                trigger = (trigger + "；" if trigger else "") + f"价格{price:.1f}进入关注区间[{w['price_low']}-{w['price_high']}]"

            pool_name = w.get("pool", "关注")
            categories.setdefault(pool_name, []).append({
                "code": code, "name": w["name"], "price": price, "change_pct": change,
                "trigger": trigger, "in_range": in_range,
                "target_low": w.get("price_low"), "target_high": w.get("price_high"),
            })
        return categories

    def _build_alerts(self, core: List[Dict], cm: Dict) -> List[Dict]:
        alerts = []
        # 金价
        gold = cm.get("gold",{})
        au = gold.get("raw_price", 0)
        if au:
            au_usd = au * 31.1 / 7.15
            if au_usd < self.ALERT_GOLD_LOW:
                alerts.append({"level":"red","stock":"紫金矿业",
                               "reason":f"沪金映射约{au_usd:.0f} USD/oz，低于预警线{self.ALERT_GOLD_LOW}"})
        # 铜价
        cu = cm.get("copper",{})
        cu_price = cu.get("raw_price", 0)
        if cu_price and cu_price < self.ALERT_COPPER_LOW * 7.15:
            alerts.append({"level":"red","stock":"紫金矿业",
                           "reason":f"沪铜{cu_price:.0f}元/吨，低于预警线"})
        # 伯特利大跌
        for s in core:
            if s["code"]=="603596" and s["change_pct"] < -7:
                alerts.append({"level":"yellow","stock":"伯特利",
                               "reason":f"单日大跌{s['change_pct']:.1f}%，检查公告"})
        if not alerts:
            alerts.append({"level":"green","stock":"全部",
                           "reason":"今日无预警信号触发，核心持仓逻辑未变。"})
        return alerts

    def _build_summary(self, core: List[Dict], watch: Dict, alerts: List[Dict]) -> str:
        parts = []
        for s in core:
            pct = s["change_pct"]
            d = "↑" if pct>0 else ("↓" if pct<0 else "→")
            parts.append(f"{s['name']}{d}{abs(pct):.1f}%")
        line = "｜".join(parts)

        triggered = []
        for cat_items in watch.values():
            for w in cat_items:
                if w.get("trigger"):
                    triggered.append(f"{w['name']}{w['change_pct']:+.1f}%")
        trig_line = "、".join(triggered) if triggered else "无触发"

        reds = [a for a in alerts if a["level"]=="red"]
        a_line = f"⚠️{len(reds)}条红色预警" if reds else "✅无预警"

        return f"{str(date.today())} | 持仓 {line} | 异动 {trig_line} | {a_line}"

    # ═══════════════════════════════════════════
    #  Markdown 格式化
    # ═══════════════════════════════════════════
    def build_markdown_report(self, data: Dict) -> str:
        today = data.get("date","")
        lines = [f"# 📊 投研日报 — {today}", ""]

        # ── 核心持仓 ──
        lines.append("---")
        lines.extend(["## 🔵 核心持仓", ""])
        for i, s in enumerate(data.get("core_holdings", []), 1):
            pct = s["change_pct"]
            emoji = "🟢" if pct>3 else ("🔴" if pct<-3 else ("🟡" if pct<-1 else "⚪"))
            lines.extend([
                f"### {i}. {s['name']}（{s['code']}）— {s['tag']}",
                "",
                f"| 指标 | 数据 |",
                f"|------|------|",
                f"| 最新价 | **{s['price']:.2f}** 元 |",
                f"| 涨跌幅 | {emoji} **{pct:+.2f}%** {s.get('direction','')} |",
                f"| 今开/最高/最低 | {s['open']:.2f} / {s['high']:.2f} / {s['low']:.2f} |",
                f"| 昨收 | {s['prev_close']:.2f} |",
                f"| 成交量 | {s['volume']/1e4:.0f} 万手 |",
                f"| 成交额 | {s['turnover']/1e8:.2f} 亿 |",
                f"| 状态 | {s.get('status','')} |",
                f"| 关注指标 | {s.get('key_metrics','')} |",
                "",
            ])
            # 金价映射
            if s.get("gold_mapped"):
                lines.append(f"> 📊 沪金折合 COMEX 约 **{s['gold_mapped']}**")
                lines.append("")

            # 24h新闻
            news_list = s.get("news_24h", [])
            if news_list:
                lines.append("**24h 新闻/公告：**")
                for n in news_list[:4]:
                    lines.append(f"- {n.get('time','')} {n.get('title','')}")

            # 关注项
            watch = s.get("watch_items", [])
            if watch:
                lines.append("")
                lines.append("**近期关注：**")
                for w in watch:
                    lines.append(f"- 📌 {w}")

            lines.append("")
            lines.append("---")
            lines.append("")

        # ── 重点观察 ──
        extended = data.get("extended_stocks", [])
        if extended:
            lines.extend(["## 🟠 重点观察（买不起 & 可赌）", ""])
            for i, s in enumerate(extended, 1):
                pct = s["change_pct"]
                emoji = "🟢" if pct>3 else ("🔴" if pct<-3 else ("🟡" if pct<-1 else "⚪"))
                pool_tag = s.get("tag","")
                lines.extend([
                    f"### {i}. {s['name']}（{s['code']}）— {pool_tag}",
                    "",
                    f"| 指标 | 数据 |",
                    f"|------|------|",
                    f"| 最新价 | **{s['price']:.2f}** 元 |",
                    f"| 涨跌幅 | {emoji} **{pct:+.2f}%** {s.get('direction','')} |",
                    f"| 今开/最高/最低 | {s['open']:.2f} / {s['high']:.2f} / {s['low']:.2f} |",
                    f"| 昨收 | {s['prev_close']:.2f} |",
                    f"| 成交量 | {s['volume']/1e4:.0f} 万手 |",
                    f"| 成交额 | {s['turnover']/1e8:.2f} 亿 |",
                    f"| 状态 | {s.get('status','')} |",
                    f"| 关注指标 | {s.get('key_metrics','')} |",
                    "",
                ])
                # 关注项
                watch = s.get("watch_items", [])
                if watch:
                    lines.append("**近期关注：**")
                    for w in watch:
                        lines.append(f"- 📌 {w}")
                    lines.append("")
                lines.append("---")
                lines.append("")

        # ── 大宗商品 ──
        lines.extend(["## 🟡 大宗商品", ""])
        lines.append("| 品种 | 最新价 | 涨跌幅 | 5日趋势 | 单位 |")
        lines.append("|------|--------|--------|---------|------|")
        for key in ["gold","copper","lithium","polysilicon","usd_index"]:
            item = data.get("commodities",{}).get(key,{})
            trend = item.get("trend_5d","—")
            lines.append(
                f"| {item.get('name',key)} | {item.get('price','—')} | "
                f"{item.get('change_pct','—')} | {trend} | {item.get('unit','')} |"
            )
        lines.append("")

        # ── 行业催化 ──
        ind = data.get("industry_news", {})
        if ind:
            lines.extend(["## 📰 行业催化新闻", ""])
            for cat, news in ind.items():
                if news:
                    lines.append(f"**{cat}：**")
                    for n in news:
                        lines.append(f"- {n}")
                    lines.append("")
        lines.append("")

        # ── 观察池 ──
        watch = data.get("watch_pool", {})
        lines.extend(["## 🟣 观察池", ""])
        for pool_name in ["等回调","埋伏","买不起","关注"]:
            items = watch.get(pool_name, [])
            if not items: continue
            has_trigger = any(w.get("trigger") for w in items)
            title = f"### {pool_name}" + (" ⚠️有触发" if has_trigger else "")
            lines.extend([title, ""])
            if any(w["price"]==0 for w in items if not w.get("trigger")):
                lines.append("| 代码 | 名称 | 最新价 | 涨跌幅 | 目标区间 | 状态 |")
                lines.append("|------|------|--------|--------|----------|------|")
                for w in items:
                    tgt = f"{w.get('target_low','')}-{w.get('target_high','')}" if w.get("target_low") else "—"
                    status = "⚪ 观察" if not w.get("trigger") else f"⚠️ {w['trigger']}"
                    lines.append(
                        f"| {w['code']} | {w['name']} | {w['price']:.2f} | "
                        f"{w['change_pct']:+.2f}% | {tgt} | {status} |"
                    )
            else:
                lines.append("| 代码 | 名称 | 最新价 | 涨跌幅 | 状态 |")
                lines.append("|------|------|--------|--------|------|")
                for w in items:
                    status = "⚪ 观察" if not w.get("trigger") else f"⚠️ {w['trigger']}"
                    lines.append(
                        f"| {w['code']} | {w['name']} | {w['price']:.2f} | "
                        f"{w['change_pct']:+.2f}% | {status} |"
                    )
            lines.append("")

        # ── 预警 ──
        lines.extend(["## 🔴 预警信号", ""])
        for a in data.get("alerts", []):
            icon = {"red":"🔴","yellow":"🟡","green":"🟢"}.get(a["level"],"⚪")
            lines.append(f"- {icon} **{a['stock']}**：{a['reason']}")
        lines.append("")

        # ── 摘要 ──
        lines.extend([
            "---",
            "",
            f"## 📝 一句话总结",
            "",
            f"> {data.get('summary','')}",
            "",
            "---",
            f"*报告生成：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "> ⚠️ 数据来源：新浪财经、东方财富。仅供参考，不构成投资建议。",
        ])
        return "\n".join(lines)

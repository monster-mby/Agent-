"""
Chart Advisor Skill — 图表推荐 (v2.1.0)
这是一个智能图表推荐与代码生成工具，能根据你的数据描述（和可选的 CSV 样本），自动识别你的展示意图，推荐最合适的图表类型，并直接生成对应库（matplotlib/plotly/seaborn）的可运行代码。下面从功能概览、核心结构、类 / 方法详解、调用链路、代码质量五个维度深度解析。
一、功能概览
这个工具的核心能力是：
意图识别：通过关键词（或可选语义模型）判断你想 “对比数值”“展示趋势” 还是 “看占比” 等。
数据特征分析：如果你给了 CSV 样本，自动识别行数、数值列数、分类数、是否有时间列。
智能推荐：结合意图和数据特征，给图表打分、排序，输出 top3 推荐。
反推荐警告：如果数据不适合某类图（比如饼图类别 > 6），直接警告。
代码生成：用 “参数化骨架” 替代硬编码，自动生成对应库的完整代码。
低置信度追问：如果不确定你的意图，会主动问你是 “对比 / 趋势 / 占比” 中的哪一种。
二、核心结构设计
代码采用分层架构，核心结构如下：
┌─────────────────────────────────────────────────────────────┐
│  1. 可选依赖层                                                │
│     - sentence_transformers（可选，用于语义意图识别）          │
├─────────────────────────────────────────────────────────────┤
│  2. 数据模型层（Pydantic）                                    │
│     - ChartAdvisorInput/Output（输入输出校验）                 │
│     - ChartRecommendation（单个推荐结果）                       │
├─────────────────────────────────────────────────────────────┤
│  3. 核心代码生成层                                            │
│     - CHART_SKELETONS（图表代码骨架字典）                     │
│     - _generate_code（参数化代码拼装函数）                     │
├─────────────────────────────────────────────────────────────┤
│  4. 规则与常量层                                              │
│     - INTENT_KEYWORDS/CHART_MAP（意图→图表映射）              │
│     - REASONS（推荐理由）、ANTI_RULES（反推荐规则）            │
├─────────────────────────────────────────────────────────────┤
│  5. 主技能类（ChartAdvisorSkill）                             │
│     - execute（主入口）、_detect_intent（意图识别）            │
│     - _analyze_data（特征分析）、_adjust（评分调整）            │
│     - _tips（使用建议）                                        │
└─────────────────────────────────────────────────────────────┘
ChartAdvisorSkill.execute()
  ├→ _detect_intent(desc)  [意图识别，带缓存]
  ├→ _analyze_data(sample)  [数据特征分析，仅当有样本时]
  ├→ 遍历候选图表：
  │   ├→ _adjust(score, feats, chart)  [评分调整]
  │   ├→ _tips(chart, feats)  [使用建议]
  │   └→ _generate_code(chart, library)  [代码生成]
  ├→ 生成反推荐警告（遍历ANTI_RULES）
  └→ 组装ChartAdvisorOutput
"""

from __future__ import annotations

import re
from functools import lru_cache
from io import StringIO
from typing import Any, ClassVar, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.skills.base.base_skill import BaseSkill

# ---------------------------------------------------------------------------
# 可选依赖
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    _ST_OK = False

# ---------------------------------------------------------------------------
# 作用：输入校验模型，保证输入格式合法。
# ---------------------------------------------------------------------------

class ChartAdvisorInput(BaseModel):
    data_description: str = Field(..., min_length=5)
    data_sample: Optional[str] = Field(default=None)
    library: Literal["matplotlib", "plotly", "seaborn", "any"] = "any"

# ---------------------------------------------------------------------------
# 作用：单个图表推荐的结构化结果。
# ---------------------------------------------------------------------------

class ChartRecommendation(BaseModel):
    chart_type: str
    en_name: str
    suitability_score: float
    reason: str
    code_skeleton: str
    tips: list[str] = []

# ---------------------------------------------------------------------------
# 作用：完整的结构化输出。
# ---------------------------------------------------------------------------
class ChartAdvisorOutput(BaseModel):
    user_intent: str
    confidence: float
    recommendations: list[ChartRecommendation]
    best_pick: str = ""
    warnings: list[str] = []
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


# ===========================================================================
# 作用：存储每种图表在不同库下的 “代码片段骨架”，替代 800 行硬编码模板。
# ===========================================================================

# 每种图表的"骨架参数"，不是完整代码
CHART_SKELETONS: dict[str, dict] = {
    "bar": {
        "plot_func": {"matplotlib": "ax.bar(categories, values)", "plotly": "px.bar(df, x='x', y='y')", "seaborn": "sns.barplot(data=df, x='x', y='y')"},
        "imports": {"matplotlib": "import matplotlib.pyplot as plt", "plotly": "import plotly.express as px\nimport pandas as pd", "seaborn": "import seaborn as sns\nimport matplotlib.pyplot as plt\nimport pandas as pd"},
        "data_prep": {"matplotlib": "categories = ['A','B','C','D']\nvalues = [25,40,35,20]", "plotly": "df = pd.DataFrame({'x': ['A','B','C','D'], 'y': [25,40,35,20]})", "seaborn": "df = pd.DataFrame({'x': ['A','B','C','D'], 'y': [25,40,35,20]})"},
        "setup": {"matplotlib": "fig, ax = plt.subplots(figsize=(8,5))", "plotly": "", "seaborn": "sns.set_style('whitegrid')"},
        "finish": {"matplotlib": "ax.set_xlabel('类别')\nax.set_ylabel('数值')\nax.set_title('图表标题')\nplt.tight_layout()\nplt.show()", "plotly": "fig.update_layout(title='图表标题')\nfig.show()", "seaborn": "plt.title('图表标题')\nplt.tight_layout()\nplt.show()"},
    },
    "barh": {
        "plot_func": {"matplotlib": "ax.barh(categories, values)", "plotly": "px.bar(df, y='x', x='y', orientation='h')", "seaborn": "sns.barplot(data=df, y='x', x='y')"},
        "imports": {"matplotlib": "import matplotlib.pyplot as plt", "plotly": "import plotly.express as px\nimport pandas as pd", "seaborn": "import seaborn as sns\nimport matplotlib.pyplot as plt\nimport pandas as pd"},
        "data_prep": {"matplotlib": "categories = ['类别A','B','C','D']\nvalues = [25,40,35,20]", "plotly": "df = pd.DataFrame({'x': ['类别A','B','C','D'], 'y': [25,40,35,20]})", "seaborn": "df = pd.DataFrame({'x': ['类别A','B','C','D'], 'y': [25,40,35,20]})"},
        "setup": {"matplotlib": "fig, ax = plt.subplots(figsize=(8,5))", "plotly": "", "seaborn": "sns.set_style('whitegrid')"},
        "finish": {"matplotlib": "ax.set_xlabel('数值')\nax.set_title('图表标题')\nplt.tight_layout()\nplt.show()", "plotly": "fig.update_layout(title='图表标题')\nfig.show()", "seaborn": "plt.title('图表标题')\nplt.tight_layout()\nplt.show()"},
    },
    "line": {
        "plot_func": {"matplotlib": "ax.plot(x, y, marker='o', linewidth=2)", "plotly": "px.line(df, x='x', y='y', markers=True)", "seaborn": "sns.lineplot(data=df, x='x', y='y', marker='o')"},
        "imports": {"matplotlib": "import matplotlib.pyplot as plt", "plotly": "import plotly.express as px\nimport pandas as pd", "seaborn": "import seaborn as sns\nimport matplotlib.pyplot as plt\nimport pandas as pd"},
        "data_prep": {"matplotlib": "x = [1,2,3,4,5,6]\ny = [10,15,13,20,18,25]", "plotly": "df = pd.DataFrame({'x': [1,2,3,4,5,6], 'y': [10,15,13,20,18,25]})", "seaborn": "df = pd.DataFrame({'x': [1,2,3,4,5,6], 'y': [10,15,13,20,18,25]})"},
        "setup": {"matplotlib": "fig, ax = plt.subplots(figsize=(8,5))", "plotly": "", "seaborn": "sns.set_style('whitegrid')"},
        "finish": {"matplotlib": "ax.set_xlabel('时间')\nax.set_ylabel('数值')\nax.set_title('趋势图')\nax.grid(True, alpha=0.3)\nplt.tight_layout()\nplt.show()", "plotly": "fig.update_layout(title='趋势图')\nfig.show()", "seaborn": "plt.title('趋势图')\nplt.tight_layout()\nplt.show()"},
    },
    "pie": {
        "plot_func": {"matplotlib": "ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)", "plotly": "px.pie(df, names='label', values='value')", "seaborn": None},  # None = 回退 matplotlib
        "imports": {"matplotlib": "import matplotlib.pyplot as plt", "plotly": "import plotly.express as px\nimport pandas as pd", "seaborn": "import matplotlib.pyplot as plt"},
        "data_prep": {"matplotlib": "labels = ['A','B','C','D']\nsizes = [30,25,25,20]", "plotly": "df = pd.DataFrame({'label': ['A','B','C','D'], 'value': [30,25,25,20]})", "seaborn": "labels = ['A','B','C','D']\nsizes = [30,25,25,20]"},
        "setup": {"matplotlib": "fig, ax = plt.subplots(figsize=(7,7))", "plotly": "", "seaborn": "fig, ax = plt.subplots(figsize=(7,7))\n# Seaborn 不直接支持饼图，回退 matplotlib"},
        "finish": {"matplotlib": "ax.set_title('占比图')\nplt.tight_layout()\nplt.show()", "plotly": "fig.update_layout(title='占比图')\nfig.show()", "seaborn": "ax.set_title('占比图')\nplt.tight_layout()\nplt.show()"},
    },
    "scatter": {
        "plot_func": {"matplotlib": "ax.scatter(x, y, alpha=0.7, s=80)", "plotly": "px.scatter(df, x='x', y='y', trendline='ols')", "seaborn": "sns.scatterplot(data=df, x='x', y='y'); sns.regplot(data=df, x='x', y='y', scatter=False, color='red')"},
        "imports": {"matplotlib": "import matplotlib.pyplot as plt", "plotly": "import plotly.express as px\nimport pandas as pd", "seaborn": "import seaborn as sns\nimport matplotlib.pyplot as plt\nimport pandas as pd"},
        "data_prep": {"matplotlib": "x = [1,2,3,4,5,6,7,8]\ny = [2,4,5,7,6,8,9,11]", "plotly": "df = pd.DataFrame({'x': [1,2,3,4,5,6,7,8], 'y': [2,4,5,7,6,8,9,11]})", "seaborn": "df = pd.DataFrame({'x': [1,2,3,4,5,6,7,8], 'y': [2,4,5,7,6,8,9,11]})"},
        "setup": {"matplotlib": "fig, ax = plt.subplots(figsize=(7,7))", "plotly": "", "seaborn": "sns.set_style('whitegrid')"},
        "finish": {"matplotlib": "ax.set_xlabel('X')\nax.set_ylabel('Y')\nax.set_title('散点图')\nax.grid(True, alpha=0.3)\nplt.tight_layout()\nplt.show()", "plotly": "fig.update_layout(title='散点图')\nfig.show()", "seaborn": "plt.title('散点图')\nplt.tight_layout()\nplt.show()"},
    },
    "heatmap": {
        "plot_func": {"matplotlib": "sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)", "plotly": "px.imshow(data, text_auto='.2f', aspect='auto')", "seaborn": "sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd')"},
        "imports": {"matplotlib": "import matplotlib.pyplot as plt\nimport seaborn as sns\nimport numpy as np", "plotly": "import plotly.express as px\nimport numpy as np", "seaborn": "import seaborn as sns\nimport matplotlib.pyplot as plt\nimport numpy as np"},
        "data_prep": {"matplotlib": "data = np.random.rand(5,5)", "plotly": "data = np.random.rand(5,5)", "seaborn": "data = np.random.rand(5,5)"},
        "setup": {"matplotlib": "fig, ax = plt.subplots(figsize=(8,6))", "plotly": "", "seaborn": "plt.figure(figsize=(8,6))"},
        "finish": {"matplotlib": "ax.set_title('热力图')\nplt.tight_layout()\nplt.show()", "plotly": "fig.update_layout(title='热力图')\nfig.show()", "seaborn": "plt.title('热力图')\nplt.tight_layout()\nplt.show()"},
    },
    "histogram": {
        "plot_func": {"matplotlib": "ax.hist(data, bins=15, alpha=0.85)", "plotly": "px.histogram(df, x='value', nbins=15)", "seaborn": "sns.histplot(data=df, x='value', bins=15, kde=True)"},
        "imports": {"matplotlib": "import matplotlib.pyplot as plt", "plotly": "import plotly.express as px\nimport pandas as pd", "seaborn": "import seaborn as sns\nimport matplotlib.pyplot as plt\nimport pandas as pd"},
        "data_prep": {"matplotlib": "data = [65,70,80,85,90,75,68,72,82,78]", "plotly": "df = pd.DataFrame({'value': [65,70,80,85,90,75,68,72,82,78]})", "seaborn": "df = pd.DataFrame({'value': [65,70,80,85,90,75,68,72,82,78]})"},
        "setup": {"matplotlib": "fig, ax = plt.subplots(figsize=(8,5))", "plotly": "", "seaborn": "sns.set_style('whitegrid')"},
        "finish": {"matplotlib": "ax.set_xlabel('数值')\nax.set_ylabel('频数')\nax.set_title('直方图')\nplt.tight_layout()\nplt.show()", "plotly": "fig.update_layout(title='直方图')\nfig.show()", "seaborn": "plt.title('直方图')\nplt.tight_layout()\nplt.show()"},
    },
    "box": {
        "plot_func": {"matplotlib": "ax.boxplot(data, vert=True, patch_artist=True)", "plotly": "px.box(df, y='value')", "seaborn": "sns.boxplot(data=df, y='value')"},
        "imports": {"matplotlib": "import matplotlib.pyplot as plt", "plotly": "import plotly.express as px\nimport pandas as pd", "seaborn": "import seaborn as sns\nimport matplotlib.pyplot as plt\nimport pandas as pd"},
        "data_prep": {"matplotlib": "data = [65,70,75,80,85,90,95,100]", "plotly": "df = pd.DataFrame({'value': [65,70,75,80,85,90,95,100]})", "seaborn": "df = pd.DataFrame({'value': [65,70,75,80,85,90,95,100]})"},
        "setup": {"matplotlib": "fig, ax = plt.subplots(figsize=(6,6))", "plotly": "", "seaborn": "sns.set_style('whitegrid')"},
        "finish": {"matplotlib": "ax.set_ylabel('数值')\nax.set_title('箱线图')\nplt.tight_layout()\nplt.show()", "plotly": "fig.update_layout(title='箱线图')\nfig.show()", "seaborn": "plt.title('箱线图')\nplt.tight_layout()\nplt.show()"},
    },
    "stacked_bar": {
        "plot_func": {"matplotlib": "(matplotlib 堆叠需手动计算 bottom，见下方完整代码)", "plotly": "px.bar(df_long, x='x', y='value', color='group')", "seaborn": None},
        "imports": {"matplotlib": "import matplotlib.pyplot as plt\nimport numpy as np", "plotly": "import plotly.express as px\nimport pandas as pd", "seaborn": "import matplotlib.pyplot as plt\nimport numpy as np"},
        "data_prep": {"matplotlib": "categories = ['Q1','Q2','Q3','Q4']\na = [10,15,12,18]; b = [8,12,10,14]; c = [5,8,7,10]", "plotly": "df = pd.DataFrame({'x': ['Q1','Q2','Q3','Q4'], 'A': [10,15,12,18], 'B': [8,12,10,14], 'C': [5,8,7,10]})\ndf_long = df.melt(id_vars='x', var_name='group', value_name='value')", "seaborn": "# Seaborn 不直接支持堆叠柱状图，建议用 plotly\ncategories = ['Q1','Q2','Q3','Q4']\na = [10,15,12,18]; b = [8,12,10,14]; c = [5,8,7,10]"},
        "setup": {"matplotlib": "fig, ax = plt.subplots(figsize=(8,5))", "plotly": "", "seaborn": "fig, ax = plt.subplots(figsize=(8,5))\n# 回退 matplotlib"},
        "finish": {"matplotlib": "ax.bar(categories, a, label='A')\nax.bar(categories, b, bottom=a, label='B')\nax.bar(categories, c, bottom=np.array(a)+np.array(b), label='C')\nax.legend()\nax.set_title('堆叠柱状图')\nplt.tight_layout()\nplt.show()", "plotly": "fig.update_layout(title='堆叠柱状图')\nfig.show()", "seaborn": "ax.bar(categories, a, label='A')\nax.bar(categories, b, bottom=a, label='B')\nax.bar(categories, c, bottom=np.array(a)+np.array(b), label='C')\nax.legend()\nax.set_title('堆叠柱状图')\nplt.tight_layout()\nplt.show()"},
    },
}

# ===========================================================================
# 作用：根据图表类型和库，从骨架里拼装出完整代码，处理回退逻辑。
# ===========================================================================
def _generate_code(chart_type: str, library: str) -> str:
    """参数化生成代码 — 核心优化：用 8 行逻辑替代 800 行模板"""
    lib = library if library != "any" else "matplotlib"
    sk = CHART_SKELETONS.get(chart_type)

    if not sk:
        return f"# {chart_type} 图表\nimport matplotlib.pyplot as plt\n\n# TODO: 替换数据\ndata = []\n\nfig, ax = plt.subplots()\nax.plot(data)\nax.set_title('图表标题')\nplt.show()"

    # 获取该库的配置，无则降级 matplotlib
    def _get(key: str) -> str:
        val = sk[key].get(lib)
        if val is None:
            val = sk[key].get("matplotlib", "")
        return val

    imports = _get("imports")
    data = _get("data_prep")
    setup = _get("setup")
    plot = _get("plot_func")
    finish = _get("finish")

    # 拼装
    parts = [imports, "", data, ""]
    if setup:
        parts.append(setup)
        parts.append("")
    parts.append(plot)
    parts.append(finish)

    fallback_note = ""
    if sk["plot_func"].get(lib) is None:
        fallback_note = f"# ⚠️ {lib} 不直接支持 {chart_type}，已回退 matplotlib\n"

    return fallback_note + "\n".join(parts)


# ===========================================================================
# 作用：意图→关键词列表的映射，用于关键词匹配。
# ===========================================================================

INTENT_KEYWORDS: dict[str, list[str]] = {
    "compare_values": ["对比", "比较", "哪个多", "哪个少", "vs", "相差", "差距", "高低", "孰优", "compare"],
    "compare_ranks": ["排名", "排行", "前几", "后几", "top", "bottom", "rank", "名次", "第几", "榜首", "垫底"],
    "show_trend": ["趋势", "变化", "走势", "增长", "下降", "随时间", "trend", "演变", "波动", "上升", "下滑", "怎么变", "走向"],
    "show_composition": ["组成", "构成", "结构", "composition", "breakdown", "包含", "成分", "细分"],
    "show_distribution": ["分布", "分散", "集中", "distribution", "spread", "离散", "频率", "密度", "直方", "形态"],
    "show_correlation": ["相关", "关联", "关系", "correlation", "relationship", "影响", "联系", "相关性", "联动"],
    "show_part_to_whole": ["比例", "份额", "百分比", "proportion", "percentage", "share", "占多少", "几成", "比重", "占比"],
    "show_geographic": ["地图", "地理", "区域", "map", "geographic", "省份", "城市", "空间"],
}

INTENT_NAMES: dict[str, str] = {
    "compare_values": "对比数值大小", "compare_ranks": "对比排名",
    "show_trend": "展示变化趋势", "show_composition": "展示构成/结构",
    "show_distribution": "展示数据分布", "show_correlation": "展示相关关系",
    "show_part_to_whole": "展示占比/份额", "show_geographic": "展示地理分布",
}

INTENT_CHART_MAP: dict[str, list[tuple[str, str, float]]] = {
    "compare_values": [("bar", "柱状图", 95), ("barh", "水平柱状图", 85), ("lollipop", "棒棒糖图", 70)],
    "compare_ranks": [("barh", "水平柱状图", 95), ("bar", "柱状图", 85)],
    "show_trend": [("line", "折线图", 95), ("area", "面积图", 85)],
    "show_composition": [("stacked_bar", "堆叠柱状图", 90), ("donut", "环形图", 85), ("pie", "饼图", 80)],
    "show_distribution": [("histogram", "直方图", 90), ("box", "箱线图", 85), ("violin", "小提琴图", 75)],
    "show_correlation": [("scatter", "散点图", 95), ("heatmap", "热力图", 85), ("bubble", "气泡图", 70)],
    "show_part_to_whole": [("donut", "环形图", 90), ("pie", "饼图", 85), ("stacked_bar", "堆叠柱状图", 80)],
    "show_geographic": [("choropleth", "等值线地图", 90)],
}

REASONS: dict[str, str] = {
    "bar": "柱状图是展示类别对比的最直观选择，阅读门槛低。",
    "barh": "水平柱状图适合类别名称较长时使用，排名一目了然。",
    "line": "折线图最适合展示时间序列趋势，清晰反映变化方向和幅度。",
    "area": "面积图在折线图基础上强调数量累积。",
    "pie": "饼图适合展示占比（3-6 个类别最佳），直观但不适合精确对比。",
    "donut": "环形图是饼图的现代替代，中心可放置总计数字。",
    "scatter": "散点图是展示两变量相关关系的最佳选择。",
    "heatmap": "热力图用颜色编码强度，适合矩阵数据。",
    "histogram": "直方图展示数值变量的频率分布。",
    "box": "箱线图展示分布的五数概括，便于发现异常值。",
    "stacked_bar": "堆叠柱状图同时展示对比和构成。",
    "violin": "小提琴图结合箱线图和密度图，展示更丰富的分布信息。",
    "bubble": "气泡图在散点图基础上增加第三维度（气泡大小）。",
    "choropleth": "等值线地图用颜色深浅表示地理区域的数值差异。",
    "lollipop": "棒棒糖图是柱状图的优雅替代，减少'墨水'使用。",
}

# 反推荐规则（精简为 3 条核心规则）
ANTI_RULES: list[dict] = [
    {"chart": "pie",  "cond": lambda f: f.get("max_categories", 0) > 6, "msg": "饼图不推荐：类别>6，建议用柱状图"},
    {"chart": "line", "cond": lambda f: f.get("num_rows", 99) < 3,     "msg": "折线图不推荐：数据点<3，建议用柱状图"},
    {"chart": "heatmap", "cond": lambda f: f.get("num_numeric", 0) < 2, "msg": "热力图不推荐：需要≥2个数值列"},
]


# ===========================================================================
# 这是工具的主类，继承自BaseSkill（自定义基类，代码中未给出），负责协调整个流程。
# ===========================================================================

class ChartAdvisorSkill(BaseSkill):
    name = "chart_advisor"
    description = "根据数据特征和展示意图，推荐最合适的图表类型并生成代码。"
    triggers = ["图表", "可视化", "画图", "绘图", "选什么图", "用什么图", "图表推荐", "chart", "plot",
                "柱状图", "折线图", "饼图", "散点图", "热力图", "不知道怎么展示", "帮我选"]
    version = "2.1.0"
    author = "EnterpriseLearningAgent"
    input_schema = ChartAdvisorInput
    output_schema = ChartAdvisorOutput

    _st_model: ClassVar[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        if ChartAdvisorSkill._st_model is None and _ST_OK:
            try:
                ChartAdvisorSkill._st_model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2"
                )
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════
    # 入口
    # ═══════════════════════════════════════════════════════════
    def execute(self, inp: ChartAdvisorInput) -> ChartAdvisorOutput:
        desc = inp.data_description
        lib = inp.library

        # 1. 意图识别
        intent, conf = self._detect_intent(desc)

        # 2. 数据特征
        feats = self._analyze_data(inp.data_sample) if inp.data_sample else {}

        # 3. 生成推荐
        candidates = INTENT_CHART_MAP.get(intent, [("bar", "柱状图", 80)])
        recs: list[ChartRecommendation] = []
        for ct, cn, base_score in candidates:
            score = self._adjust(base_score, feats, ct)
            tips = self._tips(ct, feats)
            recs.append(ChartRecommendation(
                chart_type=cn, en_name=ct,
                suitability_score=round(min(score, 100), 1),
                reason=REASONS.get(ct, f"{cn}适合展示此类数据关系。"),
                code_skeleton=_generate_code(ct, lib),
                tips=tips,
            ))

        recs.sort(key=lambda r: r.suitability_score, reverse=True)
        top3 = recs[:3]

        # 4. 反推荐
        warnings = []
        if feats:
            warnings = [r["msg"] for r in ANTI_RULES if r["cond"](feats)]

        # 5. 低置信度追问
        needs_clarify = conf < 0.3
        question = None
        if needs_clarify:
            question = "🤔 不太确定您的意图，请问主要是：1)对比数值 2)展示趋势 3)看占比？"

        best = top3[0]
        return ChartAdvisorOutput(
            user_intent=INTENT_NAMES.get(intent, "通用展示"),
            confidence=round(conf, 3),
            recommendations=top3,
            best_pick=f"🎯 {best.chart_type}（适合度 {best.suitability_score}%）——{best.reason}",
            warnings=warnings,
            needs_clarification=needs_clarify,
            clarification_question=question,
        )

    # ═══════════════════════════════════════════════════════════
    # 作用：识别用户的展示意图，带lru_cache缓存（相同描述直接返回结果，提高效率）
    # ═══════════════════════════════════════════════════════════
    @lru_cache(maxsize=128)
    def _detect_intent(self, desc: str) -> tuple[str, float]:
        desc_lower = desc.lower()

        # Level 1: 关键词匹配（带权重）
        PRIORITY_WEIGHTS = {
            "compare_values": 1.5,
            "show_part_to_whole": 1.5,
            "show_trend": 1.2,
        }
        keyword_scores = {}
        for intent, kws in INTENT_KEYWORDS.items():
            weight = PRIORITY_WEIGHTS.get(intent, 1.0)
            score = sum(1 for kw in kws if kw in desc_lower) * weight
            keyword_scores[intent] = score

        best_kw = max(keyword_scores, key=keyword_scores.get)
        if keyword_scores[best_kw] > 0:
            conf = min(keyword_scores[best_kw] / 3.0, 0.8)
            return best_kw, conf

        # Level 2 & 3: 无关键词时直接返回默认，不走语义模型
        # 原因：模糊输入下语义模型的"高置信度"是噪声，不可信
        return "compare_values", 0.1

    # ═══════════════════════════════════════════════════════════
    # 作用：分析 CSV 样本的特征，用于评分调整和反推荐。
    # ═══════════════════════════════════════════════════════════
    # ... existing code ...
    @staticmethod
    def _analyze_data(sample: str) -> dict:
        try:
            df = pd.read_csv(StringIO(sample))
            # 无效 CSV 检测：单列且行数少视为无效
            if len(df.columns) < 2:
                return {}
        except Exception:
            return {}

        numeric = df.select_dtypes(include=[np.number]).columns.tolist()

        # 先识别所有时间列
        time_cols = set()
        for col in df.columns:
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().sum() > len(df) * 0.5:
                    time_cols.add(col)
            except Exception:
                pass

        # 分类列 = object/category 列 - 时间列
        cat_cols = [
            col for col in df.select_dtypes(include=["object", "category"]).columns
            if col not in time_cols
        ]
        max_cat = int(df[cat_cols].nunique().max()) if cat_cols else 0

        # 时间列判断：列名含时间关键词 + 值可解析为时间
        TIME_KEYWORDS = ["date", "time", "year", "month", "day", "日期", "时间", "年份", "月份"]
        has_time = any(
            any(kw in col.lower() for kw in TIME_KEYWORDS)
            for col in time_cols
        )

        return {
            "num_rows": len(df),
            "num_numeric": len(numeric),
            "max_categories": max_cat,
            "has_time": has_time,
        }

    # ... existing code ...

    # ═══════════════════════════════════════════════════════════
    # 作用：根据数据特征调整图表的基础分，让推荐更贴合数据。
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def _adjust(score: float, feats: dict, chart: str) -> float:
        if not feats:
            return score

        mc = feats.get("max_categories", 0)
        nn = feats.get("num_numeric", 0)
        nr = feats.get("num_rows", 0)
        ht = feats.get("has_time", False)

        # 饼图：类别多扣分
        if chart in ("pie", "donut") and mc > 6:
            score -= (mc - 6) * 5

        # 折线图：有时间加分，点少扣分
        if chart in ("line", "area"):
            if ht: score += 12
            if nr < 3: score -= 25

        # 散点/气泡：数值列要求
        if chart == "scatter" and nn < 2: score -= 40
        if chart == "bubble" and nn < 3: score -= 40

        # 热力图
        if chart == "heatmap" and nn < 2: score -= 35

        # 堆叠柱状图：类别多扣分
        if chart == "stacked_bar" and mc > 12: score -= 15

        return max(score, 10.0)

    # ═══════════════════════════════════════════════════════════
    # 作用：生成图表的使用建议，让用户画的图更专业。
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def _tips(chart: str, feats: dict) -> list[str]:
        common = ["确保坐标轴标签清晰", "使用色盲友好配色（如 viridis）"]
        specific = {
            "bar": ["类别≤10 个以保证可读性", "按数值排序通常更好"],
            "line": ["不超过 4-5 条线", "用不同颜色/线型区分"],
            "pie": ["类别≤6 个", "从 12 点方向顺时针排列"],
            "scatter": ["考虑添加趋势线", "数据量>1000 时降低 alpha"],
            "heatmap": ["标注数值", "矩阵>50×50 时关闭 annot"],
            "box": ["标注异常值", "配合散点图展示原始数据"],
        }
        result = common + specific.get(chart, [])

        # 大数提示
        if feats.get("num_rows", 0) > 500:
            big_tips = {"scatter": "数据量大，建议用 hexbin 或降低 alpha",
                        "heatmap": "矩阵较大，建议关闭数值标注",
                        "line": "数据点>100，建议滑动窗口平滑"}
            if chart in big_tips:
                result.append(f"💡 {big_tips[chart]}")

        return result
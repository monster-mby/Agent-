import pytest
from unittest import mock
from src.skills.preset.data_analysis.chart_advisor import (
    ChartAdvisorSkill,
    ChartAdvisorInput,
    _generate_code,
)


@pytest.fixture(autouse=True)
def mock_st_import():
    """全局 Mock，阻止所有 ChartAdvisorSkill 测试下载模型"""
    with mock.patch("src.skills.preset.data_analysis.chart_advisor.skill._ST_OK", False):
        yield


@pytest.fixture
def skill():
    """创建 ChartAdvisorSkill 实例的 fixture（跳过模型加载）"""
    ChartAdvisorSkill._st_model = None
    return ChartAdvisorSkill()



# ==========================================
# 1. 测试意图识别 (_detect_intent)
# ==========================================
def test_detect_intent_keywords_compare(skill):
    """测试关键词匹配：对比数值"""
    intent, conf = skill._detect_intent("对比A和B的销量，看看哪个多")
    assert intent == "compare_values"
    assert conf > 0.1


def test_detect_intent_keywords_trend(skill):
    """测试关键词匹配：展示趋势"""
    intent, conf = skill._detect_intent("展示过去一年的销量走势")
    assert intent == "show_trend"


def test_detect_intent_keywords_composition(skill):
    """测试关键词匹配：展示占比"""
    intent, conf = skill._detect_intent("查看各部门的预算占比")
    assert intent == "show_part_to_whole"


def test_detect_intent_default(skill):
    """测试无关键词时的默认意图"""
    intent, conf = skill._detect_intent("随便看看这些数据")
    assert intent == "compare_values"
    assert conf == 0.1


# ... existing code ...
def test_detect_intent_without_st_model(skill):
    """测试无语义模型时的关键词回退"""
    skill._st_model = None  # 直接设为 None，绕过 mock 路径问题
    intent, conf = skill._detect_intent("对比数值")
    assert intent == "compare_values"
# ... existing code ...



# ==========================================
# 2. 测试数据特征分析 (_analyze_data)
# ==========================================
def test_analyze_data_valid_csv():
    """测试有效 CSV 样本解析"""
    sample = """date,value,category
2023-01-01,100,A
2023-01-02,150,B
2023-01-03,120,A
2023-01-04,180,C
"""
    feats = ChartAdvisorSkill._analyze_data(sample)
    assert feats["num_rows"] == 4
    assert feats["num_numeric"] == 1  # value 列
    assert feats["max_categories"] == 3  # A, B, C
    assert feats["has_time"] is True  # date 列


def test_analyze_data_no_time_column():
    """测试无时间列的 CSV"""
    sample = """category,value
A,100
B,150
C,120
"""
    feats = ChartAdvisorSkill._analyze_data(sample)
    assert feats["has_time"] is False


def test_analyze_data_invalid_csv():
    """测试无效 CSV 输入"""
    sample = "这不是一个合法的 CSV 格式"
    feats = ChartAdvisorSkill._analyze_data(sample)
    assert feats == {}


# ==========================================
# 3. 测试评分调整 (_adjust)
# ==========================================
def test_adjust_pie_many_categories():
    """测试饼图类别过多时扣分"""
    base_score = 80
    feats = {"max_categories": 8}  # 超过 6 个类别
    adjusted = ChartAdvisorSkill._adjust(base_score, feats, "pie")
    assert adjusted < base_score  # 应扣分


def test_adjust_line_has_time():
    """测试折线图有时间列时加分"""
    base_score = 80
    feats = {"has_time": True, "num_rows": 5}
    adjusted = ChartAdvisorSkill._adjust(base_score, feats, "line")
    assert adjusted == base_score + 12  # 加 12 分


def test_adjust_line_few_rows():
    """测试折线图数据点过少时扣分"""
    base_score = 80
    feats = {"has_time": False, "num_rows": 2}  # 少于 3 个点
    adjusted = ChartAdvisorSkill._adjust(base_score, feats, "line")
    assert adjusted == base_score - 25  # 扣 25 分


def test_adjust_scatter_few_numeric():
    """测试散点图数值列不足时扣分"""
    base_score = 80
    feats = {"num_numeric": 1}  # 少于 2 个数值列
    adjusted = ChartAdvisorSkill._adjust(base_score, feats, "scatter")
    assert adjusted == base_score - 40  # 扣 40 分


def test_adjust_no_features():
    """测试无数据特征时不调整分数"""
    base_score = 80
    feats = {}
    adjusted = ChartAdvisorSkill._adjust(base_score, feats, "bar")
    assert adjusted == base_score


# ==========================================
# 4. 测试代码生成 (_generate_code)
# ==========================================
def test_generate_code_bar_matplotlib():
    """测试生成 Matplotlib 柱状图代码"""
    code = _generate_code("bar", "matplotlib")
    assert "import matplotlib.pyplot as plt" in code
    assert "ax.bar(categories, values)" in code
    assert "plt.show()" in code


def test_generate_code_line_plotly():
    """测试生成 Plotly 折线图代码"""
    code = _generate_code("line", "plotly")
    assert "import plotly.express as px" in code
    assert "px.line(df, x='x', y='y'" in code
    assert "fig.show()" in code


def test_generate_code_pie_seaborn_fallback():
    """测试 Seaborn 不支持饼图时回退到 Matplotlib"""
    code = _generate_code("pie", "seaborn")
    assert "⚠️ seaborn 不直接支持 pie" in code
    assert "ax.pie(sizes, labels=labels" in code  # 回退到 matplotlib


def test_generate_code_library_any():
    """测试 library='any' 时默认使用 Matplotlib"""
    code = _generate_code("bar", "any")
    assert "import matplotlib.pyplot as plt" in code


def test_generate_code_unknown_chart():
    """测试未知图表类型返回默认代码"""
    code = _generate_code("unknown_chart", "matplotlib")
    assert "# unknown_chart 图表" in code
    assert "ax.plot(data)" in code


# ==========================================
# 5. 测试使用建议 (_tips)
# ==========================================
def test_tips_common_and_specific(skill):
    """测试通用建议和特定图表建议"""
    tips = ChartAdvisorSkill._tips("bar", {})
    assert "确保坐标轴标签清晰" in tips  # 通用建议
    assert "类别≤10 个以保证可读性" in tips  # 柱状图特定建议


def test_tips_big_data_scatter(skill):
    """测试大数据量下的散点图建议"""
    tips = ChartAdvisorSkill._tips("scatter", {"num_rows": 600})
    assert any("数据量大，建议用 hexbin" in t for t in tips)


def test_tips_big_data_heatmap(skill):
    """测试大数据量下的热力图建议"""
    tips = ChartAdvisorSkill._tips("heatmap", {"num_rows": 600})
    assert any("矩阵较大，建议关闭数值标注" in t for t in tips)


# ==========================================
# 6. 测试端到端 execute 方法
# ==========================================
def test_execute_simple_input(skill):
    """测试简单输入的完整流程"""
    inp = ChartAdvisorInput(
        data_description="对比各产品的销量",
        library="matplotlib"
    )
    out = skill.execute(inp)

    # 验证输出结构
    assert out.user_intent == "对比数值大小"
    assert out.confidence > 0.1
    assert len(out.recommendations) > 0
    assert "柱状图" in out.best_pick
    assert out.needs_clarification is False
    assert out.warnings == []


def test_execute_with_data_sample(skill):
    """测试带数据样本的输入"""
    sample = """product,sales
A,100
B,150
C,120
"""
    inp = ChartAdvisorInput(
        data_description="对比各产品的销量",
        data_sample=sample,
        library="matplotlib"
    )
    out = skill.execute(inp)

    assert len(out.recommendations) > 0
    for rec in out.recommendations:
        assert rec.suitability_score > 0
        assert rec.code_skeleton != ""


def test_execute_low_confidence_clarification(skill):
    """测试低置信度时的追问逻辑"""
    inp = ChartAdvisorInput(
        data_description="看看这些数据",
        library="matplotlib"
    )
    out = skill.execute(inp)

    assert out.confidence < 0.3
    assert out.needs_clarification is True
    assert out.clarification_question is not None


def test_execute_anti_rule_warning(skill):
    """测试反推荐规则触发警告"""
    # 构造类别>6的样本，触发饼图反推荐
    sample = """category,value
A,10
B,10
C,10
D,10
E,10
F,10
G,10
"""
    inp = ChartAdvisorInput(
        data_description="查看各部分的占比",
        data_sample=sample,
        library="matplotlib"
    )
    out = skill.execute(inp)

    # 验证警告存在
    assert any("饼图不推荐" in w for w in out.warnings)

if __name__ == "__main__":
    pytest.main()
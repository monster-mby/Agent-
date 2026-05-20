import pytest
from src.skills.preset.data_analysis.data_cleaner.skill import (
    DataCleanerSkill,
    DataCleanerInput,
    ColumnInfo,
    CleaningSuggestion,
)


@pytest.fixture
def skill():
    """创建 DataCleanerSkill 实例的 fixture"""
    return DataCleanerSkill()


# ==========================================
# 1. 测试格式检测 (_detect_format)
# ==========================================
def test_detect_format_csv(skill):
    """测试标准 CSV 格式检测"""
    csv_data = "name,age,city\nAlice,30,New York\nBob,25,London"
    assert skill._detect_format(csv_data) == "csv"


def test_detect_format_csv_quoted(skill):
    """测试带引号的 CSV（含引号内逗号）"""
    csv_data = 'product,price\n"Apple, Inc.",100\nBanana,50'
    assert skill._detect_format(csv_data) == "csv"


def test_detect_format_json_object(skill):
    """测试 JSON 对象格式检测"""
    json_data = '{"name": "Alice", "age": 30}'
    assert skill._detect_format(json_data) == "json"


def test_detect_format_json_array(skill):
    """测试 JSON 数组格式检测"""
    json_data = '[{"name": "Bob"}, {"name": "Charlie"}]'
    assert skill._detect_format(json_data) == "json"


def test_detect_format_text(skill):
    """测试纯文本格式检测"""
    text_data = "这是一段自然语言描述，包含姓名：张三，年龄：25"
    assert skill._detect_format(text_data) == "text"


def test_detect_format_text_long_sentence(skill):
    """测试防止将长句误判为 CSV"""
    text_data = "这是一个很长的句子，虽然有逗号，但每个部分都超过20字符，应该是文本"
    assert skill._detect_format(text_data) == "text"


# ==========================================
# 2. 测试 CSV 行分割 (_split_csv_line)
# ==========================================
def test_split_csv_line_simple(skill):
    """测试简单 CSV 行分割"""
    line = "name,age,city"
    assert skill._split_csv_line(line) == ["name", "age", "city"]


def test_split_csv_line_quoted_comma(skill):
    """测试带引号内逗号的 CSV 行分割"""
    line = '"Apple, Inc.",100,"New York"'
    assert skill._split_csv_line(line) == ["Apple, Inc.", "100", "New York"]


# ==========================================
# 3. 测试 JSON 展平 (_flatten_json)
# ==========================================
def test_flatten_json_one_level(skill):
    """测试一层嵌套 JSON 展平"""
    obj = {"user": {"name": "Bob", "age": 25}, "city": "Paris"}
    assert skill._flatten_json(obj) == {
        "user_name": "Bob",
        "user_age": 25,
        "city": "Paris"
    }


def test_flatten_json_no_nesting(skill):
    """测试无嵌套 JSON（直接返回）"""
    obj = {"name": "Alice", "age": 30}
    assert skill._flatten_json(obj) == obj


def test_flatten_json_deep_nesting(skill):
    """测试深层嵌套（只展平第一层）"""
    obj = {"a": {"b": {"c": 1}}}  # 深层嵌套
    assert skill._flatten_json(obj) == {"a_b": {"c": 1}}  # 不继续展平


# ==========================================
# 4. 测试类型推断 (_guess_dtype)
# ==========================================
def test_guess_dtype_integer(skill):
    """测试整数类型推断"""
    values = ["10", "20", "30", "null", "40"]
    assert skill._guess_dtype(values) == "integer"


def test_guess_dtype_float(skill):
    """测试浮点数类型推断"""
    values = ["10.5", "20.3", "30", "nan", "40.7"]
    assert skill._guess_dtype(values) == "float"


def test_guess_dtype_datetime(skill):
    """测试日期类型推断"""
    values = ["2023-01-01", "2023/02/15", "03-20-2023", "null"]
    assert skill._guess_dtype(values) == "datetime"


def test_guess_dtype_mixed_numeric_string(skill):
    """测试混合类型推断"""
    values = ["10", "20", "30元", "40", "abc"]
    assert skill._guess_dtype(values) == "mixed_numeric_string"


def test_guess_dtype_empty_or_null(skill):
    """测试空值列推断"""
    values = ["", "null", "nan", "n/a"]
    assert skill._guess_dtype(values) == "empty_or_null"


# ==========================================
# 5. 测试列解析 (_parse_columns)
# ==========================================
def test_parse_columns_csv(skill):
    """测试 CSV 列解析"""
    csv_data = """name,age,city,score
Alice,30,New York,85.5
Bob,25,London,
Charlie,,Paris,90.0
"""
    columns = skill._parse_columns(csv_data, "csv")
    assert len(columns) == 4

    # 验证 name 列
    name_col = next(c for c in columns if c.name == "name")
    assert name_col.dtype_guess == "string"
    assert name_col.sample_values == ["Alice", "Bob", "Charlie"]

    # 验证 age 列（含缺失值）
    age_col = next(c for c in columns if c.name == "age")
    assert age_col.dtype_guess == "integer"
    assert age_col.null_count == 1


def test_parse_columns_json(skill):
    """测试 JSON 列解析（含嵌套展平）"""
    json_data = '''{
        "user": {"name": "Alice", "age": 30},
        "city": "New York"
    }'''
    columns = skill._parse_columns(json_data, "json")
    assert len(columns) == 3  # user_name, user_age, city

    user_name_col = next(c for c in columns if c.name == "user_name")
    assert user_name_col.sample_values == ["Alice"]


# def test_parse_columns_text(skill):
#     """测试文本列解析（通过正则匹配）"""
#     text_data = "张三，年龄：25，城市：北京"
#     columns = skill._parse_columns(text_data, "text")
#     assert len(columns) == 3
#     assert {c.name for c in columns} == {"姓名", "年龄", "城市"}


# ==========================================
# 6. 测试问题检测 (_detect_issues)
# ==========================================
def test_detect_issues_missing_values(skill):
    """测试缺失值检测"""
    columns = [
        ColumnInfo(
            name="age",
            dtype_guess="integer",
            null_count=2,
            sample_values=["10", "20", "", "null", "30"]
        )
    ]
    suggestions = skill._detect_issues("", columns)
    missing_suggestion = next(s for s in suggestions if "缺失值" in s.issue)
    assert missing_suggestion.severity == "high"  # 2/5 > 0.3
    assert "fillna" in missing_suggestion.code_snippet


def test_detect_issues_mixed_type(skill):
    """测试混合类型检测"""
    columns = [
        ColumnInfo(
            name="price",
            dtype_guess="mixed_numeric_string",
            sample_values=["100", "200元", "300"]
        )
    ]
    suggestions = skill._detect_issues("", columns)
    mixed_suggestion = next(s for s in suggestions if "混合类型" in s.issue)
    assert mixed_suggestion.severity == "high"
    assert "提取数值" in mixed_suggestion.code_snippet


def test_detect_issues_empty_column(skill):
    """测试空列检测"""
    columns = [
        ColumnInfo(
            name="empty_col",
            dtype_guess="empty_or_null",
            sample_values=["", "null", "nan"]
        )
    ]
    suggestions = skill._detect_issues("", columns)
    empty_suggestion = next(s for s in suggestions if "整列可能为空" in s.issue)
    assert empty_suggestion.severity == "high"
    assert "drop" in empty_suggestion.code_snippet


def test_detect_issues_duplicate_keyword(skill):
    """测试重复数据检测（通过关键词）"""
    suggestions = skill._detect_issues("数据中有很多重复的行", [])
    duplicate_suggestion = next(s for s in suggestions if "重复数据" in s.issue)
    assert duplicate_suggestion.severity == "medium"
    assert "drop_duplicates" in duplicate_suggestion.code_snippet


# ==========================================
# 7. 测试端到端 execute 方法
# ==========================================
def test_execute_csv_end_to_end(skill):
    """测试 CSV 输入的完整执行流程"""
    input_data = DataCleanerInput(
        data_description="""name,age,city,score
Alice,30,New York,85.5
Bob,25,London,
Charlie,,Paris,90.0
""",
        format="csv"
    )
    output = skill.execute(input_data)

    # 验证输出结构
    assert output.data_shape["columns"] == 4
    assert output.data_shape["rows_estimated"] == 3  # 4行 - 表头
    assert len(output.columns) == 4
    assert len(output.suggestions) >= 2  # age 和 score 有缺失值
    assert "检测到 4 个字段" in output.summary


def test_execute_json_end_to_end(skill):
    """测试 JSON 输入的完整执行流程"""
    input_data = DataCleanerInput(
        data_description='''[{"user": {"name": "Alice"}, "city": "New York"}]''',
        format="json"
    )
    output = skill.execute(input_data)

    assert output.data_shape["columns"] == 2  # user_name, city
    assert any(c.name == "user_name" for c in output.columns)

if __name__ == "__main__":
    pytest.main()
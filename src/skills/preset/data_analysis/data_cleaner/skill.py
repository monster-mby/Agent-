"""
Data Cleaner Skill — 数据清洗建议（优化版）
这是一个企业级数据清洗建议生成工具，通过分析数据样本（CSV/JSON/ 文本）自动识别数据质量问题，并
输出结构化的清洗方案（含可执行代码片段）。下面从功能概览、核心结构、类 / 方法详解、调用链路、代码质量五个维度进行深度解析。
一、功能概览
这个工具的核心能力是：
多格式支持：自动识别 CSV（正确处理引号内逗号）、JSON（支持嵌套展平）、文本格式
列信息解析：提取列名、推测数据类型、统计缺失值 / 唯一值、采样数据
问题检测：识别缺失值、混合类型、整列为空、重复 / 异常值（基于描述）
方案输出：生成带优先级的清洗建议、Python 代码片段、处理摘要

v1.1.0 优化改动:
  - CSV 解析改用 stdlib csv 模块（正确处理引号内逗号）
  - 删除从未使用的 _ISSUE_PATTERNS + Optional import
  - JSON 解析支持嵌套结构（递归展平一层）
  - _estimate_rows 传入 fmt 避免重复检测
  - _detect_format 增加 text 误判防范
  - 缺失率统计加明确警告（样本量小）
  - _make_summary 始终显示总列数

  ┌─────────────────────────────────────────────────────────────┐
│  1. 数据模型层（Pydantic）                                    │
│     - ColumnInfo/CleaningSuggestion（业务实体）               │
│     - DataCleanerInput/Output（输入输出校验）                 │
├─────────────────────────────────────────────────────────────┤
│  2. 核心业务层（技能类）                                      │
│     - DataCleanerSkill（继承BaseSkill）                       │
└─────────────────────────────────────────────────────────────┘
DataCleanerSkill.execute()
  ├→ _detect_format()  [检测数据格式]
  ├→ _parse_columns()  [解析列信息]
  │   ├→ _split_csv_line()  [CSV分支：正确分割CSV行]
  │   ├→ _flatten_json()    [JSON分支：展平嵌套JSON]
  │   └→ _guess_dtype()      [所有分支：推测数据类型]
  ├→ _estimate_rows()  [估算数据行数]
  ├→ _detect_issues()  [检测数据问题]
  ├→ _prioritize()     [确定处理优先级]
  └→ _make_summary()   [生成清洗摘要]
"""

import csv
import json
import re
from io import StringIO
from typing import Optional  # 仍然保留因为 ColumnInfo 中用了 Optional?

from pydantic import BaseModel, Field

from src.skills.base.base_skill import BaseSkill


class ColumnInfo(BaseModel):
    """存储单个列的元信息"""
    name: str
    dtype_guess: str
    null_count: int = 0
    unique_count: int = 0
    sample_values: list[str] = Field(default_factory=list)


class CleaningSuggestion(BaseModel):
    """单条清洗建议"""
    column: str
    issue: str
    severity: str  # high / medium / low
    suggestion: str
    code_snippet: str = ""


class DataCleanerInput(BaseModel):
    """数据清洗输入校验模型"""
    data_description: str = Field(
        ...,
        description="数据描述：可以是CSV前几行、JSON样本、或自然语言描述",
        min_length=10,
    )
    format: str = Field(
        default="auto",
        description="数据格式：csv / json / text / auto",
    )


class DataCleanerOutput(BaseModel):
    """数据清洗输出校验模型"""
    data_shape: dict = Field(default_factory=dict, description="数据规模估计")
    columns: list[ColumnInfo] = Field(default_factory=list)
    suggestions: list[CleaningSuggestion] = Field(default_factory=list)
    priority_order: list[str] = Field(default_factory=list, description="建议处理顺序")
    summary: str = ""


class DataCleanerSkill(BaseSkill):
    """这是工具的主类，继承自BaseSkill（自定义基类，代码中未给出），负责协调整个清洗建议生成流程。（优化版 v1.1.0）"""

    name: str = "data_cleaner"
    description: str = (
        "分析数据样本，输出结构化的清洗方案：缺失值处理、异常值检测、"
        "类型转换建议、去重策略。不实际清洗数据，只给可执行的建议+代码片段。"
    )
    triggers: list[str] = [
        "数据清洗", "data cleaning", "清洗数据", "脏数据", "缺失值",
        "异常值", "去重", "数据预处理", "数据质量", "数据检查",
        "clean data", "data quality", "预处理",
    ]
    version: str = "1.1.0"
    author: str = "EnterpriseLearningAgent"
    changelog: str = (
        "v1.1.0: CSV解析改为stdlib csv模块（修复引号内逗号bug）；"
        "删除无用_ISSUE_PATTERNS；JSON支持嵌套展平；"
        "格式检测增加text误判防范；缺失率加警告；摘要格式统一"
    )
    input_schema = DataCleanerInput
    output_schema = DataCleanerOutput


    def execute(self, input_data: DataCleanerInput) -> DataCleanerOutput:
        desc = input_data.data_description.strip()

        # 1. 解析数据格式（只检测一次，后续复用）
        fmt = self._detect_format(desc)

        # 2. 解析列信息
        columns = self._parse_columns(desc, fmt)

        # 3. 检测问题并生成建议
        suggestions = self._detect_issues(desc, columns)

        # 4. 确定处理优先级
        priority_order = self._prioritize(suggestions)

        # 5. 生成摘要
        summary = self._make_summary(columns, suggestions)

        return DataCleanerOutput(
            data_shape={
                "rows_estimated": self._estimate_rows(desc, fmt),
                "columns": len(columns),
            },
            columns=columns,
            suggestions=suggestions,
            priority_order=priority_order,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # 格式检测（增加 text 误判防范）
    # ------------------------------------------------------------------
    def _detect_format(self, desc: str) -> str:
        """检测数据格式，增加误判防范"""
        stripped = desc.strip()

        # JSON 检测
        if stripped.startswith("{") or stripped.startswith("["):
            return "json"

        lines = stripped.split("\n")
        first_line = lines[0].strip() if lines else ""

        # CSV 检测：逗号或制表符分隔，且表头不含中文冒号
        if ("," in first_line or "\t" in first_line) and "：" not in first_line:
            # 进一步检测：逗号分隔的字段应该都是短词（非长句）
            parts = first_line.split(",")
            # 如果第一行逗号分隔后每个部分都超过 20 字符 → 更像是自然语言句子
            if all(len(p.strip()) < 20 for p in parts):
                return "csv"
            # 如果逗号分隔后只有一个部分 → 不是 CSV
            if len(parts) <= 1:
                return "text"

        return "text"

    # ------------------------------------------------------------------
    # 行数估算（传入 fmt 避免重复检测）
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_rows(desc: str, fmt: str) -> int:
        """估算行数（传入 fmt 避免重复检测）"""
        lines = desc.strip().split("\n")
        if len(lines) <= 1:
            return 1
        if fmt == "csv":
            return len(lines) - 1  # 减去表头
        return len(lines)

    # ------------------------------------------------------------------
    # CSV 行分割（用 stdlib csv 模块替代 split(",")）
    # ------------------------------------------------------------------
    @staticmethod
    def _split_csv_line(line: str) -> list[str]:
        """使用 csv.reader 正确处理引号内逗号"""
        reader = csv.reader(StringIO(line))
        return next(reader)

    # ------------------------------------------------------------------
    # JSON 展平（递归展平一层）
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_json(obj: dict, prefix: str = "") -> dict:
        """递归展平嵌套 JSON 一层（不处理深层嵌套）"""
        result = {}
        for k, v in obj.items():
            new_key = f"{prefix}{k}"
            if isinstance(v, dict) and not prefix:
                # 只展平第一层嵌套，避免过深
                result.update(DataCleanerSkill._flatten_json(v, f"{k}_"))
            else:
                result[new_key] = v
        return result

    # ------------------------------------------------------------------
    # 解析列信息
    # ------------------------------------------------------------------
    def _parse_columns(self, desc: str, fmt: str) -> list[ColumnInfo]:
        """解析列信息"""
        columns = []

        if fmt == "csv":
            lines = desc.strip().split("\n")
            if not lines:
                return columns

            header_line = lines[0]
            headers = self._split_csv_line(header_line)

            data_lines = lines[1:6] if len(lines) > 1 else []
            for i, col_name in enumerate(headers):
                col_name = col_name.strip().strip('"').strip("'")
                if not col_name:
                    col_name = f"column_{i}"

                sample_vals = []
                for dl in data_lines:
                    vals = self._split_csv_line(dl)
                    if i < len(vals):
                        sample_vals.append(vals[i].strip().strip('"').strip("'"))

                dtype = self._guess_dtype(sample_vals)
                null_count = sum(
                    1 for v in sample_vals
                    if v.lower() in ("", "null", "none", "nan", "n/a", "na")
                )

                columns.append(ColumnInfo(
                    name=col_name,
                    dtype_guess=dtype,
                    null_count=null_count,
                    unique_count=len(set(sample_vals)),
                    sample_values=sample_vals[:3],
                ))

        elif fmt == "json":
            try:
                data = json.loads(desc)
                if isinstance(data, list) and data:
                    data = data[0]
                if isinstance(data, dict):
                    # 展平嵌套结构
                    flat = self._flatten_json(data)
                    for key, val in flat.items():
                        dtype = self._guess_dtype([str(val)])
                        columns.append(ColumnInfo(
                            name=key,
                            dtype_guess=dtype,
                            sample_values=[str(val)[:50]],
                        ))
            except json.JSONDecodeError:
                pass



        elif fmt == "text":
            # 找到所有 "key：value" 候选（key 1~6 字符）
            # 添加负向回顾后发：确保 key 前面不是中文字符/字母/数字（避免匹配到 "数据包含姓名"）
            pattern = r'(?<![\w\u4e00-\u9fff])([\w\u4e00-\u9fff]{1,6})[：:]\s*[\w\u4e00-\u9fff]+'
            all_matches = list(re.finditer(pattern, desc))
            # 按冒号位置分组（同一冒号可能有多条不同长度的 key 命中）
            colon_groups: dict[int, list[tuple[str, int]]] = {}
            for m in all_matches:
                colon_pos = m.start(1) + len(m.group(1))
                colon_groups.setdefault(colon_pos, []).append((m.group(1), m.end()))
            seen_keys: set[str] = set()
            last_end = -1
            for colon_pos in sorted(colon_groups):
                # 跳过已被上一个非重叠匹配覆盖的冒号
                if colon_pos < last_end:
                    continue
                candidates = colon_groups[colon_pos]
                # 优先在 2~4 字符窗口内选最短（覆盖中英文常见字段名）
                preferred = [(k, end) for k, end in candidates if 2 <= len(k) <= 4]
                if preferred:
                    best_key, match_end = min(preferred, key=lambda x: len(x[0]))
                else:
                    # 2~4 窗内没候选时 fallback：取所有候选中最短的
                    best_key, match_end = min(candidates, key=lambda x: len(x[0]))
                if best_key not in seen_keys:
                    columns.append(ColumnInfo(
                        name=best_key,
                        dtype_guess="unknown",
                    ))
                    seen_keys.add(best_key)
                    last_end = match_end
        return columns

    # ------------------------------------------------------------------
    # 类型推断
    # ------------------------------------------------------------------
    @staticmethod
    def _guess_dtype(values: list[str]) -> str:
        """推测列的数据类型"""
        if not values:
            return "unknown"

        numeric_count = 0
        int_count = 0
        float_count = 0
        date_count = 0
        total = 0

        for v in values:
            if not v or v.lower() in ("null", "none", "nan", "n/a", "na", ""):
                continue
            total += 1

            try:
                int(v)
                int_count += 1
                numeric_count += 1
                continue
            except (ValueError, TypeError):
                pass

            try:
                float(v)
                float_count += 1
                numeric_count += 1
                continue
            except (ValueError, TypeError):
                pass

            if re.match(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', v) or \
               re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', v):
                date_count += 1
                continue

        if total == 0:
            return "empty_or_null"
        if date_count > total * 0.5:
            return "datetime"
        if int_count > total * 0.8:
            return "integer"
        if numeric_count > total * 0.8:
            return "float"
        if numeric_count > total * 0.5:
            return "mixed_numeric_string"
        return "string"

    # ------------------------------------------------------------------
    # 问题检测
    # ------------------------------------------------------------------
    def _detect_issues(self, desc: str, columns: list[ColumnInfo]) -> list[CleaningSuggestion]:
        """检测数据问题"""
        suggestions = []

        sample_size_hint = ""
        if columns:
            max_samples = max(len(c.sample_values) for c in columns)
            if max_samples <= 5:
                sample_size_hint = (
                    f" ⚠️ 注意：当前仅基于 {max_samples} 行样本进行分析，"
                    f"统计结果仅供参考，建议在全量数据上重新验证。"
                )

        for col in columns:
            # 问题1：缺失值
            if col.null_count > 0:
                sample_count = len(col.sample_values)
                ratio = col.null_count / max(sample_count, 1)
                severity = "high" if ratio > 0.3 else ("medium" if ratio > 0.1 else "low")

                suggestions.append(CleaningSuggestion(
                    column=col.name,
                    issue=f"缺失值：{col.null_count} 个空值（样本共 {sample_count} 行）{sample_size_hint}",
                    severity=severity,
                    suggestion=(
                        f"对于 {col.name} 列的缺失值："
                        f"{'如果该列是关键字段，建议填充（均值/中位数/众数/前向填充）；'
                         '如果缺失率过高（>50%），考虑删除该列。' if severity == 'high'
                         else '建议用均值（数值型）或众数（分类型）填充。'}"
                    ),
                    code_snippet=(
                        f"# 缺失值处理\n"
                        f"df['{col.name}'].fillna(df['{col.name}'].median(), inplace=True)"
                        if col.dtype_guess in ("integer", "float")
                        else f"df['{col.name}'].fillna(df['{col.name}'].mode()[0], inplace=True)"
                    ),
                ))

            # 问题2：类型混合
            if col.dtype_guess == "mixed_numeric_string":
                suggestions.append(CleaningSuggestion(
                    column=col.name,
                    issue="混合类型：数值和字符串混在同一个列中",
                    severity="high",
                    suggestion=f"建议检查 {col.name} 列，可能包含单位字符（如'元'、'kg'）、"
                              f"异常字符或格式不一致的数据。统一清洗为纯数值。",
                    code_snippet=(
                        f"# 提取数值\n"
                        f"df['{col.name}'] = df['{col.name}'].astype(str).str.extract(r'(\\d+\\.?\\d*)').astype(float)"
                    ),
                ))

            # 问题3：样本全为空
            if col.dtype_guess == "empty_or_null":
                suggestions.append(CleaningSuggestion(
                    column=col.name,
                    issue=f"整列可能为空或全为 NULL{sample_size_hint}",
                    severity="high",
                    suggestion=f"列 {col.name} 在样本中全为空值，建议确认原始数据中该列是否有有效数据。"
                              f"如果整列为空，考虑删除。",
                    code_snippet=f"df.drop(columns=['{col.name}'], inplace=True)",
                ))

        # 全局问题
        desc_lower = desc.lower()
        if "重复" in desc_lower or "duplicate" in desc_lower:
            suggestions.append(CleaningSuggestion(
                column="(全部)",
                issue="数据描述中提到重复数据",
                severity="medium",
                suggestion="建议使用 drop_duplicates() 去除完全重复的行，"
                          "或基于关键列检查重复。",
                code_snippet="df.drop_duplicates(subset=['关键列'], keep='first', inplace=True)",
            ))

        if "异常" in desc_lower or "outlier" in desc_lower or "离群" in desc_lower:
            suggestions.append(CleaningSuggestion(
                column="(数值列)",
                issue="数据描述中提到异常值",
                severity="medium",
                suggestion="使用 IQR（四分位距）或 Z-score 方法检测异常值。"
                          "可以删除、盖帽（capping）或单独标记。",
                code_snippet=(
                    "# IQR 异常值检测\n"
                    "Q1 = df['数值列'].quantile(0.25)\n"
                    "Q3 = df['数值列'].quantile(0.75)\n"
                    "IQR = Q3 - Q1\n"
                    "lower = Q1 - 1.5 * IQR\n"
                    "upper = Q3 + 1.5 * IQR\n"
                    "df = df[(df['数值列'] >= lower) & (df['数值列'] <= upper)]"
                ),
            ))

        return suggestions

    # ------------------------------------------------------------------
    # 优先级排序
    # ------------------------------------------------------------------
    @staticmethod
    def _prioritize(suggestions: list[CleaningSuggestion]) -> list[str]:
        """确定处理优先级"""
        high = [s for s in suggestions if s.severity == "high"]
        medium = [s for s in suggestions if s.severity == "medium"]
        low = [s for s in suggestions if s.severity == "low"]

        order = []
        order.append("1. 处理高优先级问题")
        for s in high:
            order.append(f"   - [{s.column}] {s.issue}")
        order.append("2. 处理中优先级问题")
        for s in medium:
            order.append(f"   - [{s.column}] {s.issue}")
        if low:
            order.append("3. 处理低优先级问题（可选）")
            for s in low:
                order.append(f"   - [{s.column}] {s.issue}")

        return order

    # ------------------------------------------------------------------
    # 摘要生成（统一显示总列数）
    # ------------------------------------------------------------------
    @staticmethod
    def _make_summary(columns: list[ColumnInfo], suggestions: list[CleaningSuggestion]) -> str:
        """生成清洗摘要，始终显示总列数"""
        high_count = sum(1 for s in suggestions if s.severity == "high")
        medium_count = sum(1 for s in suggestions if s.severity == "medium")
        low_count = sum(1 for s in suggestions if s.severity == "low")

        col_list = ", ".join(c.name for c in columns[:5])
        col_list += f"（共 {len(columns)} 列）"

        return (
            f"检测到 {len(columns)} 个字段：{col_list}。\n"
            f"共发现 {len(suggestions)} 个数据质量问题："
            f"{high_count} 个高优先级、{medium_count} 个中优先级、{low_count} 个低优先级。\n"
            f"建议按优先级顺序逐步处理。"
        )

    def __repr__(self) -> str:
        return f"DataCleanerSkill(version={self.version!r})"
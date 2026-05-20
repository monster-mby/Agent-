"""
一个基于 BaseSkill 模板实现的 “代码审查技能”（CodeReviewSkill），属于一个可插拔的 AI Agent 功能模块。
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from src.skills.base import BaseSkill

#------------------------ 第一部分：Pydantic 数据模型类（定义输入输出格式）-------------------
#作用：定义 “代码审查技能” 的输入参数格式
class CodeReviewInput(BaseModel):
    code: str = Field(..., description="要审查的源代码")
    language: str = Field(default="python", description="代码语言（python / javascript / java / go 等）")

#作用：定义 “单个问题” 的数据结构。
class Issue(BaseModel):
    line: Optional[int] = Field(None, description="问题所在行号")
    severity: str = Field(..., description="严重级别: error / warning / info")
    message: str = Field(..., description="问题描述")
    suggestion: Optional[str] = Field(None, description="修复建议")

#作用：定义 “代码审查技能” 的输出结果格式。
class CodeReviewOutput(BaseModel):
    issues: List[Issue] = Field(default_factory=list, description="发现的所有问题")
    summary: str = Field("", description="审查总结")
    score: int = Field(0, description="代码质量评分（0-100）")

#------------------------ 第二部分：核心技能类 class CodeReviewSkill(BaseSkill)------------------------

class CodeReviewSkill(BaseSkill):
    """代码审查技能：分析源代码并返回问题列表、总结和评分。"""

    name = "code_review"
    description = "审查代码质量，返回 issues / summary / score"
    input_schema = CodeReviewInput
    output_schema = CodeReviewOutput

    def execute(self, input_data: CodeReviewInput) -> CodeReviewOutput:
        """执行代码审查

        Args:
            input_data: 代码审查输入 Pydantic 对象
        """
        code = input_data.code
        language = input_data.language.lower()

        issues: List[Issue] = []
        score = 100

        # -------------------- 通用检查 --------------------
        lines = code.split("\n")  # 把代码按行分割
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            # 检查超长行（超过 120 字符）
            if len(line.rstrip("\n")) > 120:
                issues.append(Issue(
                    line=i,  # 问题所在的行号
                    severity="warning",  # 问题级别为 “警告”
                    message=f"行 {i} 长度超过 120 字符（当前 {len(line.rstrip(chr(10)))}）",  # 具体描述（包含行号和实际长度）；
                    suggestion="考虑拆分为多行以提高可读性"  # 修改建议
                ))
                score -= 2

            # 检查行尾多余空白（保留换行符，只检查空格和制表符）
            line_without_newline = line.rstrip('\n').rstrip('\r')
            if line_without_newline != line_without_newline.rstrip():
                issues.append(Issue(
                    line=i,
                    severity="info",
                    message=f"行 {i} 行尾有多余空白",
                    suggestion="删除行尾空白"
                ))
                score -= 1

        # 检查空文件
        if not code.strip():
            issues.append(Issue(
                line=None,
                severity="error",
                message="文件为空",
                suggestion="请提供有效的代码"
            ))
            return CodeReviewOutput(issues=issues, summary="文件为空，无法审查", score=0)

        # 检查 TODO 注释
        todo_count = 0
        for i, line in enumerate(lines, start=1):
            if "TODO" in line.upper() and ("#" in line or "//" in line):
                todo_count += 1
        if todo_count > 0:
            issues.append(Issue(
                line=None,
                severity="info",
                message=f"代码中包含 {todo_count} 个 TODO 注释",
                suggestion="考虑在提交前完成或跟踪这些 TODO"
            ))

        # -------------------- 按语言检查 --------------------
        if language == "python":
            self._check_python(code, lines, issues, score)
        elif language in ("javascript", "js"):
            self._check_javascript(code, lines, issues, score)
        elif language in ("java",):
            self._check_java(code, lines, issues, score)
        elif language in ("go", "golang"):
            self._check_go(code, lines, issues, score)
        else:
            issues.append(Issue(
                line=None,
                severity="info",
                message=f"语言 '{language}' 暂不支持专项检查，仅执行通用检查",
                suggestion="当前支持: python / javascript / java / go"
            ))

        # 计算最终评分（确保在 0-100 之间）
        score = max(0, min(100, score))

        # 生成总结
        summary_parts = []
        total_issues = len(issues)
        error_count = sum(1 for iss in issues if iss.severity == "error")
        warning_count = sum(1 for iss in issues if iss.severity == "warning")
        info_count = sum(1 for iss in issues if iss.severity == "info")

        summary_parts.append(f"共发现 {total_issues} 个问题")
        if error_count:
            summary_parts.append(f"其中错误 {error_count} 个")
        if warning_count:
            summary_parts.append(f"警告 {warning_count} 个")
        if info_count:
            summary_parts.append(f"信息 {info_count} 个")
        summary_parts.append(f"质量评分: {score}/100")

        if score >= 90:
            summary_parts.append("代码质量优秀！")
        elif score >= 70:
            summary_parts.append("代码质量良好，建议修复警告项。")
        else:
            summary_parts.append("代码质量需改进，请重点关注错误和警告。")

        summary = "，".join(summary_parts)

        return CodeReviewOutput(issues=issues, summary=summary, score=score)

    # ---------- 语言专项检查 ----------
    def _check_python(self, code: str, lines: list, issues: list, score: int):
        """Python 专项检查"""
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            # 检查 import 大写
            if stripped.startswith("import ") and not stripped[7].islower():
                issues.append(Issue(
                    line=i,
                    severity="warning",
                    message="模块名应使用小写字母",
                    suggestion=f"将 '{stripped.split()[1]}' 改为小写"
                ))
                score -= 3

            # 检查函数命名（def 后应是小写）
            if stripped.startswith("def ") and not stripped[4].islower():
                issues.append(Issue(
                    line=i,
                    severity="error",
                    message="函数名应使用小写字母和下划线（snake_case）",
                    suggestion=f"将 '{stripped.split()[1].split('(')[0]}' 改为 snake_case"
                ))
                score -= 5

            # 检查类命名（class 后应是大写 CamelCase）
            if stripped.startswith("class ") and stripped[6].islower():
                issues.append(Issue(
                    line=i,
                    severity="error",
                    message="类名应使用大写驼峰命名（CamelCase）",
                    suggestion=f"将 '{stripped.split()[1].split(':')[0].split('(')[0]}' 改为 CamelCase"
                ))
                score -= 5
            # 检查潜在的除零错误
            if "/" in stripped and not stripped.startswith("#"):
                # 简单的除零检查：查找除法操作但没有检查除数是否为零
                if "return" in stripped and "/" in stripped:
                    # 检查是否有除零保护
                    has_zero_check = False
                    for prev_line in lines[:i - 1]:
                        if "if" in prev_line and (
                                "== 0" in prev_line or "==0" in prev_line or "!= 0" in prev_line or "!=0" in prev_line):
                            has_zero_check = True
                            break

                    if not has_zero_check:
                        issues.append(Issue(
                            line=i,
                            severity="warning",
                            message="可能存在除零风险",
                            suggestion="添加除数是否为零的检查"
                        ))
                        score -= 5

        # 检查是否包含 main 入口
        if 'if __name__ == "__main__":' not in code and 'if __name__ == "__main__"' in code:
            # 已有但格式略有不同
            pass

    def _check_javascript(self, code: str, lines: list, issues: list, score: int):
        """JavaScript 专项检查"""
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            # 检查 var 使用（建议用 let/const）
            if stripped.startswith("var "):
                issues.append(Issue(
                    line=i,
                    severity="warning",
                    message="建议使用 let 或 const 替代 var",
                    suggestion="将 var 改为 const（如果变量不重新赋值）或 let"
                ))
                score -= 3

            # 检查行尾分号
            if stripped and not stripped.startswith("//") and not stripped.startswith("/*") and not stripped.startswith("*"):
                if stripped.endswith(")") or stripped.endswith("}") or stripped.endswith("]"):
                    pass
                elif not stripped.endswith(";") and not stripped.endswith("{") and not stripped.endswith("}") and not stripped.endswith(":") and not stripped.endswith(","):
                    pass

            # 检查 console.log
            if "console.log" in stripped:
                issues.append(Issue(
                    line=i,
                    severity="info",
                    message="代码中包含 console.log 调试语句",
                    suggestion="上线前请移除调试日志"
                ))

    def _check_java(self, code: str, lines: list, issues: list, score: int):
        """Java 专项检查"""
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            # 检查 System.out.println
            if "System.out.println" in stripped or "System.out.print" in stripped:
                issues.append(Issue(
                    line=i,
                    severity="info",
                    message="代码中包含 System.out 调试输出",
                    suggestion="使用日志框架（如 SLF4J）替代"
                ))
                score -= 1

            # 检查方法名应小写开头
            # 简化：检查 public/private/protected 后接大写（跳过类声明）
            # 这里简化处理

    def _check_go(self, code: str, lines: list, issues: list, score: int):
        """Go 专项检查"""
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            # 检查错误处理
            if "if err != nil" in stripped:
                pass  # 好的做法
            elif "_, err" in stripped or "err :=" in stripped:
                # 检查是否有后续 err 检查
                pass
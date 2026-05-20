# Changelog

本文件记录本技能的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [1.3.0] - 2025-01-19
### Added
- 新增 `score` 评分字段（0-100），根据问题数量和严重程度自动扣分
- 新增审查总结 `summary` 字段，自动汇总 error/warning/info 数量并给出质量评级

### Changed
- 输出结构从纯问题列表改为 `{issues, summary, score}` 三元组，调用方需适配新字段

## [1.2.0] - 2025-01-16
### Added
- JavaScript 专项检查：检测 `var` 声明（建议改用 let/const）、`console.log` 调试残留
- Java 专项检查：检测 `System.out.println` 调试残留，提示改用 SLF4J
- Go 专项检查：框架就位，检测 `err` 变量创建后的处理模式
- `language` 参数新增 `javascript`、`java`、`go` 支持

### Changed
- 未知语言不再报错，降级为 info 提示并仅执行通用检查

## [1.1.0] - 2025-01-14
### Added
- Python 专项检查：模块名应小写、函数名应 snake_case、类名应 CamelCase
- 潜在的除零错误检测（检查 return 语句中的除法是否有前置零值判断）

### Fixed
- 空文件不再触发后续无意义的按行检查逻辑，直接返回

## [1.0.0] - 2025-01-12
### Added
- 通用检查规则：超长行检测（>120 字符）、行尾空白检测、TODO 注释统计
- 多语言输入参数 `language`，默认 `python`
- 结构化输出 `issues` 列表，每条包含行号、严重级别、问题描述、修复建议
- 空文件检测与快捷返回

---

<!--
  ┌─────────────────────────────────────────────────────┐
  │  模仿指南：你写其他技能时，改下面 3 个地方即可       │
  ├─────────────────────────────────────────────────────┤
  │  1. 技能名 → `# Changelog` 下面的描述（可选加）     │
  │  2. 版本号 → `## [x.y.z]` 和日期                    │
  │  3. 条目   → Added / Changed / Fixed 下的内容       │
  │                                                     │
  │  不需要改的地方：                                   │
  │  - 文件头的两段声明（格式声明 + SemVer 声明）        │
  │  - 六个分类标题（Added/Changed/...）                │
  │  - Unreleased 占位区                               │
  └─────────────────────────────────────────────────────┘
-->
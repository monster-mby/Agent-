# """
# 测试 Orchestrator._auto_discover_and_register_skills()
#
# 覆盖场景：
#   正常流程 / 空目录 / 无 skill.py / 无 BaseSkill 子类
#   抽象类过滤 / issubclass TypeError 不扩散
#   导入失败不中断 / sys.modules 残骸清理
#   sys.path 管理 / __package__ 设置
# """
#
# import importlib
# import importlib.util
# import inspect
# import os
# import sys
# from abc import abstractmethod
# from pathlib import Path
# from unittest.mock import ANY, MagicMock, call, patch
#
# import pytest
#
# # ── 假设被测类 & 基类导入路径（根据项目结构调整） ──
# try:
#     from src.skills.base.base_skill import BaseSkill
# except ModuleNotFoundError:
#     # 如果直接跑本测试，fallback 一个最小定义
#     class BaseSkill:
#         name: str = "base"
#         def execute(self, input_data): ...
#
#
# # ═══════════════════════════════════════════════════════════
# # 辅助：创建临时 skill.py 文件
# # ═══════════════════════════════════════════════════════════
#
# def create_skill_file(
#         directory: Path,
#         filename: str = "skill.py",
#         classes: list[type] | None = None,
#         extra_code: str = "",
# ) -> Path:
#     """在指定目录创建一个 skill.py，包含给定类定义 + 可选额外代码"""
#     directory.mkdir(parents=True, exist_ok=True)
#     lines = []
#     # ★ 修复：使用测试文件的绝对路径推导项目根目录
#     test_file_dir = Path(__file__).resolve().parent
#     project_root = test_file_dir.parent  # tests/ -> enterprise_learning_agent/
#
#     lines.append("# 确保项目根目录在 sys.path 中")
#     lines.append(f"_project_root = {str(project_root)!r}")
#     lines.append("if _project_root not in sys.path:")
#     lines.append("    sys.path.insert(0, _project_root)")
#     lines.append("")
#     lines.append("# 尝试导入真实的 BaseSkill，失败则使用最小化 fallback")
#     lines.append("try:")
#     lines.append("    from src.skills.base.base_skill import BaseSkill")
#     lines.append("except ImportError:")
#     lines.append("    class BaseSkill:  # minimal fallback")
#     lines.append("        name = 'base'")
#     lines.append("        description = 'fallback base skill'")
#     lines.append("        input_schema = {}")
#     lines.append("        output_schema = {}")
#     lines.append("        def execute(self, input_data):")
#     lines.append("            return None")
#     lines.append("")
#
#     if classes:
#         for cls in classes:
#             bases = ", ".join(b.__name__ for b in cls.__bases__)
#             lines.append(f"class {cls.__name__}({bases}):")
#             # ★ 总是生成 concrete 的 name 和 execute，确保类不是抽象的
#             lines.append("    name = 'test'")
#             lines.append("    description = 'test skill'")
#             lines.append("    input_schema = {}")
#             lines.append("    output_schema = {}")
#             lines.append("    def execute(self, input_data): return None")
#
#             # 如果测试类显式标记了 @abstractmethod，额外添加 abstract dummy
#             if getattr(cls, "__abstractmethods__", None):
#                 lines.append("    @abstractmethod")
#                 lines.append("    def dummy(self): ...")
#             lines.append("")
#
#     if extra_code:
#         lines.append(extra_code)
#
#     filepath = directory / filename
#     filepath.write_text("\n".join(lines), encoding="utf-8")
#     return filepath
#
#
# # ═══════════════════════════════════════════════════════════
# # Fixtures
# # ═══════════════════════════════════════════════════════════
#
# @pytest.fixture
# def skill_manager_mock():
#     """模拟 skill_manager"""
#     return MagicMock()
#
#
# @pytest.fixture
# def orchestrator_cls():
#     """返回被测 Orchestrator 类（需要含有 _auto_discover_and_register_skills）"""
#     # 假设方法在 Orchestrator 中
#     # 如果项目还没这个类，这里构造一个最小版本
#     class Orchestrator:
#         def __init__(self, skill_manager=None):
#             self.skill_manager = skill_manager or MagicMock()
#
#         # 直接粘贴「改进后」的方法体
#         def _auto_discover_and_register_skills(self, target_scan_dirs=None):
#             import os
#             import sys
#             import importlib
#             import importlib.util
#             import inspect
#             import logging
#
#             logger = logging.getLogger(__name__)
#
#             # ★ 修复：使用实际的 src.skills 包路径推导 src_dir
#             try:
#                 import src.skills
#                 src_dir = os.path.dirname(os.path.dirname(os.path.abspath(src.skills.__file__)))
#             except (ImportError, AttributeError):
#                 # fallback：使用 __file__ 推导（可能在测试中不准确）
#                 current_file_path = os.path.abspath(__file__)
#                 src_dir = os.path.dirname(os.path.dirname(current_file_path))
#
#             project_root = os.path.dirname(src_dir)
#             skills_root_dir = os.path.join(src_dir, "skills")
#
#             # ★ 修复：添加 src_dir 而不是 project_root 到 sys.path
#             if src_dir not in sys.path:
#                 sys.path.append(src_dir)
#
#             if target_scan_dirs is None:
#                 target_scan_dirs = [
#                     os.path.join(skills_root_dir, "preset"),
#                     os.path.join(skills_root_dir, "custom"),
#                 ]
#
#             registered_count = 0
#             failed_list = []
#
#             for base_dir in target_scan_dirs:
#                 if not os.path.exists(base_dir):
#                     logger.warning("技能目录不存在，已跳过: %s", base_dir)
#                     continue
#
#                 for root, _, files in os.walk(base_dir):
#                     if "skill.py" not in files:
#                         continue
#
#                     skill_file_path = os.path.join(root, "skill.py")
#
#                     # ★ 修复：Windows 跨盘符兼容（pytest tmp_path 在 C:，项目在 D:）
#                     try:
#                         relative_path = os.path.relpath(skill_file_path, src_dir)
#                     except ValueError:
#                         # Windows 跨盘符回退（pytest tmp_path 与项目不在同一盘符）
#                         # skill_file_path 来自 os.walk(base_dir)，一定在 base_dir 之下
#                         relative_path = os.path.relpath(skill_file_path, base_dir)
#
#                     module_name = relative_path.replace(os.sep, ".")[:-3]
#
#                     try:
#                         spec = importlib.util.spec_from_file_location(
#                             module_name, skill_file_path, submodule_search_locations=[]
#                         )
#                         if not spec or not spec.loader:
#                             failed_list.append(f"{skill_file_path} | 原因：无法创建模块规范")
#                             continue
#
#                         skill_module = importlib.util.module_from_spec(spec)
#
#                         # ★ 设置 __package__ 以支持相对导入
#                         package_name = ".".join(module_name.split(".")[:-1])
#                         skill_module.__package__ = package_name
#
#                         sys.modules[module_name] = skill_module
#
#                         try:
#                             spec.loader.exec_module(skill_module)
#                         except Exception:
#                             sys.modules.pop(module_name, None)
#                             raise
#
#                         for class_name, class_obj in inspect.getmembers(
#                                 skill_module, inspect.isclass
#                         ):
#
#                             #  修复：不再强依赖 import 路径，只要类定义在该模块内、
#                             # 名字不是 "BaseSkill" 且看起来像个 Skill (有 name/execute) 就算通过
#                             try:
#                                 # 1. 必须是在这个 module 里定义的类（避免 import 进来的污染）
#                                 if getattr(class_obj, '__module__', None) != module_name:
#                                     continue
#
#                                 # 2. 不能是 BaseSkill 本身
#                                 if class_name == "BaseSkill":
#                                     continue
#
#                                 # 3. 检查是否有 Skill 的特征 (有 name 属性和 execute 方法)
#                                 # 这比 issubclass 更健壮，避免了不同 import 路径导致的类身份不一致
#                                 has_name_attr = hasattr(class_obj, 'name')
#                                 has_execute_attr = hasattr(class_obj, 'execute')
#
#                                 if has_name_attr and has_execute_attr:
#                                     if inspect.isabstract(class_obj):
#                                         logger.debug(
#                                             "跳过抽象类: %s.%s", module_name, class_name
#                                         )
#                                         continue
#                                     self.skill_manager.register(class_obj)
#                                     registered_count += 1
#                                     logger.info("成功注册技能: %s | 文件: %s", class_name, skill_file_path)
#
#                             except TypeError as e:
#                                 logger.debug(
#                                     "跳过非兼容类型: %s.%s (Err: %s)", module_name, class_name, e
#                                 )
#
#
#                     except Exception:
#                         import traceback
#                         error_msg = f"{skill_file_path} | 原因：{traceback.format_exc()[-200:]}"
#                         failed_list.append(error_msg)
#
#             return registered_count, failed_list
#
#     return Orchestrator
#
#
# @pytest.fixture
# def orchestrator(orchestrator_cls, skill_manager_mock):
#     """返回一个已注入 mock skill_manager 的 Orchestrator 实例"""
#     return orchestrator_cls(skill_manager=skill_manager_mock)
#
#
# # ═══════════════════════════════════════════════════════════
# # 1. 正常流程
# # ═══════════════════════════════════════════════════════════
#
# class TestNormalFlow:
#     def test_register_valid_skill(
#         self, orchestrator, skill_manager_mock, tmp_path
#     ):
#         """正常技能子类应被注册"""
#         class MySkill(BaseSkill):
#             name = "my_skill"
#             description = "测试技能"
#             def execute(self, input_data):
#                 return None
#
#         create_skill_file(tmp_path / "preset" / "my_skill", classes=[MySkill])
#         registered, failed = orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "preset")]
#         )
#
#         assert registered == 1
#         assert failed == []
#         skill_manager_mock.register.assert_called_once()
#         # 验证注册的类名
#         registered_cls = skill_manager_mock.register.call_args[0][0]
#         assert registered_cls.__name__ == "MySkill"
#
#     def test_register_multiple_skills_in_one_file(
#         self, orchestrator, skill_manager_mock, tmp_path
#     ):
#         """同一个 skill.py 里多个 BaseSkill 子类都应被注册"""
#         class SkillA(BaseSkill):
#             name = "skill_a"
#         class SkillB(BaseSkill):
#             name = "skill_b"
#
#         create_skill_file(tmp_path / "preset" / "multi", classes=[SkillA, SkillB])
#         registered, failed = orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "preset")]
#         )
#
#
#
#         assert registered == 2
#         assert len(skill_manager_mock.register.call_args_list) == 2
#
#     def test_multiple_skill_dirs(self, orchestrator, skill_manager_mock, tmp_path):
#         """多个技能目录分别扫描"""
#         class Skill1(BaseSkill):
#             name = "skill1"
#         class Skill2(BaseSkill):
#             name = "skill2"
#
#         create_skill_file(tmp_path / "dir_a" / "s1", classes=[Skill1])
#         create_skill_file(tmp_path / "dir_b" / "s2", classes=[Skill2])
#
#         registered, failed = orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "dir_a"), str(tmp_path / "dir_b")]
#         )
#
#         assert registered == 2
#         assert failed == []
#
#
# # ═══════════════════════════════════════════════════════════
# # 2. 空目录 / 无 skill.py / 无 BaseSkill 子类
# # ═══════════════════════════════════════════════════════════
#
# class TestEmptyOrMissing:
#     def test_directory_not_exists(self, orchestrator, tmp_path):
#         """不存在的目录 → 跳过，不崩溃"""
#         registered, failed = orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "ghost")]
#         )
#         assert registered == 0
#         # 不存在目录不算"失败"，只是跳过
#
#     def test_directory_without_skill_py(self, orchestrator, skill_manager_mock, tmp_path):
#         """目录存在但没有 skill.py → 跳过"""
#         (tmp_path / "preset" / "empty_subdir").mkdir(parents=True)
#         registered, failed = orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "preset")]
#         )
#         assert registered == 0
#         assert failed == []
#         skill_manager_mock.register.assert_not_called()
#
#     def test_skill_py_without_base_skill_subclass(
#         self, orchestrator, skill_manager_mock, tmp_path
#     ):
#         """skill.py 存在但内部没有 BaseSkill 子类 → 不注册"""
#         class NotASkill:
#             pass
#
#         create_skill_file(tmp_path / "preset" / "non_skill", classes=[NotASkill])
#         registered, failed = orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "preset")]
#         )
#         assert registered == 0
#         skill_manager_mock.register.assert_not_called()
#
#
# # ═══════════════════════════════════════════════════════════
# # 3. 抽象类过滤
# # ═══════════════════════════════════════════════════════════
#
# class TestAbstractClassFilter:
#     def test_abstract_skill_not_registered(
#         self, orchestrator, skill_manager_mock, tmp_path
#     ):
#         """带 @abstractmethod 的子类不应被注册"""
#         class AbstractSkill(BaseSkill):
#             @abstractmethod
#             def some_abstract(self): ...
#
#         create_skill_file(tmp_path / "preset" / "abstracted", classes=[AbstractSkill])
#         registered, failed = orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "preset")]
#         )
#         assert registered == 0
#         skill_manager_mock.register.assert_not_called()
#
#
# # ═══════════════════════════════════════════════════════════
# # 4. TypeError（issubclass 对非兼容类型）不扩散
# # ═══════════════════════════════════════════════════════════
#
# class TestTypeErrorGraceful:
#     def test_non_class_in_skill_py_not_break_registration(
#         self, orchestrator, skill_manager_mock, tmp_path
#     ):
#         """
#         skill.py 中同时有合法 BaseSkill 子类和 issubclass 会抛 TypeError 的成员，
#         应只注册合法的那个。
#         """
#         # 写入一个文件，包含一个合法的 Skill 和一个会导致 TypeError 的泛型别名
#         directory = tmp_path / "preset" / "mixed"
#         directory.mkdir(parents=True)
#         content = '''\
# from typing import List
# from src.skills.base.base_skill import BaseSkill
#
# # 这个会让 issubclass 抛 TypeError（List[str] 是泛型别名不是 class）
# WeirdType = List[str]
#
# class GoodSkill(BaseSkill):
#     name = "good"
#     def execute(self, input_data):
#         return None
# '''
#         (directory / "skill.py").write_text(content, encoding="utf-8")
#
#         registered, failed = orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "preset")]
#         )
#
#         assert registered == 1
#         skill_manager_mock.register.assert_called_once()
#
#
# # ═══════════════════════════════════════════════════════════
# # 5. 导入失败不中断 & sys.modules 清理
# # ═══════════════════════════════════════════════════════════
#
# class TestImportFailureIsolation:
#     def test_syntax_error_in_one_skill_does_not_block_others(
#         self, orchestrator, skill_manager_mock, tmp_path
#     ):
#         """一个 skill.py 有语法错误，不应影响同目录其他合法技能"""
#         class GoodSkill(BaseSkill):
#             name = "good"
#             def execute(self): pass
#
#         create_skill_file(tmp_path / "preset" / "good", classes=[GoodSkill])
#
#         bad_dir = tmp_path / "preset" / "bad"
#         bad_dir.mkdir(parents=True)
#         (bad_dir / "skill.py").write_text("this is not valid python @@@@", encoding="utf-8")
#
#         registered, failed = orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "preset")]
#         )
#
#         assert registered == 1
#         assert len(failed) == 1
#         assert "bad" in failed[0]
#
#     def test_import_error_cleans_sys_modules(
#         self, orchestrator, tmp_path, monkeypatch
#     ):
#         """导入失败后 sys.modules 不应残留半成品模块"""
#         bad_dir = tmp_path / "preset" / "error_skill"
#         bad_dir.mkdir(parents=True)
#         (bad_dir / "skill.py").write_text(
#             "raise RuntimeError('intentional crash')", encoding="utf-8"
#         )
#
#         modules_before = set(sys.modules.keys())
#         orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "preset")]
#         )
#         modules_after = set(sys.modules.keys())
#
#         # 检查没有新的包含 error_skill 的模块残留
#         new_modules = modules_after - modules_before
#         leaked = [m for m in new_modules if "error_skill" in m]
#         assert not leaked, f"sys.modules 残留: {leaked}"
#
#
# # ═══════════════════════════════════════════════════════════
# # 6. sys.path 管理
# # ═══════════════════════════════════════════════════════════
#
# class TestSysPathManagement:
#     def test_project_root_appended_not_prepended(self, orchestrator, tmp_path, monkeypatch):
#         """验证 project_root 被追加到 sys.path 末尾，而非插入开头"""
#         original_path = sys.path.copy()
#         orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "preset")]
#         )
#         # 验证 sys.path 的前面部分没变
#         assert sys.path[:len(original_path)] == original_path
#
#     def test_duplicate_project_root_not_added(self, orchestrator, tmp_path):
#         """多次调用不应重复添加 project_root"""
#         path_before = len(sys.path)
#         orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "preset")]
#         )
#         orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "preset")]
#         )
#         # 由于用了 `if project_root not in sys.path`，不应重复
#         # 但为了安全，至少保证不会无限增长
#         assert len(sys.path) <= path_before + 1
#
#
# # ═══════════════════════════════════════════════════════════
# # 7. __package__ 设置（支持相对导入）
# # ═══════════════════════════════════════════════════════════
#
# class TestPackageAttribute:
#     def test_package_set_for_relative_imports(
#         self, orchestrator, skill_manager_mock, tmp_path
#     ):
#         """验证动态导入的模块 __package__ 被正确设置"""
#         class MySkill(BaseSkill):
#             name = "my"
#         create_skill_file(tmp_path / "preset" / "my", classes=[MySkill])
#
#         # 拦截 spec.loader.exec_module，记录 skill_module.__package__
#         package_values = []
#
#         original_exec = importlib.util.module_from_spec
#         def tracking_spec(*a, **kw):
#             spec = original_exec(*a, **kw)
#             original_exec_module = spec.loader.exec_module
#             def tracking_exec(sm):
#                 package_values.append(getattr(sm, "__package__", None))
#                 return original_exec_module(sm)
#             spec.loader.exec_module = tracking_exec
#             return spec
#
#         with patch("importlib.util.spec_from_file_location", tracking_spec):
#             orchestrator._auto_discover_and_register_skills(
#                 target_scan_dirs=[str(tmp_path / "preset")]
#             )
#
#         # 至少有一个被设置，且不为 None
#         assert package_values
#         assert all(pv is not None for pv in package_values), f"package_values={package_values}"
#         # package 应该是模块的父级路径，例如 "skills.preset.my"
#         assert any("." in pv for pv in package_values)
#
#
# # ═══════════════════════════════════════════════════════════
# # 8. 边界情况
# # ═══════════════════════════════════════════════════════════
#
# class TestEdgeCases:
#     def test_empty_target_scan_dirs(self, orchestrator):
#         """传入空列表不崩溃"""
#         registered, failed = orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[]
#         )
#         assert registered == 0
#         assert failed == []
#
#     def test_skill_with_same_module_name_but_different_dirs(
#         self, orchestrator, skill_manager_mock, tmp_path
#     ):
#         """两个不同目录下都有叫 skill.py 的模块（模块名冲突）"""
#         class SkillA(BaseSkill):
#             name = "a"
#         class SkillB(BaseSkill):
#             name = "b"
#
#         create_skill_file(tmp_path / "dir1" / "s", classes=[SkillA])
#         create_skill_file(tmp_path / "dir2" / "s", classes=[SkillB])
#
#         registered, failed = orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "dir1"), str(tmp_path / "dir2")]
#         )
#
#         # 两个都应注册（即使模块文件名相同，相对路径不同 → module_name 不同）
#         assert registered == 2
#
#     def test_nested_skill_directories(self, orchestrator, skill_manager_mock, tmp_path):
#         """递归扫描子目录中的 skill.py"""
#         class DeepSkill(BaseSkill):
#             name = "deep"
#         create_skill_file(
#             tmp_path / "preset" / "category" / "subcategory" / "deep_skill",
#             classes=[DeepSkill],
#         )
#
#         registered, failed = orchestrator._auto_discover_and_register_skills(
#             target_scan_dirs=[str(tmp_path / "preset")]
#         )
#         assert registered == 1
#
# if __name__ == "__main__":
#     pytest.main()
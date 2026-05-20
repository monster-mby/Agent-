"""
DocumentLoaderSkill 单元测试
覆盖：正常加载 / 自动检测 / 文件不存在 / 超大文件 / 编码探测
"""

import tempfile
from pathlib import Path

import pytest

from src.skills.custom.rag_skills.document_loader.skill import (
    DocumentLoaderSkill,
    DocumentLoaderInput,
    DocumentLoaderOutput,
)


@pytest.fixture
def skill():
    return DocumentLoaderSkill()


@pytest.fixture
def sample_txt():
    """创建临时 txt 文件"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Hello 世界\n这是第二行\n")
        return Path(f.name)


@pytest.fixture
def sample_md():
    """创建临时 md 文件"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# 标题\n\n内容段落\n")
        return Path(f.name)


@pytest.fixture
def sample_json():
    """创建临时 json 文件"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write('{"key": "value"}')
        return Path(f.name)


# ------------------------------------------------------------------
# 正常场景
# ------------------------------------------------------------------

class TestNormalLoad:
    def test_load_txt(self, skill, sample_txt):
        inp = DocumentLoaderInput(file_path=str(sample_txt))
        out = skill.execute(inp)
        assert out.get("error") == ""  # ✅ 修改：字典访问
        assert "Hello 世界" in out.get("content")  # ✅ 修改：字典访问
        assert out.get("char_count") > 0  # ✅ 修改：字典访问
        assert out.get("line_count") >= 2  # ✅ 修改：字典访问
        assert out.get("file_type") == "plaintext"  # ✅ 修改：字典访问
        assert out.get("extension") == ".txt"  # ✅ 修改：字典访问
        assert out.get("file_size_bytes") > 0  # ✅ 修改：字典访问

    def test_load_md(self, skill, sample_md):
        inp = DocumentLoaderInput(file_path=str(sample_md))
        out = skill.execute(inp)
        assert out.get("error") == ""  # ✅ 修改：字典访问
        assert out.get("file_type") == "markdown"  # ✅ 修改：字典访问
        assert "# 标题" in out.get("content")  # ✅ 修改：字典访问
        assert out.get("extension") == ".md"  # ✅ 修改：字典访问

    def test_load_json(self, skill, sample_json):
        inp = DocumentLoaderInput(file_path=str(sample_json))
        out = skill.execute(inp)
        assert out.get("error") == ""  # ✅ 修改：字典访问
        assert out.get("file_type") == "json"  # ✅ 修改：字典访问
        assert '"key"' in out.get("content")  # ✅ 修改：字典访问

    def test_all_fields_present(self, skill, sample_txt):
        """验证输出字段完整"""
        inp = DocumentLoaderInput(file_path=str(sample_txt))
        out = skill.execute(inp)
        # ✅ 修改：out 已经是字典，不需要 model_dump()
        for field in ["content", "char_count", "line_count", "file_type",
                       "encoding_used", "file_size_bytes", "extension", "error"]:
            assert field in out, f"缺少字段: {field}"


# ------------------------------------------------------------------
# 自动检测
# ------------------------------------------------------------------

class TestAutoDetection:
    def test_auto_file_type_txt(self, skill, sample_txt):
        inp = DocumentLoaderInput(file_path=str(sample_txt), file_type="auto")
        out = skill.execute(inp)
        assert out.get("file_type") == "plaintext"  # ✅ 修改：字典访问

    def test_force_file_type(self, skill, sample_txt):
        """强制指定 file_type 应生效"""
        inp = DocumentLoaderInput(file_path=str(sample_txt), file_type="markdown")
        out = skill.execute(inp)
        assert out.get("file_type") == "markdown"  # ✅ 修改：字典访问

    def test_auto_encoding_utf8(self, skill, sample_txt):
        inp = DocumentLoaderInput(file_path=str(sample_txt), encoding="auto")
        out = skill.execute(inp)
        assert out.get("encoding_used") in ("utf-8", "gbk", "gb2312", "latin-1")  # ✅ 修改：字典访问
        assert out.get("error") == ""  # ✅ 修改：字典访问


# ------------------------------------------------------------------
# 错误场景
# ------------------------------------------------------------------

class TestErrorHandling:
    def test_file_not_found(self, skill):
        inp = DocumentLoaderInput(file_path="/nonexistent/file.txt")
        out = skill.execute(inp)
        assert out.get("error") != ""  # ✅ 修改：字典访问
        assert "不存在" in out.get("error")  # ✅ 修改：字典访问
        assert out.get("content") == ""  # ✅ 修改：字典访问

    def test_directory_passed(self, skill, tmp_path):
        """传入目录路径应报错"""
        inp = DocumentLoaderInput(file_path=str(tmp_path))
        out = skill.execute(inp)
        assert out.get("error") != ""  # ✅ 修改：字典访问
        assert "目录而非文件" in out.get("error")  # ✅ 修改：字典访问

    def test_unsupported_extension(self, skill):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False, encoding="utf-8") as f:
            f.write("data")
            path = Path(f.name)
        inp = DocumentLoaderInput(file_path=str(path))
        out = skill.execute(inp)
        assert out.get("error") != ""  # ✅ 修改：字典访问
        assert "不支持" in out.get("error")  # ✅ 修改：字典访问

    def test_max_chars_exceeded(self, skill, sample_txt):
        inp = DocumentLoaderInput(file_path=str(sample_txt), max_chars=5)
        out = skill.execute(inp)
        assert out.get("error") != ""  # ✅ 修改：字典访问
        assert "超过上限" in out.get("error")  # ✅ 修改：字典访问


# ------------------------------------------------------------------
# 边界场景
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_file(self, skill):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("")
            path = Path(f.name)
        inp = DocumentLoaderInput(file_path=str(path))
        out = skill.execute(inp)
        assert out.get("error") == ""  # ✅ 修改：字典访问
        assert out.get("content") == ""  # ✅ 修改：字典访问
        assert out.get("char_count") == 0  # ✅ 修改：字典访问
        assert out.get("line_count") == 0  # ✅ 修改：字典访问

    def test_gbk_encoded_file(self, skill):
        """GBK 编码文件应被正确读取"""
        content = "中文内容测试\n第二行"
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            f.write(content.encode("gbk"))
            path = Path(f.name)
        inp = DocumentLoaderInput(file_path=str(path), encoding="auto")
        out = skill.execute(inp)
        assert out.get("error") == ""  # ✅ 修改：字典访问
        assert "中文内容测试" in out.get("content") or out.get("encoding_used") in ("gbk", "gb2312")  # ✅ 修改：字典访问

    def test_relative_path(self, skill, sample_txt, monkeypatch):
        """相对路径应能正确解析"""
        # 临时切到文件所在目录
        monkeypatch.chdir(sample_txt.parent)
        rel_path = sample_txt.name
        inp = DocumentLoaderInput(file_path=rel_path)
        out = skill.execute(inp)
        assert out.get("error") == ""  # ✅ 修改：字典访问
        assert out.get("char_count") > 0  # ✅ 修改：字典访问


# ------------------------------------------------------------------
# 清理
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_temp_files(sample_txt, sample_md, sample_json):
    """测试后清理临时文件"""
    yield
    for p in [sample_txt, sample_md, sample_json]:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

if __name__ == "__main__":
    pytest.main()

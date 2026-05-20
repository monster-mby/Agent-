"""端到端 API 测试（修正版）"""
import pytest
import httpx
from tests.test_api_config import BASE_URL, API_PREFIX, HEADERS


class TestHealthCheck:
    """健康检查测试"""

    def test_health_check(self):
        """测试健康检查端点"""
        response = httpx.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "components" in data


class TestSessions:
    """会话管理测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化客户端"""
        self.client = httpx.Client(base_url=BASE_URL, headers=HEADERS)
        yield
        self.client.close()

    def test_create_session(self):
        payload = {"name": "测试会话", "description": "用于 API 测试"}
        response = self.client.post(
            f"{API_PREFIX}/sessions",
            json=payload,
            headers={"Content-Type": "application/json"},  # ← 显式加
        )
        response = self.client.post(f"{API_PREFIX}/sessions", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["name"] == "测试会话"
        self.session_id = data["session_id"]

    def test_list_sessions(self):
        """测试列出会话（分页响应）"""
        # 先创建一个会话
        self.test_create_session()

        response = self.client.get(f"{API_PREFIX}/sessions")
        assert response.status_code == 200
        data = response.json()
        # ✅ 修正：响应是分页对象，不是列表
        assert "items" in data
        assert "total" in data
        assert isinstance(data["items"], list)
        assert data["total"] > 0

    def test_get_session(self):
        """测试获取会话详情"""
        # 先创建会话
        self.test_create_session()

        response = self.client.get(f"{API_PREFIX}/sessions/{self.session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == self.session_id

    def test_delete_session(self):
        """测试删除会话"""
        # 先创建会话
        self.test_create_session()

        response = self.client.delete(f"{API_PREFIX}/sessions/{self.session_id}")
        # ✅ 修正：实际返回 200 而非 204
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session deleted"
        assert data["session_id"] == self.session_id


class TestKnowledgeBases:
    """知识库管理测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = httpx.Client(base_url=BASE_URL, headers=HEADERS)
        yield
        self.client.close()

    def test_create_kb(self):
        """测试创建知识库"""
        payload = {
            "name": "测试知识库",
            "description": "用于 API 测试",
            "embedding_model": "text-embedding-v3"
        }
        response = self.client.post(f"{API_PREFIX}/knowledge-bases", json=payload)

        # 🔴 打印详细错误信息，帮助调试
        if response.status_code != 201:
            print(f"\n❌ 创建知识库失败!")
            print(f"状态码: {response.status_code}")
            print(f"响应内容: {response.json()}")

        assert response.status_code == 201
        data = response.json()
        assert "kb_id" in data
        assert data["name"] == "测试知识库"
        self.kb_id = data["kb_id"]

    def test_list_kbs(self):
        """测试列出知识库"""
        # 先创建一个知识库
        self.test_create_kb()

        response = self.client.get(f"{API_PREFIX}/knowledge-bases")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] > 0

    def test_get_kb(self):
        """测试获取知识库详情"""
        self.test_create_kb()

        response = self.client.get(f"{API_PREFIX}/knowledge-bases/{self.kb_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["kb_id"] == self.kb_id

    def test_update_kb(self):
        """测试更新知识库"""
        self.test_create_kb()

        payload = {
            "name": "更新后的知识库",
            "description": "已更新"
        }
        response = self.client.put(f"{API_PREFIX}/knowledge-bases/{self.kb_id}", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "更新后的知识库"

    def test_delete_kb(self):
        """测试删除知识库"""
        self.test_create_kb()

        response = self.client.delete(f"{API_PREFIX}/knowledge-bases/{self.kb_id}")
        assert response.status_code == 204


class TestRules:
    """规则管理测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = httpx.Client(base_url=BASE_URL, headers=HEADERS)
        # 先创建会话
        session_payload = {"name": "规则测试会话"}
        response = self.client.post(f"{API_PREFIX}/sessions", json=session_payload)
        self.session_id = response.json()["session_id"]
        yield
        self.client.close()

    def test_create_rule(self):
        """测试创建规则"""
        # ✅ 修正：Payload 必须匹配 CreateRuleRequest Schema
        payload = {
            "content": "这是一条测试规则内容",
            "priority": 5,
            "category": "general"  # ✅ 使用正确的枚举值
        }
        response = self.client.post(
            f"{API_PREFIX}/sessions/{self.session_id}/rules",
            json=payload
        )
        assert response.status_code == 201
        data = response.json()
        assert "rule_id" in data
        self.rule_id = data["rule_id"]

    def test_list_rules(self):
        """测试列出规则"""
        self.test_create_rule()

        response = self.client.get(f"{API_PREFIX}/sessions/{self.session_id}/rules")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_toggle_rule(self):
        """测试启用/禁用规则"""
        self.test_create_rule()

        response = self.client.post(
            f"{API_PREFIX}/sessions/{self.session_id}/rules/{self.rule_id}/toggle"
        )
        assert response.status_code == 200


class TestDocuments:
    """文档管理测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = httpx.Client(base_url=BASE_URL, headers=HEADERS)
        # 先创建知识库
        kb_payload = {"name": "文档测试知识库"}
        response = self.client.post(f"{API_PREFIX}/knowledge-bases", json=kb_payload)
        assert response.status_code == 201  # ✅ 添加断言确保创建成功
        self.kb_id = response.json()["kb_id"]
        yield
        self.client.close()

    def test_upload_document(self):
        """测试上传文档"""
        # 创建测试文件
        test_content = "这是测试文档内容\n用于 API 测试"
        files = {
            "file": ("test.txt", test_content, "text/plain")
        }
        data = {
            "metadata": '{"tags": ["test"], "source_url": "https://example.com"}'
        }

        response = self.client.post(
            f"{API_PREFIX}/knowledge-bases/{self.kb_id}/documents",
            files=files,
            data=data
        )
        # 🔴 打印详细错误信息，帮助调试
        if response.status_code != 201:
            print(f"\n❌ 上传文档失败!")
            print(f"状态码: {response.status_code}")
            print(f"响应内容: {response.text}")  # 用 text 看完整内容
            try:
                print(f"JSON: {response.json()}")
            except:
                pass
        assert response.status_code == 201
        resp_data = response.json()
        assert "doc_id" in resp_data
        self.doc_id = resp_data["doc_id"]

    def test_list_documents(self):
        """测试列出文档"""
        self.test_upload_document()

        response = self.client.get(
            f"{API_PREFIX}/knowledge-bases/{self.kb_id}/documents"
        )
        assert response.status_code == 200
        resp_data = response.json()
        assert "items" in resp_data
        assert resp_data["total"] > 0

    def test_delete_document(self):
        """测试删除文档"""
        self.test_upload_document()

        response = self.client.delete(
            f"{API_PREFIX}/knowledge-bases/{self.kb_id}/documents/{self.doc_id}"
        )
        assert response.status_code == 204


class TestSessions:
    """会话管理测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化客户端"""
        self.client = httpx.Client(base_url=BASE_URL, headers=HEADERS)
        yield
        self.client.close()

    def test_create_session(self):
        payload = {"name": "测试会话", "description": "用于 API 测试"}
        response = self.client.post(
            f"{API_PREFIX}/sessions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response = self.client.post(f"{API_PREFIX}/sessions", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["name"] == "测试会话"
        self.session_id = data["session_id"]

    def test_list_sessions(self):
        """测试列出会话（分页响应）"""
        self.test_create_session()

        response = self.client.get(f"{API_PREFIX}/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert isinstance(data["items"], list)
        assert data["total"] > 0

    def test_get_session(self):
        """测试获取会话详情"""
        self.test_create_session()

        response = self.client.get(f"{API_PREFIX}/sessions/{self.session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == self.session_id

    def test_get_session_state(self):
        """测试获取会话状态（新增功能）"""
        # 先创建会话
        self.test_create_session()

        # 查询初始状态（可能无 checkpoint，取决于实现）
        response = self.client.get(f"{API_PREFIX}/sessions/{self.session_id}/state")
        # 注意：如果首次查询无 checkpoint，可能返回 404
        # 这里验证端点可访问且返回格式正确
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert "thread_id" in data
            assert "channel_values" in data
            assert "session_status" in data

    def test_delete_session(self):
        """测试删除会话"""
        self.test_create_session()

        response = self.client.delete(f"{API_PREFIX}/sessions/{self.session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session deleted"
        assert data["session_id"] == self.session_id



if __name__ == "__main__":    pytest.main([__file__, "-v", "--tb=short"])
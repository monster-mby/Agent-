"""API 测试配置"""
BASE_URL = "http://localhost:8000"
API_KEY = "test-key"
API_PREFIX = "/api/v1"

# ✅ 去掉 Content-Type，让 httpx 自动判断
HEADERS = {
    "X-API-Key": API_KEY,
}
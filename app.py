"""
企业学习助手 - Streamlit 全功能前端（优化版）
应用了以下优化：
- 流式聊天（SSE）
- 数据缓存（@st.cache_data）
- 列表分页
- 公共组件（确认删除、安全API调用等）
- 完整状态初始化与清理
- 用户可配置API地址/密钥
- 向量搜索测试（基础实现）
- 性能与可维护性大幅提升
"""
import codecs

import streamlit as st
import requests
import json
import os          # ← 补这一行
import re
from datetime import datetime
from typing import List, Optional, Dict, Any

# ==================== 持久化配置 ====================
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("API_KEY", "")
if "page_size" not in st.session_state:
    st.session_state.page_size = 10

API_BASE_URL = st.session_state.api_base_url
API_KEY = st.session_state.api_key

st.set_page_config(
    page_title="企业学习助手（优化版）",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义样式 ====================
st.markdown("""
<style>
    .main { padding: 1rem; }
    .stButton > button { width: 100%; }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    .error-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        margin: 10px 0;
    }
    .info-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        margin: 10px 0;
    }
    .chat-user {
        padding: 10px; margin: 10px 0; background-color: #e3f2fd; border-radius: 10px;
        border-left: 4px solid #2196f3;
    }
    .chat-assistant {
        padding: 10px; margin: 10px 0; background-color: #f5f5f5; border-radius: 10px;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Session State 完整初始化 ====================
st.session_state.setdefault("current_session_id", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("selected_kb_id", None)
st.session_state.setdefault("show_create_kb", False)
st.session_state.setdefault("edit_kb_id", None)
st.session_state.setdefault("graph_state", None)
st.session_state.setdefault("current_page", 1)  # 通用分页页号
st.session_state.setdefault("page_cache", {})   # 缓存分页数据
st.session_state.setdefault("status_text", "")  # ✅ 新增：中间状态文本
st.session_state.setdefault("history_loaded", False)  # ✅ 新增：是否已加载历史

# 重置函数：切换会话时清空消息
def reset_session():
    st.session_state.messages = []
    st.session_state.current_page = 1
    st.session_state.page_cache = {}

# ==================== 公共组件 ====================
def safe_api_call(func, error_msg="操作失败", **kwargs):
    """统一异常处理包装器"""
    try:
        return func(**kwargs)
    except Exception as e:
        st.error(f"{error_msg}: {str(e)}")
        return None

def confirm_delete_dialog(obj_name: str) -> bool:
    """通用删除确认对话框"""
    return st.checkbox(f"⚠️ 确认删除 {obj_name}？", key=f"del_confirm_{obj_name}")

def pagination_controls(total: int, page_size: int, key_prefix: str) -> int:
    """返回当前页号，自动渲染翻页控件"""
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.session_state.get(f"{key_prefix}_page", 1)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ 上一页", disabled=page <= 1, key=f"{key_prefix}_prev"):
            page = max(1, page - 1)
            st.session_state[f"{key_prefix}_page"] = page
            st.rerun()
    with col2:
        st.write(f"第 {page} / {total_pages} 页（共 {total} 项）")
    with col3:
        if st.button("下一页 ➡️", disabled=page >= total_pages, key=f"{key_prefix}_next"):
            page = min(total_pages, page + 1)
            st.session_state[f"{key_prefix}_page"] = page
            st.rerun()
    return page

def sse_parser(response):
    """解析 server-sent events 流（容错版）"""
    buffer = b""
    for chunk in response.iter_content(chunk_size=1):
        if chunk:
            buffer += chunk
            try:
                text = buffer.decode("utf-8")
                buffer = b""
                while '\n\n' in text:
                    event_data, text = text.split('\n\n', 1)
                    for line in event_data.split('\n'):
                        if line.startswith('data: '):
                            yield line[6:]
            except UnicodeDecodeError:
                pass
    if buffer:
        try:
            text = buffer.decode("utf-8")
            for line in text.strip().split('\n'):
                if line.startswith('data: '):
                    yield line[6:]
        except UnicodeDecodeError:
            pass

# ==================== API 客户端（优化版） ====================
class APIClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip('/')
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    # ---------- 缓存装饰器 ----------
    @st.cache_data(ttl=60, show_spinner=False)
    def _cached_get_sessions(_self, base_url: str, headers_json: str, limit: int, offset: int):
        headers = json.loads(headers_json)
        return safe_api_call(
            lambda: requests.get(f"{base_url}/sessions", headers=headers,
                                params={"limit": limit, "offset": offset}).json()
        )
    @st.cache_data(ttl=60, show_spinner=False)
    def _cached_get_kbs(_self, base_url: str, headers_json: str, search: str | None, status: str | None, limit: int, offset: int):
        headers = json.loads(headers_json)
        params = {"limit": limit, "offset": offset}
        if search: params["search"] = search
        if status: params["status"] = status
        return safe_api_call(
            lambda: requests.get(f"{base_url}/knowledge-bases", headers=headers,
                                params=params).json()
        )
    def _get_rules(_self, base_url: str, headers_json: str, session_id: str):
        headers = json.loads(headers_json)
        return safe_api_call(
            lambda: requests.get(f"{base_url}/sessions/{session_id}/rules", headers=headers).json()
        )
    # ---------- 会话管理 ----------
    def get_sessions(self, page=1, page_size=None):
        if page_size is None: page_size = st.session_state.page_size
        offset = (page - 1) * page_size
        return self._cached_get_sessions(
            self.base_url,
            json.dumps(self.headers),
            limit=page_size,
            offset=offset
        )

    def create_session(self, name: str, kb_ids: List[str] = None):
        return safe_api_call(
            lambda: requests.post(f"{self.base_url}/sessions", headers=self.headers,
                                  json={"name": name, "knowledge_base_ids": kb_ids or []}).json(),
            error_msg="创建会话失败"
        )

    def delete_session(self, session_id: str):
        return safe_api_call(
            lambda: requests.delete(f"{self.base_url}/sessions/{session_id}", headers=self.headers),
            error_msg="删除会话失败"
        ) is not None

    def get_session_history(self, session_id: str, limit: int = 50):
        """获取会话历史消息（返回列表，非分页结构）"""
        return safe_api_call(
            lambda: requests.get(f"{self.base_url}/sessions/{session_id}/history", headers=self.headers,
                                 params={"limit": limit}).json(),
            error_msg="获取历史消息失败"
        )

    def chat_stream(self, session_id: str, query: str, kb_id: str = None):
        """流式聊天，返回生成器"""
        body = {"query": query, "stream": True}
        if kb_id:
            body["knowledge_base_ids"] = [kb_id]
        resp = requests.post(
            f"{self.base_url}/sessions/{session_id}/chat",
            headers=self.headers,
            json=body,
            stream=True
        )
        resp.raise_for_status()
        return sse_parser(resp)

    def get_session_state(self, session_id: str):
        return safe_api_call(
            lambda: requests.get(f"{self.base_url}/sessions/{session_id}/state", headers=self.headers).json()
        )

    # ---------- 知识库管理 ----------
    def get_knowledge_bases(self, page=1, page_size=None, search=None, status=None):
        if page_size is None: page_size = st.session_state.page_size
        offset = (page - 1) * page_size
        return self._cached_get_kbs(self.base_url, json.dumps(self.headers), search=search, status=status, limit=page_size, offset=offset)

    def create_knowledge_base(self, name: str, description: str = "", embedding_model: str = "text-embedding-ada-002"):
        return safe_api_call(
            lambda: requests.post(f"{self.base_url}/knowledge-bases", headers=self.headers,
                                  json={"name": name, "description": description, "embedding_model": embedding_model}).json(),
            error_msg="创建知识库失败"
        )

    def delete_knowledge_base(self, kb_id: str):
        return safe_api_call(
            lambda: requests.delete(f"{self.base_url}/knowledge-bases/{kb_id}", headers=self.headers),
            error_msg="删除知识库失败"
        ) is not None

    def get_kb_documents(self, kb_id: str, page=1, page_size=None):
        if page_size is None: page_size = st.session_state.page_size
        offset = (page - 1) * page_size
        return safe_api_call(
            lambda: requests.get(f"{self.base_url}/knowledge-bases/{kb_id}/documents", headers=self.headers,
                                 params={"limit": page_size, "offset": offset}).json(),
            error_msg="获取文档列表失败"
        )

    def upload_document(self, kb_id: str, file_obj, metadata: dict = None):
        try:
            files = {'file': (file_obj.name, file_obj, file_obj.type)}
            data = {}
            if metadata:
                data['metadata'] = json.dumps(metadata)
            # ✅ 修复：上传文件时去掉 Content-Type，让 requests 自动设为 multipart/form-data
            upload_headers = {k: v for k, v in self.headers.items() if k != "Content-Type"}
            resp = requests.post(f"{self.base_url}/knowledge-bases/{kb_id}/documents",
                                 headers=upload_headers, files=files, data=data)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            st.error(f"上传文档失败: {str(e)}")
            return None

    def delete_document(self, kb_id: str, doc_id: str):
        return safe_api_call(
            lambda: requests.delete(f"{self.base_url}/knowledge-bases/{kb_id}/documents/{doc_id}", headers=self.headers),
            error_msg="删除文档失败"
        ) is not None

    def reindex_kb(self, kb_id: str, force: bool = False):
        return safe_api_call(
            lambda: requests.post(f"{self.base_url}/knowledge-bases/{kb_id}/reindex", headers=self.headers,
                                  json={"force": force}).json(),
            error_msg="索引任务提交失败"
        )

    def search_kb_documents(self, kb_id: str, query: str, top_k: int = 5):
        return safe_api_call(
            lambda: requests.post(f"{self.base_url}/knowledge-bases/{kb_id}/search", headers=self.headers,
                                  json={"query": query, "top_k": top_k}).json(),
            error_msg="向量搜索失败"
        )

    # ---------- 规则管理 ----------
    def get_rules(self, session_id: str):
        return self._get_rules(self.base_url, json.dumps(self.headers), session_id)

    def create_rule(self, session_id: str, content: str, priority: int = 0, category: str = "general"):
        return safe_api_call(
            lambda: requests.post(f"{self.base_url}/sessions/{session_id}/rules", headers=self.headers,
                                  json={"content": content, "priority": priority, "category": category}).json(),
            error_msg="创建规则失败"
        )

    def delete_rule(self, session_id: str, rule_id: str):
        return safe_api_call(
            lambda: requests.delete(f"{self.base_url}/sessions/{session_id}/rules/{rule_id}", headers=self.headers),
            error_msg="删除规则失败"
        ) is not None

    def toggle_rule(self, session_id: str, rule_id: str):
        return safe_api_call(
            lambda: requests.post(f"{self.base_url}/sessions/{session_id}/rules/{rule_id}/toggle", headers=self.headers).json(),
            error_msg="切换规则状态失败"
        )

# 初始化 API 客户端
api_client = APIClient(API_BASE_URL, API_KEY)

# ==================== 侧边栏导航 ====================
with st.sidebar:
    st.title("🎓 企业学习助手（优化版）")
    st.markdown("---")

    # 配置面板（可折叠）
    with st.expander("⚙️ API 配置"):
        new_url = st.text_input("API 地址", value=st.session_state.api_base_url)
        new_key = st.text_input("API Key", value=st.session_state.api_key, type="password")
        if st.button("💾 保存配置"):
            st.session_state.api_base_url = new_url
            st.session_state.api_key = new_key
            st.rerun()

    main_menu = st.radio(
        "主要功能",
        ["💬 智能对话", "📈 投研日报", "🔬 技术雷达", "📚 知识库管理", "⚙️ 规则配置", "🛠️ 技能与流水线", "📊 系统监控"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    st.subheader("💭 当前会话")
    # 会话列表分页
    page = st.session_state.get("sessions_page", 1)
    sessions_data = api_client.get_sessions(page=page, page_size=5)
    total = sessions_data.get("total", 0) if sessions_data else 0

    if sessions_data and sessions_data.get("items"):
        session_options = {s["name"]: s["session_id"] for s in sessions_data["items"]}
        selected_session_name = st.selectbox(
            "选择会话",
            options=list(session_options.keys()),
            format_func=lambda x: f"💬 {x}",
            help="选择后自动加载历史"
        )

        # 翻页按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("◀", disabled=page<=1, key="sess_prev"):
                st.session_state.sessions_page = max(1, page-1)
                st.rerun()
        with col2:
            if st.button("▶", disabled=page*5 >= total, key="sess_next"):
                st.session_state.sessions_page = page+1
                st.rerun()

        # ✅ 显示当前选中的知识库
        if st.session_state.selected_kb_id:
            st.caption(f"📌 当前知识库: `{st.session_state.selected_kb_id[:8]}...`（新建会话将自动关联）")
        else:
            st.caption("⚠️ 未选择知识库，请先在知识库管理页面选择一个")

        if st.button("➕ 新建会话"):
            new_name = st.text_input("会话名称", value=f"新会话 {datetime.now().strftime('%H:%M')}", key="new_sess_name")
            if st.button("确认创建", key="confirm_create_sess"):
                # ✅ 修复：关联当前选中的知识库
                kb_ids = [st.session_state.selected_kb_id] if st.session_state.selected_kb_id else []
                result = api_client.create_session(new_name, kb_ids)
                if result:
                    st.success(f"会话 '{new_name}' 创建成功")
                    st.session_state.sessions_page = 1
                    st.rerun()

        if selected_session_name:
            new_id = session_options[selected_session_name]
            if st.session_state.current_session_id != new_id:
                st.session_state.current_session_id = new_id
                reset_session()
                st.rerun()
            st.info(f"当前会话: `{st.session_state.current_session_id[:8]}...`")
    else:
        st.warning("暂无会话")
        if st.button("创建第一个会话"):
            kb_ids = [st.session_state.selected_kb_id] if st.session_state.selected_kb_id else []
            result = api_client.create_session("默认会话", kb_ids)
            if result:
                st.success("会话创建成功")
                st.rerun()

# ==================== 主页面内容 ====================

# ---------- 页面 1: 智能对话 ----------
if main_menu == "💬 智能对话":
    st.header("💬 智能对话")

    if not st.session_state.current_session_id:
        st.warning("请先在侧边栏选择或创建一个会话")
        st.stop()

    # ✅ 新增：加载历史消息 + 断网恢复
    if not st.session_state.get("history_loaded") or not st.session_state.messages:
        history = api_client.get_session_history(st.session_state.current_session_id, limit=100)
        if history and isinstance(history, list):
            st.session_state.messages = history
            st.session_state.history_loaded = True
        elif history:
            st.session_state.messages = history.get("items", []) if isinstance(history, dict) else []
            st.session_state.history_loaded = True

    # 聊天容器（显示最近50条消息，避免页面过长）
    display_messages = st.session_state.messages[-50:] if len(
        st.session_state.messages) > 50 else st.session_state.messages

    chat_container = st.container()
    with chat_container:
        for msg in display_messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user"><strong>👤 用户:</strong><br>{msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-assistant"><strong>🤖 助手:</strong><br>{msg["content"]}</div>',
                            unsafe_allow_html=True)

    # 输入框（流式聊天）
    if prompt := st.chat_input("输入消息..."):
        # ✅ 立即显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # 占位符用于流式显示
        placeholder = st.empty()
        status_placeholder = st.empty()  # ✅ 新增：中间状态
        full_response = ""

        try:
            # 流式读取（传递当前选中的知识库 ID）
            kb_id = st.session_state.get("selected_kb_id")
            for event in api_client.chat_stream(st.session_state.current_session_id, prompt, kb_id=kb_id):
                try:
                    data = json.loads(event)
                except:
                    continue

                # ✅ 新增：处理中间状态事件
                if data.get("event") == "status" or "message" in data:
                    status_msg = data.get("message", data.get("status", ""))
                    if status_msg:
                        status_placeholder.info(f"⏳ {status_msg}")
                    continue

                chunk = data.get("chunk") or data.get("answer") or data.get("delta") or ""
                if chunk:
                    # 清除状态提示，显示回答
                    status_placeholder.empty()
                    full_response += str(chunk)
                    placeholder.markdown(
                        f'<div class="chat-assistant"><strong>🤖 助手:</strong><br>{full_response}</div>',
                        unsafe_allow_html=True)
        except Exception as e:
            status_placeholder.empty()
            st.error(f"流式响应失败: {str(e)}")
            full_response = "抱歉，服务暂时不可用。"

        # ✅ 流式未拿到内容时，从数据库取最新回答
        if not full_response.strip():
            try:
                history = api_client.get_session_history(st.session_state.current_session_id, limit=2)
                if history and isinstance(history, list):
                    for msg in reversed(history):
                        if msg.get("role") == "assistant" and msg.get("content", "").strip():
                            full_response = msg["content"]
                            break
            except Exception:
                pass

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # 更新占位符为最终内容
        placeholder.markdown(f'<div class="chat-assistant"><strong>🤖 助手:</strong><br>{full_response}</div>',
                             unsafe_allow_html=True)

    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

# ---------- 页面 2: 投研日报 ----------
elif main_menu == "📈 投研日报":
    st.header("📈 投研日报")

    st.markdown("""
    <div class="info-box">
    <strong>📋 追踪清单</strong><br>
    🟢 <strong>核心持仓</strong>：紫金矿业(601899)、特变电工(600089)、伯特利(603596)<br>
    🟡 <strong>等回调</strong>：中复神鹰(688295)、四方股份(601126)<br>
    🔵 <strong>埋伏</strong>：光威复材(300699)、锡业股份(000960)、北方稀土(600111)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 一键生成投研日报", type="primary", use_container_width=True):
            with st.spinner("正在拉取行情数据... 可能需要 10-30 秒"):
                try:
                    headers = {"Content-Type": "application/json"}
                    if API_KEY:
                        headers["Authorization"] = f"Bearer {API_KEY}"

                    # 请求 Markdown 格式（同时获取 JSON 原始数据）
                    resp = requests.get(
                        f"{API_BASE_URL}/stock-report/markdown",
                        headers=headers,
                        timeout=60
                    )
                    if resp.status_code == 200:
                        report = resp.json()
                        st.session_state.stock_report = report

                        # 保存到本地缓存，方便对比历史
                        st.session_state.stock_report_history = st.session_state.get("stock_report_history", [])
                        st.session_state.stock_report_history.insert(0, {
                            "date": report.get("date", ""),
                            "summary": report.get("json", {}).get("summary", ""),
                            "alert_count": len(report.get("json", {}).get("alerts", [])),
                        })
                        # 只保留最近 10 条
                        st.session_state.stock_report_history = st.session_state.stock_report_history[:10]

                        st.success("✅ 日报生成成功！")
                    else:
                        st.error(f"API 请求失败: HTTP {resp.status_code} — {resp.text[:200]}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ 请求超时，请检查后端服务是否正常运行")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 无法连接到后端，请确认 FastAPI 服务已启动")
                except Exception as e:
                    st.error(f"❌ 请求失败: {str(e)}")

    # 显示报告
    if "stock_report" in st.session_state and st.session_state.stock_report:
        report = st.session_state.stock_report

        st.markdown("---")

        # ===== Markdown 渲染 =====
        md_content = report.get("markdown", "")
        if md_content:
            st.markdown(md_content)

        # ===== 原始 JSON（可折叠） =====
        with st.expander("📄 查看原始 JSON 数据", expanded=False):
            json_data = report.get("json", {})
            st.json(json_data, expanded=False)

        # ===== 历史记录 =====
        history = st.session_state.get("stock_report_history", [])
        if len(history) > 1:
            with st.expander(f"📜 历史记录（最近 {len(history)} 条）", expanded=False):
                for h in history[1:]:  # 跳过第一条（当前）
                    alert_emoji = "🔴" if h.get("alert_count", 0) > 0 else "✅"
                    st.caption(f"{alert_emoji} {h['date']} — {h.get('summary', '')[:80]}...")

    else:
        st.info("👆 点击上方按钮生成今日投研日报")

# ---------- 页面 3: 技术雷达 ----------
elif main_menu == "🔬 技术雷达":
    st.header("🔬 技术雷达 — 硬核技术突破追踪")

    st.markdown("""
    <div class="info-box">
    <strong>📡 跟踪范围</strong><br>
    🧬 <strong>14 个核心技术领域</strong>：端侧大模型、Agent智能体、多模态AI、大模型可解释性<br>
    &nbsp;&nbsp;&nbsp;&nbsp;先进封装、存算一体、Chiplet标准 | 全固态电池、钠离子、钙钛矿光伏<br>
    &nbsp;&nbsp;&nbsp;&nbsp;人形机器人、具身智能 | eVTOL、卫星互联网<br>
    🔄 新闻来源：东方财富 + 百度新闻 | 自动评分 & 映射持仓
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 筛选栏
    col_filter, col_btn = st.columns([3, 1])
    with col_filter:
        area_options = {
            "all": "全部14个领域",
            "ai_edge_llm,ai_agent,ai_multimodal,ai_explainability": "AI领域",
            "semicon_advanced_pkg,semicon_compute_storage,semicon_chiplet": "半导体领域",
            "energy_solid_state,energy_sodium_ion,energy_perovskite": "电池能源",
            "robot_humanoid,robot_embodied": "机器人领域",
            "aviation_evtol,satellite_internet": "低空航天",
        }
        selected_area = st.selectbox("技术领域筛选", options=list(area_options.keys()),
                                     format_func=lambda x: area_options[x], key="tech_radar_area")
    with col_btn:
        st.write("")  # 对齐
        st.write("")
        scan_btn = st.button("🚀 一键扫描技术突破", type="primary", use_container_width=True)

    # 自定义筛选ID
    custom_ids = st.text_input("或自定义技术ID（逗号分隔，留空=全部）", placeholder="energy_solid_state,energy_perovskite", key="tech_radar_custom")

    if scan_btn:
        with st.spinner("正在扫描14个技术领域的突破新闻... 可能需要 20-40 秒"):
            try:
                headers = {"Content-Type": "application/json"}
                if API_KEY:
                    headers["Authorization"] = f"Bearer {API_KEY}"

                # 确定area_ids参数
                area_param = custom_ids.strip() if custom_ids.strip() else (None if selected_area == "all" else selected_area)

                # 请求 Markdown 报告
                params = {}
                if area_param:
                    params["area_ids"] = area_param

                resp = requests.get(
                    f"{API_BASE_URL}/tech-radar/markdown",
                    headers=headers,
                    params=params,
                    timeout=90
                )
                if resp.status_code == 200:
                    report = resp.json()
                    st.session_state.tech_radar_report = report

                    # 保存历史
                    hist = st.session_state.get("tech_radar_history", [])
                    json_data = report.get("json", {})
                    stats = json_data.get("stats", {})
                    hist.insert(0, {
                        "date": report.get("date", ""),
                        "high": stats.get("high", 0),
                        "medium": stats.get("medium", 0),
                        "summary": json_data.get("summary", ""),
                    })
                    st.session_state.tech_radar_history = hist[:10]
                    st.success("✅ 技术雷达扫描完成！")
                else:
                    st.error(f"API 请求失败: HTTP {resp.status_code} — {resp.text[:200]}")
            except requests.exceptions.Timeout:
                st.error("⏱️ 请求超时，扫描技术领域较多，请稍后重试")
            except requests.exceptions.ConnectionError:
                st.error("🔌 无法连接到后端，请确认 FastAPI 服务已启动")
            except Exception as e:
                st.error(f"❌ 请求失败: {str(e)}")

    # 显示报告
    if "tech_radar_report" in st.session_state and st.session_state.tech_radar_report:
        report = st.session_state.tech_radar_report
        st.markdown("---")

        # Markdown 渲染
        md_content = report.get("markdown", "")
        if md_content:
            st.markdown(md_content)

        # JSON 数据查看
        with st.expander("📄 查看原始 JSON 数据", expanded=False):
            json_data = report.get("json", {})
            st.json(json_data, expanded=False)

        # 历史记录
        hist = st.session_state.get("tech_radar_history", [])
        if len(hist) > 1:
            with st.expander(f"📜 扫描历史（最近 {len(hist)} 条）", expanded=False):
                for h in hist[1:]:
                    st.caption(f"{h['date']} — 🔴{h.get('high',0)} 🟡{h.get('medium',0)} 条 | {h.get('summary','')[:80]}...")
    else:
        st.info("👆 点击上方按钮扫描技术突破新闻")

# ---------- 页面 4: 知识库管理 ----------
elif main_menu == "📚 知识库管理":
    st.header("📚 知识库管理")

    tab1, tab2, tab3 = st.tabs(["📋 知识库列表", "📄 文档管理", "🔍 向量搜索测试"])

    with tab1:
        st.subheader("知识库列表")
        col1, col2 = st.columns([3,1])
        with col1:
            search_query = st.text_input("搜索知识库", placeholder="输入名称关键词...", key="kb_search")
        with col2:
            if st.button("➕ 新建知识库"):
                st.session_state.show_create_kb = True

        if st.session_state.show_create_kb:
            with st.form("create_kb_form"):
                kb_name = st.text_input("知识库名称")
                kb_desc = st.text_area("描述", value="")
                embed_model = st.selectbox("嵌入模型", ["text-embedding-ada-002", "bge-m3", "text-embedding-zh"])
                col_sub, col_canc = st.columns(2)
                with col_sub:
                    if st.form_submit_button("✅ 创建"):
                        if kb_name:
                            res = api_client.create_knowledge_base(kb_name, kb_desc, embed_model)
                            if res:
                                st.success(f"知识库 '{kb_name}' 创建成功")
                                st.session_state.show_create_kb = False
                                st.rerun()
                with col_canc:
                    if st.form_submit_button("❌ 取消"):
                        st.session_state.show_create_kb = False
                        st.rerun()

        # 知识库列表（分页）
        kb_page = st.session_state.get("kb_page", 1)
        kbs_data = api_client.get_knowledge_bases(page=kb_page, page_size=5, search=search_query if search_query else None)
        total_kb = kbs_data.get("total", 0) if kbs_data else 0
        kb_page = pagination_controls(total_kb, 5, "kb")
        if kb_page != st.session_state.get("kb_page", 1):
            st.session_state.kb_page = kb_page
            st.rerun()

        if kbs_data and kbs_data.get("items"):
            for kb in kbs_data["items"]:
                with st.expander(f"📁 {kb['name']} (ID: {kb['kb_id'][:8]}...)"):
                    st.write(f"**描述:** {kb.get('description', '无')}")
                    st.write(f"**嵌入模型:** {kb.get('embedding_model', 'N/A')}")
                    st.write(f"**状态:** {kb.get('status', 'N/A')}")
                    st.write(f"**文档数:** {kb.get('document_count', 0)}")
                    st.write(f"**创建时间:** {kb.get('created_at', 'N/A')}")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("✏️ 编辑", key=f"edit_{kb['kb_id']}"):
                            st.session_state.edit_kb_id = kb['kb_id']
                    with col_b:
                        if st.button("🗑️ 删除", key=f"del_{kb['kb_id']}"):
                            if confirm_delete_dialog(kb['name']):
                                if api_client.delete_knowledge_base(kb['kb_id']):
                                    st.success("删除成功")
                                    st.rerun()
                    with col_c:
                        if st.button("👁️ 查看文档", key=f"view_{kb['kb_id']}"):
                            st.session_state.selected_kb_id = kb['kb_id']
                            st.session_state.doc_page = 1
                            st.rerun()
        else:
            st.info("暂无知识库")

    with tab2:
        st.subheader("文档管理")
        if not st.session_state.selected_kb_id:
            st.warning("请先在左侧选择一个知识库")
        else:
            st.info(f"当前知识库: `{st.session_state.selected_kb_id[:8]}...`")
            uploaded_file = st.file_uploader("上传文档", type=['pdf','docx','txt','md'], key="doc_upload")
            if uploaded_file and st.button("📤 上传"):
                with st.spinner("上传并索引中..."):
                    result = api_client.upload_document(st.session_state.selected_kb_id, uploaded_file)
                    if result:
                        st.success(f"文档 '{uploaded_file.name}' 上传成功")
                        st.rerun()

            st.markdown("---")
            # 文档分页
            doc_page = st.session_state.get("doc_page", 1)
            docs_data = api_client.get_kb_documents(st.session_state.selected_kb_id, page=doc_page, page_size=5)
            total_doc = docs_data.get("total", 0) if docs_data else 0
            doc_page = pagination_controls(total_doc, 5, "doc")
            if doc_page != st.session_state.get("doc_page", 1):
                st.session_state.doc_page = doc_page
                st.rerun()

            if docs_data and docs_data.get("items"):
                for doc in docs_data["items"]:
                    col_doc, col_del = st.columns([4,1])
                    with col_doc:
                        # ✅ 新增：显示索引状态
                        idx_status = doc.get("indexing_status", doc.get("status", "unknown"))
                        status_icon = {"done": "✅", "processing": "⏳", "pending": "🕐", "error": "❌"}.get(idx_status, "📄")
                        st.write(f"{status_icon} {doc['file_name']}")
                        st.caption(f"上传时间: {doc.get('created_at', 'N/A')} | 索引状态: {idx_status}")
                    with col_del:
                        if st.button("🗑️", key=f"del_doc_{doc['doc_id']}"):
                            if confirm_delete_dialog(doc['file_name']):
                                if api_client.delete_document(st.session_state.selected_kb_id, doc['doc_id']):
                                    st.success("删除成功")
                                    st.rerun()
            else:
                st.info("暂无文档")

            if st.button("🔄 重新索引整个知识库"):
                if confirm_delete_dialog("重新索引将重新处理所有文档"):
                    with st.spinner("提交索引任务..."):
                        result = api_client.reindex_kb(st.session_state.selected_kb_id, force=True)
                        if result:
                            st.success(f"索引任务已提交 (Task ID: {result.get('task_id', 'N/A')})")

    with tab3:
        st.subheader("向量搜索测试")
        if not st.session_state.selected_kb_id:
            st.warning("请先在文档管理页面选择一个知识库")
        else:
            test_query = st.text_input("搜索查询", key="search_test_query")
            top_k = st.slider("返回数量", 1, 10, 5)
            if st.button("🔍 搜索"):
                if test_query:
                    with st.spinner("向量检索中..."):
                        results = api_client.search_kb_documents(st.session_state.selected_kb_id, test_query, top_k)
                        if results and results.get("items"):
                            for item in results["items"]:
                                st.markdown(f"**📄 {item.get('file_name', '未知')}** (相似度: {item.get('score', 0):.3f})")
                                st.text(item.get("content", "")[:200] + "...")
                        else:
                            st.warning("未找到匹配文档")
                else:
                    st.warning("请输入查询文本")

# ---------- 页面 4: 规则配置 ----------
elif main_menu == "⚙️ 规则配置":
    st.header("⚙️ 规则配置")
    if not st.session_state.current_session_id:
        st.warning("请先选择会话")
        st.stop()

    st.info(f"当前会话: `{st.session_state.current_session_id[:8]}...`")

    with st.expander("➕ 创建新规则", expanded=False):
        with st.form("create_rule_form"):
            rule_content = st.text_area("规则内容", placeholder="例如：回答必须简洁")
            rule_priority = st.slider("优先级", 0, 10, 5)
            rule_category = st.selectbox("分类", ["general", "format", "tone", "constraint"])
            if st.form_submit_button("✅ 创建规则"):
                if rule_content:
                    result = api_client.create_rule(st.session_state.current_session_id, rule_content, rule_priority, rule_category)
                    if result:
                        st.success("规则创建成功")
                        st.rerun()

    rules = api_client.get_rules(st.session_state.current_session_id)
    if rules:
        st.write(f"**共 {len(rules)} 条规则**")
        for rule in rules:
            is_enabled = rule.get('is_enabled', True)
            status_badge = "🟢 已启用" if is_enabled else "⚫ 已禁用"
            with st.container():
                col1, col2, col3 = st.columns([5, 1, 1])
                with col1:
                    st.markdown(f"{status_badge} | `{rule.get('category', 'N/A')}` | {rule['content'][:60]}{'...' if len(rule['content'])>60 else ''}")
                    st.caption(f"优先级: {rule['priority']}")
                with col2:
                    toggle_label = "⏸ 禁用" if is_enabled else "▶ 启用"
                    if st.button(toggle_label, key=f"toggle_{rule['rule_id']}"):
                        res = api_client.toggle_rule(st.session_state.current_session_id, rule['rule_id'])
                        if res:
                            st.rerun()
                with col3:
                    if st.button("🗑️", key=f"del_rule_{rule['rule_id']}", help="删除规则"):
                        if confirm_delete_dialog(rule['content'][:20]):
                            if api_client.delete_rule(st.session_state.current_session_id, rule['rule_id']):
                                st.success("删除成功")
                                st.rerun()
            st.divider()
    else:
        st.info("暂无规则")

# ---------- 页面 6: 技能与流水线 ----------
elif main_menu == "🛠️ 技能与流水线":
    st.header("🛠️ 技能与流水线")
    tab1, tab2 = st.tabs(["🚀 预定义流水线", "🎯 技能列表"])

    with tab1:
        st.subheader("预定义流水线")
        st.caption("聚焦知识库构建与检索，与「企业学习助手」定位一致")
        pipelines = {
            "index_then_search": {
                "name": "单文档索引后检索", "desc": "对单个文档构建 GraphRAG 知识图谱索引，再进行问答检索",
                "steps": ["graphrag_indexer → graphrag_searcher"]
            },
            "multi_file_index_then_search": {
                "name": "多文件批量索引后检索", "desc": "批量索引多个文件后统一检索",
                "steps": ["graphrag_indexer(批量) → graphrag_searcher"]
            }
        }

        selected = st.selectbox(
            "选择流水线",
            options=list(pipelines.keys()),
            format_func=lambda x: f"{pipelines[x]['name']} — {pipelines[x]['desc']}"
        )
        if selected:
            st.write(f"**步骤:** {' → '.join(pipelines[selected]['steps'])}")
            st.info("💡 在「智能对话」页面直接描述需求即可自动触发对应流水线，例如：「帮我索引这篇文档然后提问」")

    with tab2:
        st.subheader("可用技能列表")
        skill_categories = {
            "RAG 技能": ["document_loader", "document_chunker", "text_embedder",
                         "vector_search", "query_rewrite_skill", "rerank_skill",
                         "rag_answer", "graphrag_indexer", "graphrag_searcher", "kb_manager_skill"],
            "内容创作": ["text_summarizer", "outline_generator"],
            "数据分析": ["data_cleaner", "chart_advisor"],
            "办公效率": ["translator", "email_drafter", "meeting_summarizer"],
            "技术开发": ["code_explainer", "unit_test_generator", "code_review"]
        }
        for cat, skills in skill_categories.items():
            with st.expander(f"📂 {cat} ({len(skills)}个)"):
                for skill in skills:
                    st.write(f"• `{skill}`")
        st.info("技能可直接在对话中通过 Orchestrator 自动调用。")

# ---------- 页面 7: 系统监控 ----------
elif main_menu == "📊 系统监控":
    st.header("📊 系统监控")
    tab1, tab2, tab3 = st.tabs(["LangGraph 状态", "会话信息", "系统配置"])

    with tab1:
        st.subheader("LangGraph Checkpoint 状态")
        if not st.session_state.current_session_id:
            st.warning("请先选择会话")
        else:
            if st.button("🔄 刷新状态"):
                state = api_client.get_session_state(st.session_state.current_session_id)
                if state:
                    st.session_state.graph_state = state
            if st.session_state.graph_state:
                st.json(st.session_state.graph_state, expanded=False)
            else:
                st.info("点击刷新按钮查看状态")

    with tab2:
        st.subheader("当前会话信息")
        if st.session_state.current_session_id:
            session_info = {
                "会话 ID": st.session_state.current_session_id,
                "消息数量": len(st.session_state.messages),
                "最后更新时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.table(session_info)

    with tab3:
        st.subheader("系统配置")
        config_info = {
            "API 地址": st.session_state.api_base_url,
            "API Key": "已配置" if st.session_state.api_key else "未配置",
            "Python 版本": "3.12+",
            "框架": "Streamlit + FastAPI + LangGraph"
        }
        st.table(config_info)
        st.write("**技术栈:** FastAPI · LangGraph · ChromaDB · Streamlit · LiteLLM")

# ==================== 页脚 ====================
st.markdown("---")
st.caption("企业学习助手 v0.2.0（优化版）| 流式对话 · 分页 · 缓存 · 可配置")
"""FastAPI 应用入口（优化版）"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.api.routes.sessions import router as sessions_router
from src.api.middleware.error_handler import register_error_handlers
from src.infrastructure.session_manager import SessionManager
from src.agent.langgraph.checkpointer import get_checkpointer

# from src.api.routes.knowledge_bases import router as rules_router  # ← 新增这一行
from src.api.routes.knowledge_bases import router as knowledge_bases_router
from src.api.routes.rules import router as rules_router          # ← 新增
from src.api.routes.stock_report import router as stock_report_router  # ← 投研日报
from src.api.routes.tech_radar import router as tech_radar_router      # ← 技术雷达
# 使用结构化日志（示例使用标准 logging + JSON 格式，实际可替换为 structlog）
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# 应用生命周期：资源初始化与清理
# ---------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化单例资源
    logger.info("正在初始化检查点...")
    checkpointer = get_checkpointer()
    app.state.checkpointer = checkpointer
    logger.info("正在预热 SessionManager...")
    from src.api.deps import get_session_manager

    # 先触发一次注入，确保单例就绪（实际由 deps 中的单例逻辑管理）
    # 这里直接调用 deps 中的工厂以完成初始化
    # 注意：真实项目中应在 deps 中提供异步初始化方法
    # 此处仅为示意
    logger.info("启动完成，所有服务已就绪")
    yield
    # 关闭时清理资源（如关闭数据库连接池等）
    logger.info("正在关闭服务...")
    # 如果有需要清理的同步资源，可在此处调用
    # await close_db_connections()
    logger.info("服务已安全关闭")


# ---------------------------------------------------------------
# 创建应用
# ---------------------------------------------------------------
app = FastAPI(
    title="Enterprise Learning Agent API",
    description="企业学习助手 REST API",
    version="0.1.0",
    lifespan=lifespan,
)

# 注册全局异常处理器（含 request_id 中间件）
register_error_handlers(app)

# CORS 配置：生产环境通过环境变量严格限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,          # 从配置读取，例 ["https://app.example.com"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# 注册路由（版本前缀）
app.include_router(sessions_router, prefix="/api/v1")
app.include_router(rules_router, prefix="/api/v1")  # ← 新增这一行
app.include_router(knowledge_bases_router, prefix="/api/v1")  # ← 新增这一行
app.include_router(stock_report_router, prefix="/api/v1")            # ← 投研日报
app.include_router(tech_radar_router, prefix="/api/v1")              # ← 技术雷达


# 健康检查：附带关键组件状态
@app.get("/health")
def health_check():
    """健康检查，返回组件状态"""
    # 检查数据库连通性等（示例为模拟）
    db_ok = True  # 可调用 SessionManager 的健康检查方法
    return {
        "status": "ok" if db_ok else "degraded",
        "version": "0.1.0",
        "components": {
            "database": "ok" if db_ok else "error",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
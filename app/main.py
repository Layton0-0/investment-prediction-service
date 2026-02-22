"""
FastAPI 애플리케이션 진입점
"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="AI Prediction Service",
    description="Investment Choi AI 예측 서비스",
    version="1.0.0"
)

# CORS: 프로덕션에서는 CORS_ORIGINS 환경 변수로 허용 오리진 제한 (쉼표 구분). 미설정 시 "*"
_cors_origins = os.environ.get("CORS_ORIGINS", "*")
_origins = [s.strip() for s in _cors_origins.split(",") if s.strip()] if _cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(router, prefix="/api/v1", tags=["prediction"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "AI Prediction Service",
        "version": "1.0.0",
        "status": "running"
    }

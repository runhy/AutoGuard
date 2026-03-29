# app/api/websearch.py

from fastapi import APIRouter
import asyncio
import os
from dotenv import load_dotenv

router = APIRouter()

@router.get("/search")
def search(query: str):
    # TODO: Web Search API 결정되면 여기 채우기
    return{
        "module": "URL_Analyzer",
        "is_malicious": 0,
        "confidence_score": 0.0,
        "detected_features": ["Web Search API 미정 - 추후 연동 예정"]
    }
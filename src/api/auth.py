"""API Key 鉴权（哈希校验）"""
import hashlib
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader
from src.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def verify_api_key(key: str = Depends(api_key_header)) -> str:
    if settings.env == "dev":
        return "dev_user"
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    hashed = _hash(key)
    for pair in settings.api_keys:
        stored_hash, user_id = pair.split(":", 1)
        if hashed == stored_hash:
            return user_id
    raise HTTPException(status_code=401, detail="Invalid API key")
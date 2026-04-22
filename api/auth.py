import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

SECRET_KEY = os.getenv("OPEN_NOTEBOOK_JWT_SECRET", "change-me-to-a-secret-string-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

PUBLIC_PATHS = {
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/status",
    "/api/config",
}


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Validates JWT on every request except public paths. Attaches user info to request.state."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            request.state.user_id = payload["sub"]
            request.state.user_role = payload.get("role", "user")
            request.state.user_email = payload.get("email", "")
            request.state.user_name = payload.get("name", "")
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Token has expired. Please log in again."},
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            logger.debug(f"Invalid JWT token: {e}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authentication token"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)


async def get_current_user(request: Request) -> dict:
    """FastAPI dependency that extracts the current user from request.state."""
    if not hasattr(request.state, "user_id"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "id": request.state.user_id,
        "role": request.state.user_role,
        "email": request.state.user_email,
        "name": request.state.user_name,
    }


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """FastAPI dependency that requires admin role."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# Legacy: kept for backward compat with any code that imports this
security = HTTPBearer(auto_error=False)

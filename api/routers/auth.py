from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from api.auth import create_access_token, get_current_user, require_admin
from open_notebook.domain.user import User

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class RegisterRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")
    name: str = Field(..., min_length=1, description="Display name")


class InviteRequest(BaseModel):
    email: str = Field(..., description="New user email")
    name: str = Field(..., description="Display name")
    password: str = Field(..., min_length=6, description="Initial password")
    role: str = Field("user", description="Role: 'user' or 'admin'")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str


@router.get("/status")
async def get_auth_status():
    """Check if authentication is enabled and whether open registration is allowed."""
    import os

    registration_disabled = os.getenv("OPEN_NOTEBOOK_DISABLE_REGISTRATION", "").lower() in (
        "1", "true", "yes"
    )
    return {
        "auth_enabled": True,
        "message": "JWT authentication is active",
        "mode": "multi-user",
        "registration_enabled": not registration_disabled,
    }


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate and return a JWT token."""
    user = await User.get_by_email(request.email)
    if not user or not user.verify_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({
        "sub": str(user.id),
        "email": user.email,
        "name": user.name,
        "role": user.role,
    })

    return TokenResponse(
        access_token=token,
        user={"id": str(user.id), "email": user.email, "name": user.name, "role": user.role},
    )


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    """Register a new user account (open registration).

    Set OPEN_NOTEBOOK_DISABLE_REGISTRATION=true to disable this endpoint and
    require admin invitation for all new accounts.
    """
    import os

    if os.getenv("OPEN_NOTEBOOK_DISABLE_REGISTRATION", "").lower() in ("1", "true", "yes"):
        raise HTTPException(
            status_code=403,
            detail="Open registration is disabled. Contact your administrator to create an account.",
        )

    existing = await User.get_by_email(request.email)
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user = await User.create_user(
        email=request.email,
        password=request.password,
        name=request.name,
        role="user",
    )

    token = create_access_token({
        "sub": str(user.id),
        "email": user.email,
        "name": user.name,
        "role": user.role,
    })

    return TokenResponse(
        access_token=token,
        user={"id": str(user.id), "email": user.email, "name": user.name, "role": user.role},
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get the current authenticated user's profile."""
    user = await User.get(current_user["id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        role=user.role,
    )


@router.post("/invite", response_model=UserResponse)
async def invite_user(
    request: InviteRequest,
    _admin: dict = Depends(require_admin),
):
    """Admin: create a new user account (invite flow)."""
    existing = await User.get_by_email(request.email)
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    if request.role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")

    user = await User.create_user(
        email=request.email,
        password=request.password,
        name=request.name,
        role=request.role,
    )

    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        role=user.role,
    )

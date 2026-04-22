from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import get_current_user, require_admin
from open_notebook.database.repository import repo_query
from open_notebook.domain.user import User

router = APIRouter(prefix="/users", tags=["users"])


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    created: Optional[str] = None
    updated: Optional[str] = None


class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    password: Optional[str] = Field(None, min_length=6)


@router.get("", response_model=List[UserResponse])
async def list_users(_admin: dict = Depends(require_admin)):
    """Admin: list all users."""
    users = await User.get_all(order_by="created asc")
    return [
        UserResponse(
            id=str(u.id),
            email=u.email,
            name=u.name,
            role=u.role,
            created=str(u.created) if u.created else None,
            updated=str(u.updated) if u.updated else None,
        )
        for u in users
    ]


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """Get the currently authenticated user's profile."""
    user = await User.get(current_user["id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        role=user.role,
        created=str(user.created) if user.created else None,
        updated=str(user.updated) if user.updated else None,
    )


@router.put("/me", response_model=UserResponse)
async def update_my_profile(
    request: UpdateUserRequest,
    current_user: dict = Depends(get_current_user),
):
    """Update current user's own name or password."""
    user = await User.get(current_user["id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if request.name is not None:
        user.name = request.name

    if request.password is not None:
        from passlib.context import CryptContext
        user.password_hash = CryptContext(schemes=["bcrypt"], deprecated="auto").hash(request.password)

    await user.save()
    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        role=user.role,
        created=str(user.created) if user.created else None,
        updated=str(user.updated) if user.updated else None,
    )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    _admin: dict = Depends(require_admin),
):
    """Admin: update any user's name, role, or password."""
    full_id = user_id if user_id.startswith("user:") else f"user:{user_id}"
    user = await User.get(full_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if request.name is not None:
        user.name = request.name

    if request.role is not None:
        if request.role not in ("user", "admin"):
            raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")
        user.role = request.role

    if request.password is not None:
        from passlib.context import CryptContext
        user.password_hash = CryptContext(schemes=["bcrypt"], deprecated="auto").hash(request.password)

    await user.save()
    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        role=user.role,
        created=str(user.created) if user.created else None,
        updated=str(user.updated) if user.updated else None,
    )


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    current_admin: dict = Depends(require_admin),
):
    """Admin: delete a user (cannot delete yourself)."""
    full_id = user_id if user_id.startswith("user:") else f"user:{user_id}"
    if full_id == current_admin["id"] or user_id == current_admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    user = await User.get(full_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await user.delete()
    return {"message": "User deleted successfully"}

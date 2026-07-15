from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

from production_rag.models.enums import UserRole


class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr


class UserUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    email: EmailStr | None = None


class UserResponse(BaseModel):
    id: UUID
    name: str
    email: EmailStr
    role: UserRole
    created_at: datetime
    updated_at: datetime
    # hashed_password is NOT here because shoud be never exposed
    model_config = {"from_attributes": True}


class UserListResponse(BaseModel):
    users: list[UserResponse]
    total: int
    limit: int
    offset: int


class UserRoleUpdate(BaseModel):
    role: UserRole

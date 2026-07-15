from fastapi import APIRouter, Depends

from production_rag.dependencies import get_auth_service
from production_rag.schemas.auth import LoginRequest, RefreshRequest, SignupRequest, TokenResponse
from production_rag.schemas.user import UserResponse
from production_rag.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserResponse, status_code=201)
async def signup(
    data: SignupRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    user = await auth_service.signup(
        name=data.name,
        email=data.email,
        password=data.password,
    )
    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    data: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    return await auth_service.login(email=data.email, password=data.password)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    data: RefreshRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    return await auth_service.refresh(refresh_token_str=data.refresh_token)

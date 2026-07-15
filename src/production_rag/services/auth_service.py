from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from jose import JWTError, jwt
from passlib.context import CryptContext

from production_rag.config import get_settings
from production_rag.exceptions import PaddingtonError
from production_rag.models import refresh_token
from production_rag.repositories.refresh_token_repository import RefreshTokenRepository
from production_rag.repositories.user_repository import UserRepository
from production_rag.schemas.auth import TokenResponse

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


class InvalidCredentialsError(PaddingtonError):
    status_code = 401


class InvalidTokenError(PaddingtonError):
    status_code = 401


def hash_password(pasword: str) -> str:
    return pwd_context.hash(pasword)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: UUID, email: str, role: str) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)

    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": str(uuid4()),
    }

    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=settings.jwt_algorithm)
        return payload
    except JWTError as e:
        raise InvalidTokenError("Invalid or expired token") from e


class AuthService:
    def __init__(
        self,
        user_repository: UserRepository,
        refresh_token_repository: RefreshTokenRepository,
    ) -> None:
        self._user_repo = user_repository
        self._refresh_repo = refresh_token_repository

    async def signup(self, name: str, email: str, password: str):
        return await self._user_repo.create(
            name=name,
            email=email,
            hashed_password=hash_password(password),
        )

    async def login(self, email: str, password: str) -> TokenResponse:
        user = await self._user_repo.get_by_email(email)
        if user is None:
            raise InvalidCredentialsError("Invalid email or password")
        if not verify_password(password, user.hashed_password):
            raise InvalidCredentialsError("Invalid email or password")

        access_token = create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
        )
        new_refresh_token = await self._refresh_repo.create(user_id=user.id)

        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token.token,
        )

    async def refresh(self, refresh_token_str: str) -> TokenResponse:
        old_token = await self._refresh_repo.get_valid_token(refresh_token_str)
        await self._refresh_repo.revoke(old_token)

        user = await self._user_repo.get_by_id(old_token.user_id)
        new_access_token = create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
        )
        new_refresh_token = await self._refresh_repo.create(user_id=user.id)

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token.token,
        )

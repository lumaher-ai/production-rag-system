from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.exceptions import AlreadyExistsError, NotFoundError
from production_rag.models import User


class UserAlreadyExistsError(AlreadyExistsError):
    pass


class UserNotFoundError(NotFoundError):
    pass


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, name: str, email: str, hashed_password: str) -> User:
        user = User(name=name, email=email, hashed_password=hashed_password)
        self._session.add(user)
        try:
            await self._session.flush()
        except IntegrityError as e:
            await self._session.rollback()
            raise UserAlreadyExistsError(f"User with email {email} already exists") from e
        return user

    async def get_by_id(self, user_id: UUID) -> User:
        result = await self._session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")
        return user

    async def get_by_email(self, email: str) -> User | None:
        result = await self._session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if user is None:
            raise UserNotFoundError(f"User with email: {email} not found")
        return user

    async def list_all(self, limit: int = 20, offset: int = 0) -> list[User]:
        result = await self._session.execute(
            select(User).order_by(User.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        from sqlalchemy import func

        result = await self._session.execute(select(func.count()).select_from(User))
        return result.scalar_one()

    async def update(self, user_id: UUID, **kwargs) -> User:
        user = await self.get_by_id(user_id)
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
            else:
                raise ValueError(f"User has no attribute '{key}'")
        try:
            await self._session.flush()
        except IntegrityError as e:
            await self._session.rollback()
            raise UserAlreadyExistsError("Email already taken") from e
        return user

    async def delete(self, user_id: UUID) -> None:
        user = await self.get_by_id(user_id)
        await self._session.delete(user)
        await self._session.flush()

    async def update_role(self, user_id: UUID, role: str) -> User:
        user = await self.get_by_id(user_id)
        user.role = role
        await self._session.flush()
        return user

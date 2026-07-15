import asyncio

from production_rag.config import get_settings
from production_rag.database import close_db, get_session, init_db
from production_rag.models.enums import UserRole
from production_rag.repositories.user_repository import UserRepository
from production_rag.services.auth_service import hash_password


async def create_admin(email: str, name: str, password: str) -> None:
    await init_db()

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.create(
            name=name,
            email=email,
            hashed_password=hash_password(password),
        )
        await repo.update_role(user.id, UserRole.ADMIN.value)

    await close_db()
    print(f"Admin user created: {email}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python -m production_rag.cli <email> <name> <password>")
        sys.exit(1)

    asyncio.run(create_admin(sys.argv[1], sys.argv[2], sys.argv[3]))

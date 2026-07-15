# ruff: noqa: E402
# load_dotenv() runs before any other import so .env is in os.environ
# before modules like litellm read provider keys at import/call time.
from dotenv import load_dotenv

load_dotenv()

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from production_rag.config import get_settings
from production_rag.database import close_db, init_db
from production_rag.exception_handlers import register_exception_handlers
from production_rag.logging_config import configure_logging, get_logger
from production_rag.routes import agent, auth, chat, documents, echo, health, users

configure_logging()

logger = get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()

    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(get_settings().checkpointer_url)
        )
        await checkpointer.setup()
        app.state.checkpointer = checkpointer

        logger.info(
            "application_started",
            debug_dir=settings.debug_dir,
        )
        yield

    await close_db()
    logger.info("application_stopped")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(echo.router)
app.include_router(users.router)
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(agent.router)

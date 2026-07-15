from datetime import datetime, timezone

from production_rag.logging_config import get_logger
from production_rag.schemas.echo import EchoRequest, EchoResponse

logger = get_logger(__name__)


def create_echo(request: EchoRequest) -> EchoResponse:
    logger.info(
        "echo_requested",
        message_length=len(request.message),
        repeat_count=request.repeat,
    )
    response = EchoResponse(
        original=request.message,
        echoed=[request.message] * request.repeat,
        received_at=datetime.now(timezone.utc),
    )
    logger.info(
        "echo_created",
        total_chars=len(request.message) * request.repeat,
    )

    return response

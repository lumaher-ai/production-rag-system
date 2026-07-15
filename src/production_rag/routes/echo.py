from fastapi import APIRouter

from production_rag.schemas.echo import EchoRequest, EchoResponse
from production_rag.services.echo_service import create_echo

router = APIRouter(tags=["echo"])


@router.post("/echo", response_model=EchoResponse)
async def echo(request: EchoRequest) -> EchoResponse:
    return create_echo(request)

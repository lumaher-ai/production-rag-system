from datetime import datetime

from pydantic import BaseModel, Field


class EchoRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    repeat: int = Field(default=1, ge=1, le=10)


class EchoResponse(BaseModel):
    original: str
    echoed: list[str]
    received_at: datetime

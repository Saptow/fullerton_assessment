from typing import Any

from pydantic import BaseModel, Field


class DocumentClassificationError(Exception):
    """Raised when the model cannot confidently classify a supported document."""

    def __init__(
        self,
        message: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.payload = payload or {}


class ErrorResponse(BaseModel):
    error: str = Field(examples=["internal_server_error"])

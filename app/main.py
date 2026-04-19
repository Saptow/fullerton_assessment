from fastapi import FastAPI

from app.api.routes import router
from app.schemas import HealthResponse


def create_app() -> FastAPI:
    app = FastAPI(
        title="Fullerton Assessment API",
        description="Basic FastAPI service with a multipart OCR endpoint.",
        version="0.1.0",
    )
    app.get("/health", response_model=HealthResponse, tags=["system"])(health_check)
    app.include_router(router)
    return app


async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


app = create_app()

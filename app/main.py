from fastapi import FastAPI
from app.core.config import settings
from app.apis import router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for AI predictions",
    version="1.0.0"
)

app.include_router(router, prefix='/api/v1')
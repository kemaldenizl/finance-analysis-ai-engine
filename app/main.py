from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.inputs import router as inputs_router
from app.core.config import settings
from app.db.base import create_db_tables
from app.api.routes.preprocessing import router as preprocessing_router
from app.api.routes.extractions import router as extractions_router


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        debug=settings.DEBUG,
    )

    app.include_router(health_router)
    app.include_router(inputs_router)
    app.include_router(preprocessing_router)
    app.include_router(extractions_router)

    @app.on_event("startup")
    def on_startup():
        create_db_tables()

    return app


app = create_app()

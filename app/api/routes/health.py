from fastapi import APIRouter, Response, status

from app.core.config import settings
from app.services.health_service import HealthService


router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    service = HealthService()
    readiness = service.readiness()

    return {
        "service": settings.APP_NAME,
        "environment": settings.ENV,
        "status": readiness["status"],
        "components": readiness["components"],
    }


@router.get("/health/live")
def live():
    return {
        "status": "ok",
        "service": settings.APP_NAME,
    }


@router.get("/health/ready")
def ready(response: Response):
    service = HealthService()
    readiness = service.readiness()

    if readiness["status"] != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return readiness
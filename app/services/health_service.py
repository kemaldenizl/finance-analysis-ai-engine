from typing import Any

from redis import Redis
from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine


class HealthService:
    def check_database(self) -> dict[str, Any]:
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))

            return {
                "status": "ok",
                "component": "postgresql",
            }

        except Exception as exc:
            return {
                "status": "error",
                "component": "postgresql",
                "error": str(exc),
            }

    def check_redis(self) -> dict[str, Any]:
        try:
            redis_client = Redis.from_url(
                settings.REDIS_URL,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )

            pong = redis_client.ping()

            return {
                "status": "ok" if pong else "error",
                "component": "redis",
            }

        except Exception as exc:
            return {
                "status": "error",
                "component": "redis",
                "error": str(exc),
            }

    def readiness(self) -> dict[str, Any]:
        database = self.check_database()
        redis = self.check_redis()

        components = {
            "database": database,
            "redis": redis,
        }

        is_ready = all(component["status"] == "ok" for component in components.values())

        return {
            "status": "ok" if is_ready else "error",
            "components": components,
        }
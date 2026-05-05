import os
import tempfile

from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.services.mime_detector import MimeDetector


class UploadService:
    async def read_and_validate(self, file: UploadFile) -> tuple[bytes, str]:
        file_bytes = await file.read()

        max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        if len(file_bytes) > max_size_bytes:
            raise HTTPException(status_code=413, detail="File too large")

        mime_type = MimeDetector.detect(file_bytes)

        if not MimeDetector.is_allowed(mime_type):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {mime_type}",
            )

        return file_bytes, mime_type

    def write_temp_file(self, file_bytes: bytes, mime_type: str) -> str:
        suffix = self._suffix_from_mime(mime_type)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(file_bytes)
        tmp.close()

        return tmp.name

    def cleanup_temp_file(self, path: str) -> None:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def _suffix_from_mime(self, mime_type: str) -> str:
        mapping = {
            "application/pdf": ".pdf",
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/heic": ".heic",
            "image/heif": ".heic",
        }

        return mapping.get(mime_type, "")

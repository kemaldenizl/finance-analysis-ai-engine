import magic


class MimeDetector:
    ALLOWED_MIME_TYPES = {
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    }

    @staticmethod
    def detect(file_bytes: bytes) -> str:
        return magic.from_buffer(file_bytes, mime=True)

    @classmethod
    def is_allowed(cls, mime_type: str) -> bool:
        return mime_type in cls.ALLOWED_MIME_TYPES

    @staticmethod
    def is_pdf(mime_type: str) -> bool:
        return mime_type == "application/pdf"

    @staticmethod
    def is_image(mime_type: str) -> bool:
        return mime_type.startswith("image/")
from pydantic import BaseModel

from app.schemas.classification import ClassificationResult


class InputUploadResponse(BaseModel):
    input_id: str
    status: str
    classification: ClassificationResult
    next_stage: str
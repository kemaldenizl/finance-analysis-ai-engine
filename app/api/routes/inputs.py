from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.input_record import InputClassification, InputRecord
from app.schemas.input_response import InputUploadResponse
from app.services.classification_service import ClassificationService
from app.services.upload_service import UploadService
from app.storage.object_storage import ObjectStorage
from app.workers.stage2_dispatcher import Stage2Dispatcher

router = APIRouter(prefix="/v1/inputs", tags=["inputs"])


@router.post("", response_model=InputUploadResponse)
async def upload_and_classify_input(
    file: UploadFile = File(...),
    user_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    upload_service = UploadService()
    classification_service = ClassificationService()
    object_storage = ObjectStorage()
    dispatcher = Stage2Dispatcher()

    file_bytes, mime_type = await upload_service.read_and_validate(file)

    storage_key, storage_url = object_storage.upload_bytes(
        file_bytes=file_bytes,
        mime_type=mime_type,
        original_filename=file.filename,
    )

    input_record = InputRecord(
        user_id=user_id,
        original_filename=file.filename,
        mime_type=mime_type,
        file_size=len(file_bytes),
        storage_key=storage_key,
        storage_url=storage_url,
        status="uploaded",
    )

    db.add(input_record)
    db.commit()
    db.refresh(input_record)

    temp_path = upload_service.write_temp_file(
        file_bytes=file_bytes,
        mime_type=mime_type,
    )

    try:
        classification_result = classification_service.classify(
            file_path=temp_path,
            mime_type=mime_type,
        )
    finally:
        upload_service.cleanup_temp_file(temp_path)

    classification_record = InputClassification(
        input_id=input_record.id,
        kind=classification_result.kind.value,
        confidence=classification_result.confidence,
        needs_ocr=classification_result.needs_ocr,
        needs_preprocessing=classification_result.needs_preprocessing,
        routing_key=classification_result.routing_key,
        features_json=classification_result.features,
        warnings_json=classification_result.warnings,
        model_version=classification_result.model_version,
    )

    input_record.status = "classified"

    db.add(classification_record)
    db.add(input_record)
    db.commit()

    dispatcher.dispatch(
        input_id=input_record.id,
        storage_key=storage_key,
        routing_key=classification_result.routing_key,
        classification_kind=classification_result.kind.value,
    )

    return InputUploadResponse(
        input_id=input_record.id,
        status="classified",
        classification=classification_result,
        next_stage=classification_result.routing_key,
    )

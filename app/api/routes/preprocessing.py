from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.preprocessing_record import InputPreprocessingRecord


router = APIRouter(prefix="/v1/preprocessings", tags=["preprocessings"])


@router.get("/{input_id}")
def get_preprocessing_result(
    input_id: str,
    db: Session = Depends(get_db),
):
    record = (
        db.query(InputPreprocessingRecord)
        .filter(InputPreprocessingRecord.input_id == input_id)
        .order_by(InputPreprocessingRecord.created_at.desc())
        .first()
    )

    if record is None:
        raise HTTPException(status_code=404, detail="Preprocessing result not found")

    return {
        "id": record.id,
        "input_id": record.input_id,
        "source_kind": record.source_kind,
        "status": record.status,
        "output_type": record.output_type,
        "output_storage_key": record.output_storage_key,
        "output_storage_url": record.output_storage_url,
        "page_count": record.page_count,
        "operations": record.operations_json,
        "quality_before": record.quality_before_json,
        "quality_after": record.quality_after_json,
        "outputs": record.outputs_json,
        "warnings": record.warnings_json,
        "is_ready_for_extraction": record.is_ready_for_extraction,
        "preprocessing_version": record.preprocessing_version,
        "created_at": record.created_at,
    }
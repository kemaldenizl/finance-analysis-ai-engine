from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.extraction_record import DataExtractionRecord
from app.models.normalization_record import NormalizationRecord
from app.schemas.normalization import Stage4Response
from app.services.normalization.normalization_service import NormalizationService


router = APIRouter(prefix="/v1/normalizations", tags=["normalizations"])


@router.post("/{input_id}", response_model=Stage4Response)
def normalize_latest_extraction(
    input_id: str,
    db: Session = Depends(get_db),
):
    extraction = (
        db.query(DataExtractionRecord)
        .filter(DataExtractionRecord.input_id == input_id)
        .order_by(DataExtractionRecord.created_at.desc())
        .first()
    )

    if extraction is None:
        raise HTTPException(status_code=404, detail="Extraction result not found")

    extraction_result = extraction.result_json or {}

    if "transactions" not in extraction_result:
        extraction_result = extraction_result.get("result") or extraction_result

    service = NormalizationService()

    response = service.normalize_extraction_result(
        input_id=input_id,
        extraction_result=extraction_result,
        extraction_method=extraction.extraction_method,
    )

    record = NormalizationRecord(
        input_id=input_id,
        extraction_id=extraction.id,
        status=response.status,
        normalization_version="normalization-v1",
        transaction_count=response.result.summary.transaction_count,
        duplicate_removed_count=response.result.summary.duplicate_removed_count,
        low_confidence_count=response.result.summary.low_confidence_count,
        overall_confidence=response.scores.summary.overall_confidence,
        result_json=response.result.model_dump(),
        scores_json=response.scores.model_dump(),
        warnings_json=response.warnings,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return response


@router.get("/{input_id}/latest")
def get_latest_normalization(
    input_id: str,
    db: Session = Depends(get_db),
):
    record = (
        db.query(NormalizationRecord)
        .filter(NormalizationRecord.input_id == input_id)
        .order_by(NormalizationRecord.created_at.desc())
        .first()
    )

    if record is None:
        raise HTTPException(status_code=404, detail="Normalization result not found")

    return {
        "id": record.id,
        "input_id": record.input_id,
        "extraction_id": record.extraction_id,
        "status": record.status,
        "normalization_version": record.normalization_version,
        "result": record.result_json,
        "scores": record.scores_json,
        "warnings": record.warnings_json,
        "created_at": record.created_at,
    }
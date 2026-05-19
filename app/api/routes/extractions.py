from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.extraction_record import DataExtractionRecord
from app.models.input_record import InputClassification, InputRecord
from app.models.preprocessing_record import InputPreprocessingRecord
from app.schemas.extraction import PdfExtractionApiResponse
from app.services.extraction.pdf_extraction_service import PdfExtractionService


router = APIRouter(prefix="/v1/extractions", tags=["extractions"])


@router.post("/pdf/{input_id}", response_model=PdfExtractionApiResponse)
def extract_pdf_data(
    input_id: str,
    include_debug: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    input_record = db.query(InputRecord).filter(InputRecord.id == input_id).first()

    if input_record is None:
        raise HTTPException(status_code=404, detail="Input not found")

    classification = (
        db.query(InputClassification)
        .filter(InputClassification.input_id == input_id)
        .first()
    )

    if classification is None:
        raise HTTPException(status_code=409, detail="Input has not been classified yet")

    if classification.kind != "real_pdf":
        raise HTTPException(
            status_code=400,
            detail=f"PDF native extraction supports only real_pdf. Current kind: {classification.kind}",
        )

    preprocessing = (
        db.query(InputPreprocessingRecord)
        .filter(InputPreprocessingRecord.input_id == input_id)
        .order_by(InputPreprocessingRecord.created_at.desc())
        .first()
    )

    if preprocessing is None:
        raise HTTPException(status_code=409, detail="Input has not been preprocessed yet")

    if preprocessing.preferred_extraction_method != "native_pdf_text":
        raise HTTPException(
            status_code=400,
            detail=(
                "Input is not prepared for native PDF text extraction. "
                f"Current method: {preprocessing.preferred_extraction_method}"
            ),
        )

    pdf_path = preprocessing.preferred_output_storage_url or input_record.storage_url

    extraction_service = PdfExtractionService()
    result = extraction_service.extract(
        pdf_path=pdf_path,
        include_debug=include_debug,
    )

    extraction_record = DataExtractionRecord(
        input_id=input_id,
        source_kind=classification.kind,
        extraction_type="pdf_data_extraction",
        extraction_method="native_pdf_text",
        status="completed",
        extraction_version="pdf-native-v1",
        transaction_count=result.summary.transaction_count,
        low_confidence_count=result.summary.low_confidence_count,
        average_confidence=result.summary.average_confidence,
        result_json=result.model_dump(exclude={"debug"}),
        debug_json=result.debug.model_dump() if result.debug else {},
        warnings_json=result.warnings,
    )

    db.add(extraction_record)
    db.commit()
    db.refresh(extraction_record)

    return PdfExtractionApiResponse(
        input_id=input_id,
        status="completed",
        stage1={
            "kind": classification.kind,
            "confidence": classification.confidence,
            "routing_key": classification.routing_key,
            "needs_ocr": classification.needs_ocr,
            "needs_preprocessing": classification.needs_preprocessing,
        },
        stage2={
            "source_kind": preprocessing.source_kind,
            "output_type": preprocessing.output_type,
            "preferred_output_variant": preprocessing.preferred_output_variant,
            "preferred_extraction_method": preprocessing.preferred_extraction_method,
            "preferred_output_storage_url": preprocessing.preferred_output_storage_url,
            "extraction_risk": preprocessing.extraction_risk,
            "page_count": preprocessing.page_count,
            "is_ready_for_extraction": preprocessing.is_ready_for_extraction,
        },
        stage3={
            "extraction_id": extraction_record.id,
            "extraction_type": extraction_record.extraction_type,
            "extraction_method": extraction_record.extraction_method,
            "extraction_version": extraction_record.extraction_version,
        },
        result=result,
    )


@router.get("/{input_id}/latest")
def get_latest_extraction(
    input_id: str,
    db: Session = Depends(get_db),
):
    record = (
        db.query(DataExtractionRecord)
        .filter(DataExtractionRecord.input_id == input_id)
        .order_by(DataExtractionRecord.created_at.desc())
        .first()
    )

    if record is None:
        raise HTTPException(status_code=404, detail="Extraction result not found")

    return {
        "id": record.id,
        "input_id": record.input_id,
        "source_kind": record.source_kind,
        "extraction_type": record.extraction_type,
        "extraction_method": record.extraction_method,
        "status": record.status,
        "extraction_version": record.extraction_version,
        "transaction_count": record.transaction_count,
        "low_confidence_count": record.low_confidence_count,
        "average_confidence": record.average_confidence,
        "result": record.result_json,
        "debug": record.debug_json,
        "warnings": record.warnings_json,
        "created_at": record.created_at,
    }
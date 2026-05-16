from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.input_record import InputRecord
from app.models.preprocessing_record import InputPreprocessingRecord
from app.services.preprocessing.preprocessing_service import PreprocessingService
from app.workers.celery_app import celery_app


@celery_app.task(name="stage2.process_input")
def process_input(payload: dict):
    input_id = payload["input_id"]
    classification_kind = payload["classification_kind"]

    db: Session = SessionLocal()

    try:
        input_record = db.query(InputRecord).filter(InputRecord.id == input_id).first()

        if input_record is None:
            return {
                "status": "error",
                "reason": "input_record_not_found",
                "input_id": input_id,
            }

        preprocessing_service = PreprocessingService()

        result = preprocessing_service.preprocess(
            input_id=input_record.id,
            source_path=input_record.storage_url,
            source_kind=classification_kind,
        )

        preprocessing_record = InputPreprocessingRecord(
            input_id=input_record.id,
            source_kind=result.source_kind,
            status=result.status,
            output_type=result.output_type,
            output_storage_key=result.output_storage_key,
            output_storage_url=result.output_storage_url,
            preferred_output_storage_key=result.preferred_output_storage_key,
            preferred_output_storage_url=result.preferred_output_storage_url,
            preferred_output_variant=result.preferred_output_variant,
            preferred_extraction_method=result.preferred_extraction_method,
            extraction_risk=result.extraction_risk,
            page_count=result.page_count,
            preprocessing_version=result.preprocessing_version,
            operations_json=result.operations,
            quality_before_json=result.quality_before,
            quality_after_json=result.quality_after,
            outputs_json=[output.model_dump() for output in result.outputs],
            warnings_json=result.warnings,
            average_quality_score_before=result.average_quality_score_before,
            average_quality_score_after=result.average_quality_score_after,
            ocr_readiness_score=result.ocr_readiness_score,
            is_ready_for_extraction=result.is_ready_for_extraction,
        )

        if result.is_ready_for_extraction:
            input_record.status = "preprocessed"
        else:
            input_record.status = "preprocessing_needs_review"

        db.add(preprocessing_record)
        db.add(input_record)
        db.commit()

        return {
            "status": result.status,
            "input_id": input_id,
            "source_kind": result.source_kind,
            "output_type": result.output_type,
            "page_count": result.page_count,
            "preferred_output_variant": result.preferred_output_variant,
            "preferred_extraction_method": result.preferred_extraction_method,
            "extraction_risk": result.extraction_risk,
            "ocr_readiness_score": result.ocr_readiness_score,
            "is_ready_for_extraction": result.is_ready_for_extraction,
            "warnings": result.warnings,
        }

    except Exception as exc:
        db.rollback()

        input_record = db.query(InputRecord).filter(InputRecord.id == input_id).first()

        if input_record is not None:
            input_record.status = "preprocessing_failed"
            db.add(input_record)
            db.commit()

        return {
            "status": "error",
            "input_id": input_id,
            "reason": str(exc),
        }

    finally:
        db.close()
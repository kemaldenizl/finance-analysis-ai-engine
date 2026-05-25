import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from app.models.input_record import Base


class NormalizationRecord(Base):
    __tablename__ = "normalizations"

    id = Column(String, primary_key=True, default=lambda: f"norm_{uuid.uuid4().hex}")

    input_id = Column(String, ForeignKey("inputs.id"), nullable=False)
    extraction_id = Column(String, ForeignKey("data_extractions.id"), nullable=False)

    status = Column(String, nullable=False, default="completed")
    normalization_version = Column(String, nullable=False, default="normalization-v1")

    transaction_count = Column(Integer, nullable=False, default=0)
    duplicate_removed_count = Column(Integer, nullable=False, default=0)
    low_confidence_count = Column(Integer, nullable=False, default=0)

    overall_confidence = Column(Float, nullable=True)

    result_json = Column(JSON, nullable=False, default=dict)
    scores_json = Column(JSON, nullable=False, default=dict)
    warnings_json = Column(JSON, nullable=False, default=list)

    created_at = Column(DateTime, default=datetime.utcnow)

    input_record = relationship("InputRecord")
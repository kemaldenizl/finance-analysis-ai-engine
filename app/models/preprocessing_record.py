import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from app.models.input_record import Base


class InputPreprocessingRecord(Base):
    __tablename__ = "input_preprocessings"

    id = Column(String, primary_key=True, default=lambda: f"prep_{uuid.uuid4().hex}")

    input_id = Column(String, ForeignKey("inputs.id"), nullable=False)

    source_kind = Column(String, nullable=False)
    status = Column(String, nullable=False, default="completed")

    output_type = Column(String, nullable=False)
    output_storage_key = Column(String, nullable=True)
    output_storage_url = Column(String, nullable=True)

    page_count = Column(Integer, nullable=False, default=0)

    preprocessing_version = Column(String, nullable=False)

    operations_json = Column(JSON, nullable=False, default=list)
    quality_before_json = Column(JSON, nullable=False, default=dict)
    quality_after_json = Column(JSON, nullable=False, default=dict)
    outputs_json = Column(JSON, nullable=False, default=list)
    warnings_json = Column(JSON, nullable=False, default=list)

    average_quality_score_before = Column(Float, nullable=True)
    average_quality_score_after = Column(Float, nullable=True)

    is_ready_for_extraction = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    input_record = relationship("InputRecord")
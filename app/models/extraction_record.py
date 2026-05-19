import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from app.models.input_record import Base


class DataExtractionRecord(Base):
    __tablename__ = "data_extractions"

    id = Column(String, primary_key=True, default=lambda: f"ext_{uuid.uuid4().hex}")

    input_id = Column(String, ForeignKey("inputs.id"), nullable=False)

    source_kind = Column(String, nullable=False)
    extraction_type = Column(String, nullable=False)
    extraction_method = Column(String, nullable=False)

    status = Column(String, nullable=False, default="completed")

    extraction_version = Column(String, nullable=False, default="pdf-native-v1")

    transaction_count = Column(Integer, nullable=False, default=0)
    low_confidence_count = Column(Integer, nullable=False, default=0)

    average_confidence = Column(Float, nullable=True)

    result_json = Column(JSON, nullable=False, default=dict)
    debug_json = Column(JSON, nullable=False, default=dict)
    warnings_json = Column(JSON, nullable=False, default=list)

    created_at = Column(DateTime, default=datetime.utcnow)

    input_record = relationship("InputRecord")
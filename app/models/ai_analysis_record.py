import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, JSON, String

from app.models.input_record import Base


class AIAnalysisRecord(Base):
    __tablename__ = "ai_analysis_records"

    id = Column(String, primary_key=True, default=lambda: f"ai_{uuid.uuid4().hex}")

    input_id = Column(String, nullable=False, index=True)

    status = Column(String, nullable=False, default="completed")
    analysis_version = Column(String, nullable=False)

    llm_model = Column(String, nullable=True)
    llm_available = Column(String, nullable=False, default="false")

    analysis_confidence = Column(Float, nullable=True)

    request_json = Column(JSON, nullable=False, default=dict)
    result_json = Column(JSON, nullable=False, default=dict)
    warnings_json = Column(JSON, nullable=False, default=list)

    created_at = Column(DateTime, default=datetime.utcnow)
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class InputRecord(Base):
    __tablename__ = "inputs"

    id = Column(String, primary_key=True, default=lambda: f"inp_{uuid.uuid4().hex}")
    user_id = Column(String, nullable=True)

    original_filename = Column(String, nullable=True)
    mime_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)

    storage_key = Column(String, nullable=False)
    storage_url = Column(String, nullable=True)

    status = Column(String, nullable=False, default="uploaded")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    classification = relationship(
        "InputClassification",
        back_populates="input_record",
        uselist=False,
        cascade="all, delete-orphan",
    )


class InputClassification(Base):
    __tablename__ = "input_classifications"

    id = Column(String, primary_key=True, default=lambda: f"cls_{uuid.uuid4().hex}")

    input_id = Column(String, ForeignKey("inputs.id"), nullable=False)

    kind = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)

    needs_ocr = Column(Boolean, nullable=False)
    needs_preprocessing = Column(Boolean, nullable=False)

    routing_key = Column(String, nullable=False)

    features_json = Column(JSON, nullable=False, default=dict)
    warnings_json = Column(JSON, nullable=False, default=list)

    model_version = Column(String, nullable=False, default="rules-v1")

    created_at = Column(DateTime, default=datetime.utcnow)

    input_record = relationship("InputRecord", back_populates="classification")

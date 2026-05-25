from app.db.session import engine
from app.models.input_record import Base
from app.models import preprocessing_record  # noqa: F401
from app.models import extraction_record  # noqa: F401
from app.models import normalization_record  # noqa: F401

def create_db_tables():
    Base.metadata.create_all(bind=engine)
from app.db.session import engine
from app.models.input_record import Base


def create_db_tables():
    Base.metadata.create_all(bind=engine)

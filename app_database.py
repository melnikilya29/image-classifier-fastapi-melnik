from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime

DATABASE_URL = "sqlite:///./predictions.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionDB(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    predicted_class = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Repository functions
def create_prediction(db: Session, filename: str, pred_class: str, conf: float):
    db_pred = PredictionDB(filename=filename, predicted_class=pred_class, confidence=conf)
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred

def get_all_predictions(db: Session):
    return db.query(PredictionDB).order_by(PredictionDB.created_at.desc()).all()
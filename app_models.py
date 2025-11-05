from pydantic import BaseModel
from datetime import datetime

class PredictionResponse(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True
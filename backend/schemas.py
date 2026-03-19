from pydantic import BaseModel
from typing import Optional

class LoadModelResponse(BaseModel):
    message: str
    crop: str

class PreprocessResponse(BaseModel):
    message: str
    shape: list

class PredictResponse(BaseModel):
    crop: str
    disease: str
    disease_label: str
    confidence: float
    all_scores: dict

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    crop: str
    disease: str
    disease_label: str      # ← added
    confidence: float
    message: str
    history: list[ChatMessage]

class ChatResponse(BaseModel):
    response: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
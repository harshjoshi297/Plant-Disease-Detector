from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from schemas import (
    LoadModelResponse,
    PreprocessResponse,
    PredictResponse,
    ChatRequest,
    ChatResponse,
)
import model as ml
import llm

load_dotenv()

app = FastAPI(title="Plant Disease Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Health check ───────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "message": "Plant Disease Detector API"}

@app.get("/status")
def status():
    return ml.get_status()

# ── /load-model ────────────────────────────────────────────────
@app.post("/load-model", response_model=LoadModelResponse)
def load_model(crop: str = Form(...)):
    result = ml.load_model(crop.lower())
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return LoadModelResponse(**result)

# ── /preprocess ────────────────────────────────────────────────
@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(image: UploadFile = File(...)):
    image_bytes = await image.read()

    from preprocess import validate_image
    if not validate_image(image_bytes):
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = ml.store_preprocessed(image_bytes)
    return PreprocessResponse(**result)

# ── /predict ───────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict():
    result = ml.run_inference()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return PredictResponse(**result)

# ── /chat ──────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.history]

    response = llm.get_chat_response(
        crop         = req.crop,
        disease      = req.disease,
        disease_label= req.disease_label,
        confidence   = req.confidence,
        message      = req.message,
        history      = history
    )
    return ChatResponse(response=response)
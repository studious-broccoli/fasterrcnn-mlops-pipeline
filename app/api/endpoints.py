from fastapi import APIRouter, UploadFile, File
from app.services.inference import run_inference
from app.schemas.request_response import InferenceResponse

router = APIRouter()

@router.post("/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = run_inference(image_bytes)
    return result

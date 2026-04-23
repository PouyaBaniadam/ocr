from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
import io
from PIL import Image
from datetime import datetime
from ocr_manager import get_ocr_provider, MODEL_MAPPING
from logger_config import logger

app = FastAPI(title="Smart Multi-Model OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract-text/")
async def extract_text(
        file: UploadFile = File(...),
        model: str = Query("gemini-2.5-flash", description="Choose a model (e.g., gpt-4o, gemini-2.5-flash)"),
        lang: str = Query("en", description="Target language (e.g., en, fa)")
):
    req_id = str(uuid.uuid4())
    logger.info(f"[{req_id}] Request | Model: {model} | Lang: {lang}")

    if not file.content_type.startswith("image/"):
        return {"id": req_id, "error": "Invalid format. Upload an image.", "status": "failed"}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        ocr_service = get_ocr_provider(model)
        extracted_text = await ocr_service.extract_text(image, language=lang)

        logger.info(f"[{req_id}] OCR successful.")
        return {
            "id": req_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_used": model,
            "text": extracted_text,
            "status": "success"
        }

    except ValueError as ve:
        logger.warning(f"[{req_id}] Validation Error: {str(ve)}")
        return {"id": req_id, "error": str(ve), "status": "failed"}
    except Exception as e:
        logger.critical(f"[{req_id}] System Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/models")
async def list_models():
    return {
        "supported_models": list(MODEL_MAPPING.keys()),
        "total_count": len(MODEL_MAPPING)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
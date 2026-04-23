import io
import uuid
from datetime import datetime

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, HTTPException

from logger_config import logger
from ocr_manager import get_ocr_provider

app = FastAPI(title="Professional Multi-Provider OCR API")


@app.post("/extract-text/")
async def extract_text(
        file: UploadFile = File(...),
        provider: str = Query("gemini", description="AI Provider (gemini, azure, etc.)"),
        lang: str = Query("en", description="Primary language (e.g., en, fa)")
):
    req_id = str(uuid.uuid4())
    logger.info(f"[{req_id}] OCR Request | Provider: {provider} | Lang: {lang} | File: {file.filename}")

    if not file.content_type.startswith("image/"):
        return {"id": req_id, "error": "Invalid file format. Please upload an image.", "status": "failed"}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        ocr_service = get_ocr_provider(provider)

        extracted_text = await ocr_service.extract_text(image, language=lang)

        logger.info(f"[{req_id}] OCR successful.")
        return {
            "id": req_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "provider_used": provider,
            "language": lang,
            "filename": file.filename,
            "text": extracted_text,
            "status": "success"
        }

    except ValueError as ve:
        logger.error(f"[{req_id}] Validation/Config Error: {str(ve)}")
        return {"id": req_id, "error": str(ve), "status": "failed"}
    except Exception as e:
        logger.critical(f"[{req_id}] System Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/providers")
def get_supported_providers():
    return {
        "supported_providers": ["gemini"],
        "coming_soon": ["azure", "openai"]
    }

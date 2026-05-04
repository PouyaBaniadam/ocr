import os
from PIL import Image
from dotenv import load_dotenv
from providers.azure_provider import AzureOCRProvider
from providers.gemini_provider import GeminiOCRProvider

load_dotenv()

MODEL_MAPPING = {
    "gemini-2.5-flash": "gemini",
    "gemini-1.5-pro": "gemini",
    "gemini-1.5-flash": "gemini",
    "gpt-4o": "azure",
    "gpt-4o-mini": "azure"
}

def get_ocr_provider(model_name: str):
    provider_type = MODEL_MAPPING.get(model_name)
    if provider_type == "gemini":
        return GeminiOCRProvider(os.getenv("GEMINI_API_KEY"), model_name)
    elif provider_type == "azure":
        return AzureOCRProvider(os.getenv("AZURE_API_KEY"), "https://models.inference.ai.azure.com", model_name)
    raise ValueError("Unsupported model")

async def process_image_to_text(file_path: str, model_name: str, language: str) -> str:
    provider = get_ocr_provider(model_name)
    with Image.open(file_path) as img:
        return await provider.extract_text(img, language)
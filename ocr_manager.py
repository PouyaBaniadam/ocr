import os

from dotenv import load_dotenv

from providers.azure_provider import AzureOCRProvider
from providers.gemini_provider import GeminiOCRProvider

load_dotenv()

MODEL_MAPPING = {
    "gemini-2.5-flash": "gemini",
    "gemini-1.5-pro": "gemini",
    "gemini-1.5-flash": "gemini",
    "gpt-4o": "azure",
    "gpt-4-turbo": "azure",
    "gpt-4o-mini": "azure"
}


def get_ocr_provider(model_name: str):
    provider_name = MODEL_MAPPING.get(model_name)

    if not provider_name:
        supported = ", ".join(MODEL_MAPPING.keys())
        raise ValueError(f"Model '{model_name}' is not supported. Supported models are: {supported}")

    match provider_name:
        case "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key: raise ValueError("GEMINI_API_KEY is missing.")
            return GeminiOCRProvider(api_key=api_key, model_name=model_name)

        case "azure":
            api_key = os.environ.get("AZURE_API_KEY")
            endpoint = "https://models.inference.ai.azure.com"
            if not api_key: raise ValueError("AZURE_API_KEY is missing.")
            return AzureOCRProvider(api_key=api_key, endpoint=endpoint, model_name=model_name)
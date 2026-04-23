import os

from logger_config import logger
from providers.gemini_provider import GeminiOCRProvider


def get_ocr_provider(provider_name: str):
    provider_name = provider_name.lower()

    match provider_name:
        case "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.critical("GEMINI_API_KEY is missing from environment variables.")
                raise ValueError("API Key for Gemini is not configured.")
            return GeminiOCRProvider(api_key=api_key)

        case "azure":
            pass

        case _:
            raise ValueError(f"Provider '{provider_name}' is not supported yet.")

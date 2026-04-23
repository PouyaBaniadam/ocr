from google import genai
from PIL import Image
from .base import BaseOCRProvider
from logger_config import logger


class GeminiOCRProvider(BaseOCRProvider):
    def __init__(self, api_key: str, model_name: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def extract_text(self, image: Image.Image, language: str) -> str:
        logger.info(f"Sending image to Gemini (Model: {self.model_name} | Lang: {language})...")

        prompt = (
            f"Extract all the text from this image accurately. "
            f"The primary language of the text is {language}. "
            f"Maintain the original formatting, paragraphs, and line breaks where possible. "
            f"Return ONLY the extracted text."
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini API Error: {str(e)}")
            raise ValueError(f"Gemini processing failed: {str(e)}")
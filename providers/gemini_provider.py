import google.generativeai as genai
from PIL import Image
from .base import BaseOCRProvider
from logger_config import logger


class GeminiOCRProvider(BaseOCRProvider):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    async def extract_text(self, image: Image.Image, language: str) -> str:
        logger.info("Sending image to Google Gemini...")

        prompt = (
            f"Extract all the text from this image accurately. "
            f"The primary language of the text is {language}. "
            f"Maintain the original formatting, paragraphs, and line breaks where possible. "
            f"Return ONLY the extracted text without any markdown blocks or extra explanations."
        )

        try:
            response = await self.model.generate_content_async([prompt, image])
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API Error: {str(e)}")
            raise ValueError(f"Failed to process image with Gemini: {str(e)}")
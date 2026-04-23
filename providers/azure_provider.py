import base64
import io
import asyncio
from PIL import Image
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from .base import BaseOCRProvider
from logger_config import logger


class AzureOCRProvider(BaseOCRProvider):
    def __init__(self, api_key: str, endpoint: str, model_name: str):
        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )
        self.model_name = model_name

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buffered = io.BytesIO()
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    async def extract_text(self, image: Image.Image, language: str) -> str:
        logger.info(f"Sending image to Azure (Model: {self.model_name} | Lang: {language})...")

        base64_data = self._image_to_base64(image)

        prompt = (
            f"Extract all the text from this image accurately. "
            f"The primary language of the text is {language}. "
            f"Maintain the original formatting, paragraphs, and line breaks where possible. "
            f"Return ONLY the extracted text."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}
                    }
                ]
            }
        ]

        try:
            response = await asyncio.to_thread(
                self.client.complete,
                model=self.model_name,
                messages=messages,
                temperature=0.1,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Azure API Error: {str(e)}")
            raise ValueError(f"Azure processing failed: {str(e)}")

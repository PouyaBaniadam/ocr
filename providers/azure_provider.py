import base64, io, asyncio
from PIL import Image
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from .base import BaseOCRProvider
from logger_config import logger


class AzureOCRProvider(BaseOCRProvider):
    def __init__(self, api_key: str, endpoint: str, model_name: str):
        self.client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        self.model_name = model_name

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    async def extract_text(self, image: Image.Image, language: str) -> str:
        base64_data = self._image_to_base64(image)
        prompt = f"Extract text in {language}. Return ONLY the text."
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{base64_data}"}}]}]

        response = await asyncio.to_thread(self.client.complete, model=self.model_name, messages=messages)
        return response.choices[0].message.content.strip()
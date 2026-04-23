from abc import ABC, abstractmethod
from PIL import Image

class BaseOCRProvider(ABC):
    @abstractmethod
    async def extract_text(self, image: Image.Image, language: str) -> str:
        """
        All future providers (Azure, OpenAI, etc.) MUST implement this method.
        """
        pass
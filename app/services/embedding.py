import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from app.config import settings

class EmbeddingService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = settings.MODEL_NAME
        self._model = None
        self._processor = None

    @property
    def model(self):
        if self._model is None:
            # Lazy load the model
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            # Lazy load the processor
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._processor

    def extract_embedding(self, image: Image.Image) -> list[float]:
        """
        Extracts a 512-dimensional vector embedding for a PIL Image.
        The resulting vector is L2-normalized.
        """
        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # L2 Normalize the features
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy()[0].tolist()
        return embedding

# Singleton instance
embedding_service = EmbeddingService()

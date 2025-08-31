import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List
import logging

class ClipEmbeddingService:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self.logger = logging.getLogger(__name__)
        self._load_model()

    def _load_model(self):
        """Load the CLIP model and processor."""
        try:
            self.logger.info(f"Loading CLIP model: {self.config.CLIP_MODEL}")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"Using device: {device}")

            self.model = CLIPModel.from_pretrained(self.config.CLIP_MODEL).to(device)
            self.processor = CLIPProcessor.from_pretrained(self.config.CLIP_MODEL)
            
            self.logger.info(f"CLIP model loaded successfully on {device}")

        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {str(e)}")
            raise

    def embed_images(self, images: List[Image.Image]) -> List[List[float]]:
        """
        Generate embeddings for a list of images.

        Args:
            images: List of PIL Images.

        Returns:
            List of image embeddings.
        """
        if not self.model or not self.processor:
            raise RuntimeError("CLIP model not loaded")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            inputs = self.processor(images=images, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize embeddings
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate image embeddings: {str(e)}")
            return []

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings.

        Returns:
            List of text embeddings.
        """
        if not self.model or not self.processor:
            raise RuntimeError("CLIP model not loaded")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

            # Normalize embeddings
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            return text_features.cpu().numpy().tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate text embeddings: {str(e)}")
            return []

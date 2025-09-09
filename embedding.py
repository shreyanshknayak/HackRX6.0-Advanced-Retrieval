# file: embedding.py

import torch
from sentence_transformers import SentenceTransformer
from typing import List

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/stsb-xlm-r-multilingual"

class EmbeddingClient:
    """A client for generating text embeddings using a local sentence transformer model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"EmbeddingClient initialized with model '{model_name}' on device '{self.device}'.")

    def create_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Generates embeddings for a list of text chunks.

        Args:
            texts: A list of strings to be embedded.

        Returns:
            A torch.Tensor containing the generated embeddings.
        """
        if not texts:
            return torch.tensor([])
            
        print(f"Generating embeddings for {len(texts)} text chunks on {self.device}...")
        try:
            embeddings = self.model.encode(
                texts, convert_to_tensor=True, show_progress_bar=False
            )
            print("Embeddings generated successfully.")
            return embeddings
        except Exception as e:
            print(f"An error occurred during embedding generation: {e}")
            raise
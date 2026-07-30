import os
from typing import Any

from dotenv import load_dotenv
from fastembed import TextEmbedding


class FastEmbedder:
    """ONNX-backed text embedder using FastEmbed."""

    def __init__(self, model_name: str | None = None, dimensions: int | None = None):
        load_dotenv()
        self.model_name = model_name or os.getenv(
            "FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"
        )
        self.dimensions = dimensions or int(os.getenv("VECTOR_DIMENSIONS", "384"))
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH")
        kwargs: dict[str, Any] = {"model_name": self.model_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self.model = TextEmbedding(**kwargs)

    def encode(self, text: str) -> list[float]:
        vector = next(self.model.embed([text]))
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        if len(values) != self.dimensions:
            raise ValueError(
                f"embedding dimension {len(values)} does not match "
                f"VECTOR_DIMENSIONS={self.dimensions}"
            )
        return [float(value) for value in values]

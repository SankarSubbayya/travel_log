"""
Image caption and title generation using Ollama's LLaVA model.

This module provides functionality to generate descriptive captions and titles
for travel photos using Ollama's multimodal LLaVA model for true image analysis.

Requires Ollama to be running with LLaVA model:
  ollama pull llava:7b
  ollama serve

This implementation performs actual image analysis, not just text generation.
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Generate captions and titles for images using Ollama's LLaVA model."""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "llava:7b",
        timeout: int = 120,
    ):
        """
        Initialize the caption generator with Ollama LLaVA model.

        Args:
            ollama_base_url: Base URL for Ollama API (default: http://localhost:11434).
            model_name: Model name in Ollama (default: llava:7b).
            timeout: Request timeout in seconds.

        Raises:
            ConnectionError: If Ollama service is not running or not accessible.
        """
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.timeout = timeout
        self.api_endpoint = f"{ollama_base_url}/api/generate"

        # Verify Ollama is running and model is available
        self._verify_ollama_connection()

        logger.info(f"✓ CaptionGenerator initialized with Ollama model: {model_name}")
        logger.info(f"  Using Ollama at: {ollama_base_url}")

    def _verify_ollama_connection(self) -> None:
        """Verify that Ollama service is running and accessible."""
        try:
            response = requests.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=self.timeout,
            )
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            logger.info(f"Available Ollama models: {model_names}")

            if not any(self.model_name in name for name in model_names):
                raise ConnectionError(
                    f"Model '{self.model_name}' not found in Ollama. "
                    f"Pull it with: ollama pull {self.model_name}"
                )

        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.ollama_base_url}. "
                f"Make sure Ollama is running with: ollama serve\n"
                f"Error: {e}"
            ) from e

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """
        Prepare image for model input.

        Args:
            image: PIL Image object.

        Returns:
            Prepared PIL Image in RGB mode, scaled to reasonable size.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize image if too large (to reduce bandwidth and processing time)
        max_dimension = 1024
        if image.width > max_dimension or image.height > max_dimension:
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        return image

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string for Ollama API.

        Args:
            image: PIL Image object.

        Returns:
            Base64 encoded image string.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def _generate_with_image(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str:
        """
        Internal method to generate text using Ollama's multimodal LLaVA.

        Args:
            image: PIL Image object.
            prompt: Input prompt text.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string from image analysis.

        Raises:
            RuntimeError: If Ollama API call fails.
        """
        try:
            # Convert image to base64
            image_b64 = self._image_to_base64(image)

            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False,
            }

            # Call Ollama API
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip()

        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s. "
                "Try increasing timeout or ensuring Ollama has sufficient resources."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}") from e
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Failed to generate response: {e}") from e

    def generate_caption(
        self,
        image: Image.Image,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a detailed caption for an image using LLaVA vision analysis.

        Args:
            image: PIL Image object or path to image file.
            max_tokens: Maximum number of tokens in the caption.
            temperature: Sampling temperature (0.0-1.0). Lower = more deterministic.

        Returns:
            Generated caption string based on actual image analysis.

        Raises:
            ValueError: If image is invalid or unsupported format.
            RuntimeError: If caption generation fails.
        """
        if isinstance(image, (str, Path)):
            try:
                image = Image.open(image)
            except Exception as e:
                raise ValueError(f"Failed to load image: {e}") from e

        image = self._prepare_image(image)

        prompt = (
            "Generate a detailed, vivid description of this image that shows "
            "the actual scene with interesting visual elements, composition, colors, and mood. "
            "The description should be 2-3 sentences and capture the essence of what's shown."
        )
        return self._generate_with_image(image, prompt, max_tokens, temperature)

    def generate_title(
        self,
        image: Image.Image,
        max_tokens: int = 15,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a short title for an image using LLaVA vision analysis.

        Args:
            image: PIL Image object or path to image file.
            max_tokens: Maximum number of tokens in the title (default 15).
            temperature: Sampling temperature (0.0-1.0). Lower = more deterministic.

        Returns:
            Generated title string based on actual image content.

        Raises:
            ValueError: If image is invalid or unsupported format.
            RuntimeError: If title generation fails.
        """
        if isinstance(image, (str, Path)):
            try:
                image = Image.open(image)
            except Exception as e:
                raise ValueError(f"Failed to load image: {e}") from e

        image = self._prepare_image(image)

        prompt = "Provide a short, descriptive title for this image in 2-4 words based on what you see."
        return self._generate_with_image(image, prompt, max_tokens, temperature)

    def generate_travel_caption(
        self,
        image: Image.Image,
        max_tokens: int = 120,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a travel-specific caption for an image using LLaVA vision analysis.

        Args:
            image: PIL Image object or path to image file.
            max_tokens: Maximum number of tokens in the caption.
            temperature: Sampling temperature (0.0-1.0). Lower = more deterministic.

        Returns:
            Generated travel caption string based on actual image analysis.

        Raises:
            ValueError: If image is invalid or unsupported format.
            RuntimeError: If caption generation fails.
        """
        if isinstance(image, (str, Path)):
            try:
                image = Image.open(image)
            except Exception as e:
                raise ValueError(f"Failed to load image: {e}") from e

        image = self._prepare_image(image)

        prompt = (
            "Describe this travel photo in detail, analyzing what you see in the image. "
            "Include: the location/scenery, any activities or people, and interesting details. "
            "Keep it engaging and descriptive."
        )
        return self._generate_with_image(image, prompt, max_tokens, temperature)

    def generate_all(
        self,
        image: Image.Image,
        max_caption_tokens: int = 100,
        max_title_tokens: int = 15,
        temperature: float = 0.7,
    ) -> dict[str, str]:
        """
        Generate caption, title, and travel caption for an image using LLaVA.

        Args:
            image: PIL Image object or path to image file.
            max_caption_tokens: Maximum tokens for detailed caption.
            max_title_tokens: Maximum tokens for title.
            temperature: Sampling temperature (0.0-1.0).

        Returns:
            Dictionary with keys: 'caption', 'title', 'travel_caption'
            All values are based on actual image analysis.

        Raises:
            ValueError: If image is invalid or unsupported format.
            RuntimeError: If any generation step fails.
        """
        if isinstance(image, (str, Path)):
            try:
                image = Image.open(image)
            except Exception as e:
                raise ValueError(f"Failed to load image: {e}") from e

        image = self._prepare_image(image)

        logger.info("Generating caption, title, and travel caption using LLaVA analysis")

        return {
            "caption": self.generate_caption(
                image, max_tokens=max_caption_tokens, temperature=temperature
            ),
            "title": self.generate_title(
                image, max_tokens=max_title_tokens, temperature=temperature
            ),
            "travel_caption": self.generate_travel_caption(
                image, max_tokens=max_caption_tokens, temperature=temperature
            ),
        }

"""
DSPy + LLaVA Integration for Travel Log

This module directly integrates DSPy with LLaVA through Ollama,
allowing DSPy to work with vision-language understanding.
"""

import dspy
from typing import Dict, List, Optional, Union
from PIL import Image
from pathlib import Path
import base64
import io
import json
from datetime import datetime


class OllamaLLaVA(dspy.LM):
    """
    Custom DSPy LM adapter for Ollama's LLaVA model.
    This allows DSPy to directly use LLaVA's vision capabilities.
    """

    def __init__(
        self,
        model: str = "llava:7b",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """Initialize LLaVA model through Ollama."""
        super().__init__(model)
        self.model = model
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        self.kwargs = kwargs
        self.history = []

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 for Ollama API."""
        buffered = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode()

    def basic_request(self, prompt: str, image: Optional[Image.Image] = None, **kwargs) -> Dict:
        """Make request to Ollama LLaVA."""
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **self.kwargs,
            **kwargs
        }

        # Add image if provided
        if image is not None:
            payload["images"] = [self._image_to_base64(image)]

        response = requests.post(self.api_endpoint, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        return {"response": result.get("response", "").strip()}

    def __call__(self, prompt: str, image: Optional[Image.Image] = None, **kwargs):
        """DSPy-compatible call method."""
        result = self.basic_request(prompt, image=image, **kwargs)
        self.history.append({
            "prompt": prompt,
            "response": result["response"],
            "has_image": image is not None
        })
        return [result["response"]]


# Configure DSPy to use both LLaVA (vision) and Llama3 (text reasoning)
llava_model = OllamaLLaVA(model="llava:7b")
# Use dspy.LM with ollama provider format
text_model = dspy.LM(model="ollama/llama3", api_base="http://localhost:11434")


class VisionAnalysisSignature(dspy.Signature):
    """Analyze image content with LLaVA."""
    prompt = dspy.InputField(desc="what to look for in the image")
    image_analysis = dspy.OutputField(desc="detailed visual analysis")


class ContextualCaptionSignature(dspy.Signature):
    """Create contextual captions from visual analysis."""
    visual_content = dspy.InputField(desc="what LLaVA sees in image")
    people_names = dspy.InputField(desc="identified people from face recognition")
    location = dspy.InputField(desc="GPS or location name")
    time = dspy.InputField(desc="when photo was taken")

    scene_type = dspy.OutputField(desc="category of scene")
    mood = dspy.OutputField(desc="emotional tone")
    title = dspy.OutputField(desc="short catchy title (2-4 words)")
    caption = dspy.OutputField(desc="engaging 2-3 sentence caption with names")
    hashtags = dspy.OutputField(desc="relevant hashtags")


class DSPyLLaVACaptionGenerator(dspy.Module):
    """
    Complete caption generator using DSPy with LLaVA.

    Flow:
    1. LLaVA analyzes image visually
    2. DSPy combines visual analysis with context (faces, location, time)
    3. Generates optimized, multi-format captions
    """

    def __init__(self):
        super().__init__()

        # Vision analysis with LLaVA
        self.vision_analyzer = dspy.Predict(VisionAnalysisSignature)

        # Context-aware caption generation with text model
        self.caption_creator = dspy.ChainOfThought(ContextualCaptionSignature)

    def forward(
        self,
        image: Image.Image,
        face_names: Optional[List[str]] = None,
        location: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        analysis_prompt: str = "Describe this photo in detail, including the scene, activities, mood, and any notable elements."
    ) -> Dict[str, str]:
        """
        Generate comprehensive captions using DSPy + LLaVA.

        Args:
            image: PIL Image to analyze
            face_names: Names from face recognition
            location: GPS or location name
            timestamp: Photo timestamp
            analysis_prompt: Custom prompt for LLaVA

        Returns:
            Dictionary with title, caption, scene_type, mood, hashtags
        """

        # Step 1: Use LLaVA for visual analysis
        print("👁️  Analyzing image with LLaVA...")

        # Call LLaVA directly with image
        visual_result = llava_model.basic_request(
            prompt=analysis_prompt,
            image=image
        )
        visual_analysis = visual_result["response"]

        print(f"✓ Visual analysis: {visual_analysis[:100]}...")

        # Step 2: Prepare context
        people_text = ", ".join(face_names) if face_names else "no identified people"
        location_text = location or "unknown location"

        time_text = "unknown time"
        if timestamp:
            time_text = timestamp.strftime("%B %d, %Y at %I:%M %p")
            hour = timestamp.hour
            if 5 <= hour < 12:
                time_text += " (morning)"
            elif 12 <= hour < 17:
                time_text += " (afternoon)"
            elif 17 <= hour < 21:
                time_text += " (evening)"
            else:
                time_text += " (night)"

        # Step 3: Use DSPy to create contextual captions
        print("🧠 Enhancing with DSPy reasoning...")

        # Switch to text model for reasoning
        with dspy.context(lm=text_model):
            result = self.caption_creator(
                visual_content=visual_analysis,
                people_names=people_text,
                location=location_text,
                time=time_text
            )

        return {
            "raw_visual_analysis": visual_analysis,
            "scene_type": result.scene_type,
            "mood": result.mood,
            "title": result.title,
            "caption": result.caption,
            "hashtags": result.hashtags,
            "reasoning": result.rationale if hasattr(result, 'rationale') else None
        }


class SmartTravelPhotoCaptioner(dspy.Module):
    """
    Advanced captioner that adapts prompts based on scene type.
    """

    def __init__(self):
        super().__init__()

        # First pass: detect scene type
        self.scene_detector = dspy.Predict(
            "image_description -> scene_category, key_elements"
        )

        # Specialized generators for different scenes
        self.landscape_captioner = dspy.ChainOfThought(
            "visual_scene, location, time -> scenic_caption, photo_tip"
        )

        self.portrait_captioner = dspy.ChainOfThought(
            "people_description, names, setting, time -> personal_caption, memory_note"
        )

        self.activity_captioner = dspy.ChainOfThought(
            "activity, participants, location, context -> action_caption, story"
        )

    def forward(
        self,
        image: Image.Image,
        face_names: List[str] = None,
        metadata: Dict = None
    ) -> Dict[str, str]:
        """Generate specialized captions based on scene type."""

        # Get visual analysis from LLaVA
        visual_prompt = "Analyze this photo and describe: 1) What type of scene is it? 2) What are the key elements? 3) What's happening?"

        visual_result = llava_model.basic_request(
            prompt=visual_prompt,
            image=image
        )
        visual_analysis = visual_result["response"]

        # Detect scene type
        with dspy.context(lm=text_model):
            scene = self.scene_detector(image_description=visual_analysis)
            scene_type = scene.scene_category.lower()

            # Choose appropriate captioner
            if "landscape" in scene_type or "scenery" in scene_type:
                result = self.landscape_captioner(
                    visual_scene=visual_analysis,
                    location=metadata.get("location", "unknown"),
                    time=metadata.get("time", "unknown")
                )
                return {
                    "type": "landscape",
                    "caption": result.scenic_caption,
                    "extra": result.photo_tip
                }

            elif face_names and len(face_names) > 0:
                result = self.portrait_captioner(
                    people_description=visual_analysis,
                    names=", ".join(face_names),
                    setting=metadata.get("location", "unknown"),
                    time=metadata.get("time", "unknown")
                )
                return {
                    "type": "portrait",
                    "caption": result.personal_caption,
                    "extra": result.memory_note
                }
            else:
                result = self.activity_captioner(
                    activity=visual_analysis,
                    participants=", ".join(face_names) if face_names else "unknown",
                    location=metadata.get("location", "unknown"),
                    context=scene.key_elements
                )
                return {
                    "type": "activity",
                    "caption": result.action_caption,
                    "extra": result.story
                }


class BatchPhotoAnalyzer(dspy.Module):
    """
    Analyze multiple photos to find patterns and create narratives.
    Combines vision understanding with temporal reasoning.
    """

    def __init__(self):
        super().__init__()

        self.event_grouper = dspy.ChainOfThought(
            "photo_descriptions, timestamps, locations -> event_groups, event_names"
        )

        self.narrative_builder = dspy.ChainOfThought(
            "event_name, photo_sequence, people_involved -> travel_narrative"
        )

    def analyze_photo_collection(
        self,
        photos: List[Dict]  # Each dict has: image, caption, timestamp, location
    ) -> Dict:
        """
        Analyze a collection of photos to create a travel story.

        Args:
            photos: List of dicts with 'image', 'caption', 'timestamp', 'location'

        Returns:
            Dict with grouped events and narrative
        """

        # Analyze each photo with LLaVA
        analyses = []
        for photo in photos:
            result = llava_model.basic_request(
                prompt="Briefly describe what's happening in this photo.",
                image=photo['image']
            )
            analyses.append(result["response"])

        # Prepare data for grouping
        descriptions = " | ".join(analyses)
        timestamps = " | ".join(str(p.get("timestamp", "")) for p in photos)
        locations = " | ".join(str(p.get("location", "")) for p in photos)

        # Group into events
        with dspy.context(lm=text_model):
            events = self.event_grouper(
                photo_descriptions=descriptions,
                timestamps=timestamps,
                locations=locations
            )

            # Create narrative
            narrative = self.narrative_builder(
                event_name=events.event_names,
                photo_sequence=descriptions,
                people_involved="varies"  # Could extract from face recognition
            )

        return {
            "event_groups": events.event_groups,
            "event_names": events.event_names,
            "narrative": narrative.travel_narrative,
            "photo_count": len(photos)
        }


# Convenience function for easy integration
def generate_smart_captions(
    image: Union[str, Path, Image.Image],
    face_names: Optional[List[str]] = None,
    location: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> Dict[str, str]:
    """
    Easy-to-use function for generating captions with DSPy + LLaVA.

    Usage:
        from travel_log.dspy_llava_integration import generate_smart_captions

        captions = generate_smart_captions(
            image="photo.jpg",
            face_names=["Sarah", "John"],
            location="Golden Gate Bridge, SF",
            timestamp=datetime.now()
        )

        print(captions['title'])
        print(captions['caption'])
    """

    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    # Initialize generator
    generator = DSPyLLaVACaptionGenerator()

    # Generate captions
    return generator.forward(
        image=image,
        face_names=face_names,
        location=location,
        timestamp=timestamp
    )


if __name__ == "__main__":
    print("DSPy + LLaVA Integration for Travel Log")
    print("=" * 60)
    print()
    print("This module combines:")
    print("  • LLaVA's vision capabilities (sees the image)")
    print("  • DSPy's reasoning capabilities (understands context)")
    print("  • Face recognition results (who's in the photo)")
    print("  • EXIF metadata (where and when)")
    print()
    print("To create intelligent, contextual captions!")
    print()
    print("=" * 60)

    # Example usage would go here
    # Requires: Ollama running with both llava:7b and llama3
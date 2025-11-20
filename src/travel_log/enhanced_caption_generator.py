"""
Enhanced Caption Generator combining LLaVA's vision with DSPy's reasoning.

This module shows how DSPy enhances LLaVA's raw visual understanding
with intelligent reasoning, context integration, and optimization.
"""

import dspy
from typing import Dict, List, Optional
from PIL import Image
from pathlib import Path
import json

# Import your existing caption generator
from .caption_generator import CaptionGenerator


# Configure DSPy with Ollama (using text model for reasoning)
ollama_text = dspy.OllamaLocal(
    model="llama3",  # Text model for reasoning
    base_url="http://localhost:11434"
)
dspy.settings.configure(lm=ollama_text)


class EnhancedCaptionSignature(dspy.Signature):
    """Transform raw visual description into rich, contextual captions."""

    # Input from LLaVA's visual analysis
    visual_description = dspy.InputField(
        desc="what LLaVA sees in the image"
    )

    # Additional context we can provide
    identified_people = dspy.InputField(
        desc="names of people identified via face recognition"
    )
    location_info = dspy.InputField(
        desc="GPS location or place name from EXIF"
    )
    time_info = dspy.InputField(
        desc="date and time when photo was taken"
    )
    photo_context = dspy.InputField(
        desc="other photos from same event/time"
    )

    # Rich outputs that LLaVA alone doesn't provide
    scene_category = dspy.OutputField(
        desc="type: landscape/portrait/group/activity/food/architecture"
    )
    emotional_tone = dspy.OutputField(
        desc="mood: joyful/peaceful/adventurous/nostalgic/celebratory"
    )
    key_moments = dspy.OutputField(
        desc="what's happening: 'sharing meal', 'sunset viewing', etc"
    )
    personalized_caption = dspy.OutputField(
        desc="caption mentioning people by name and their actions"
    )
    social_media_caption = dspy.OutputField(
        desc="Instagram-ready caption with emojis and hashtags"
    )
    memory_description = dspy.OutputField(
        desc="personal memory note for photo album"
    )


class EnhancedCaptionGenerator:
    """
    Combines LLaVA's vision capabilities with DSPy's reasoning
    to create rich, multi-purpose captions.
    """

    def __init__(self):
        # Original LLaVA-based generator
        self.llava_generator = CaptionGenerator()

        # DSPy module for enhanced reasoning
        self.enhancer = dspy.ChainOfThought(EnhancedCaptionSignature)

        # Specialized modules for different purposes
        self.story_connector = dspy.Predict(
            "previous_photo, current_photo, next_photo -> narrative_connection"
        )

        self.location_enricher = dspy.Predict(
            "visual_scene, gps_coords -> location_description, landmark_guess"
        )

    def generate_rich_captions(
        self,
        image: Image.Image,
        face_names: Optional[List[str]] = None,
        location: Optional[Dict] = None,
        timestamp: Optional[str] = None,
        nearby_photos: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive captions using LLaVA + DSPy.

        Args:
            image: PIL Image to analyze
            face_names: Names from face recognition
            location: GPS/location metadata
            timestamp: When photo was taken
            nearby_photos: Captions of temporally close photos

        Returns:
            Rich caption dictionary with multiple formats
        """

        # Step 1: Get raw visual description from LLaVA
        print("🔍 Analyzing image with LLaVA...")
        llava_caption = self.llava_generator.generate_caption(image)

        # Step 2: Use DSPy to enhance with context and reasoning
        print("🧠 Enhancing with DSPy reasoning...")

        # Prepare context
        people_text = ", ".join(face_names) if face_names else "unidentified people"
        location_text = self._format_location(location) if location else "unknown location"
        time_text = timestamp or "unknown time"
        context_text = "; ".join(nearby_photos[:3]) if nearby_photos else "standalone photo"

        # Generate enhanced captions
        enhanced = self.enhancer(
            visual_description=llava_caption,
            identified_people=people_text,
            location_info=location_text,
            time_info=time_text,
            photo_context=context_text
        )

        return {
            # Original from LLaVA
            "llava_raw": llava_caption,

            # Enhanced by DSPy
            "scene_type": enhanced.scene_category,
            "mood": enhanced.emotional_tone,
            "activity": enhanced.key_moments,
            "personal_caption": enhanced.personalized_caption,
            "social_caption": enhanced.social_media_caption,
            "memory_note": enhanced.memory_description,

            # Reasoning trace (helpful for debugging)
            "reasoning": enhanced.completions.rationale if hasattr(enhanced.completions, 'rationale') else None
        }

    def generate_trip_narrative(
        self,
        photos_with_captions: List[Dict]
    ) -> str:
        """
        Generate a cohesive travel story from multiple photos.
        This is where DSPy really shines - connecting individual
        observations into a narrative.
        """

        narrative_parts = []

        for i, photo in enumerate(photos_with_captions):
            if i > 0:
                # Connect to previous photo
                connection = self.story_connector(
                    previous_photo=photos_with_captions[i-1]["personal_caption"],
                    current_photo=photo["personal_caption"],
                    next_photo=photos_with_captions[i+1]["personal_caption"] if i < len(photos_with_captions)-1 else "end of trip"
                )
                narrative_parts.append(connection.narrative_connection)
            else:
                narrative_parts.append(photo["personal_caption"])

        return " ".join(narrative_parts)

    def _format_location(self, location: Dict) -> str:
        """Format location data for context."""
        if isinstance(location, dict):
            if 'latitude' in location and 'longitude' in location:
                return f"GPS: {location['latitude']:.4f}, {location['longitude']:.4f}"
            elif 'name' in location:
                return location['name']
        return str(location)


class DSPyLLaVAComparison:
    """
    Demonstrates the difference between LLaVA alone vs LLaVA+DSPy.
    """

    @staticmethod
    def show_difference(image_path: str):
        """
        Show what each component contributes.
        """
        image = Image.open(image_path)

        print("=" * 60)
        print("LLaVA ALONE vs LLaVA + DSPy")
        print("=" * 60)

        # LLaVA alone
        basic_gen = CaptionGenerator()
        llava_only = basic_gen.generate_caption(image)

        print("\n📷 LLaVA ALONE (Visual Understanding):")
        print(f"Caption: {llava_only}")
        print("\nWhat LLaVA provides:")
        print("✓ Describes what's visually in the image")
        print("✗ No personalization with names")
        print("✗ No context from other photos")
        print("✗ No social media optimization")
        print("✗ No emotional interpretation")

        # LLaVA + DSPy
        enhanced_gen = EnhancedCaptionGenerator()
        enhanced = enhanced_gen.generate_rich_captions(
            image,
            face_names=["Sarah", "John"],  # From face recognition
            location={"name": "Sunset Beach, California"},  # From GPS
            timestamp="2024-11-08 18:30",  # From EXIF
            nearby_photos=["Beach walk earlier", "Dinner at seafood restaurant"]  # Context
        )

        print("\n🧠 LLaVA + DSPy (Visual + Intelligent Reasoning):")
        print(f"Scene Type: {enhanced['scene_type']}")
        print(f"Mood: {enhanced['mood']}")
        print(f"Activity: {enhanced['activity']}")
        print(f"Personal: {enhanced['personal_caption']}")
        print(f"Social Media: {enhanced['social_caption']}")
        print(f"Memory: {enhanced['memory_note']}")

        print("\nWhat DSPy adds:")
        print("✓ Integrates face recognition (names people)")
        print("✓ Uses location context")
        print("✓ Considers time of day")
        print("✓ Connects to nearby photos")
        print("✓ Generates multiple caption styles")
        print("✓ Adds emotional understanding")
        print("✓ Creates social media ready content")

        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("- LLaVA: Sees and describes the image")
        print("- DSPy: Thinks about what the image means in context")
        print("- Together: Complete understanding + intelligent output")
        print("=" * 60)


# Example: How to integrate into existing app
def upgrade_existing_caption_generator(
    image: Image.Image,
    face_recognition_results: Optional[List[str]] = None,
    exif_metadata: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Drop-in replacement for current caption generator
    that adds DSPy enhancements.
    """

    # Parse metadata
    location = None
    timestamp = None

    if exif_metadata:
        if 'latitude' in exif_metadata and 'longitude' in exif_metadata:
            location = {
                'latitude': exif_metadata['latitude'],
                'longitude': exif_metadata['longitude']
            }
        if 'datetime_str' in exif_metadata:
            timestamp = exif_metadata['datetime_str']

    # Generate enhanced captions
    generator = EnhancedCaptionGenerator()
    return generator.generate_rich_captions(
        image=image,
        face_names=face_recognition_results,
        location=location,
        timestamp=timestamp
    )


if __name__ == "__main__":
    print("Enhanced Caption Generator - LLaVA + DSPy Demo")
    print("-" * 50)

    # This would show the comparison if you had a test image
    # DSPyLLaVAComparison.show_difference("test_image.jpg")

    print("\nKey Benefits of DSPy with LLaVA:")
    print("1. LLaVA tells you WHAT is in the image")
    print("2. DSPy figures out WHY it matters and HOW to describe it")
    print("3. Together: Smarter, more contextual, more useful captions")
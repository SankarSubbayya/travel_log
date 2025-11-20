"""
DSPy-powered caption optimization for Travel Log.

This module uses DSPy to create optimized, context-aware captions
that combine visual analysis, face recognition, and metadata.
"""

import dspy
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

# Configure DSPy with Ollama
ollama_llm = dspy.OllamaLocal(
    model="llama3",  # or any text model
    base_url="http://localhost:11434"
)
dspy.settings.configure(lm=ollama_llm)


class TravelPhotoSignature(dspy.Signature):
    """Generate comprehensive travel photo analysis."""

    visual_elements = dspy.InputField(desc="what's visible in the photo")
    detected_faces = dspy.InputField(desc="names of people identified")
    location_data = dspy.InputField(desc="GPS coordinates or location name")
    time_of_day = dspy.InputField(desc="morning/afternoon/evening/night")

    scene_type = dspy.OutputField(desc="type of scene: landscape/portrait/group/activity")
    mood = dspy.OutputField(desc="emotional tone: joyful/serene/adventurous/contemplative")
    title = dspy.OutputField(desc="catchy 2-4 word title")
    caption = dspy.OutputField(desc="engaging 2-3 sentence description")
    social_tags = dspy.OutputField(desc="relevant hashtags for social media")


class SmartTravelCaptionGenerator(dspy.Module):
    """
    DSPy module for intelligent travel photo caption generation.
    Combines multiple aspects of a photo into cohesive, engaging captions.
    """

    def __init__(self, style: str = "balanced"):
        """
        Initialize the caption generator.

        Args:
            style: Caption style - 'casual', 'professional', 'poetic', 'humorous'
        """
        super().__init__()
        self.style = style

        # Main caption generation with reasoning
        self.generate = dspy.ChainOfThought(TravelPhotoSignature)

        # Style adapter
        self.style_adapter = dspy.Predict(
            "caption, target_style -> styled_caption"
        )

        # Memory for learning from feedback
        self.feedback_memory = []

    def forward(
        self,
        visual_description: str,
        face_names: List[str],
        location: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, str]:
        """
        Generate optimized captions for a travel photo.

        Args:
            visual_description: What LLaVA sees in the image
            face_names: List of identified people
            location: GPS or location name
            timestamp: When photo was taken

        Returns:
            Dictionary with title, caption, mood, tags
        """
        # Prepare inputs
        faces_text = ", ".join(face_names) if face_names else "no identified faces"
        location_text = location or "unknown location"

        time_of_day = "unknown"
        if timestamp:
            hour = timestamp.hour
            if 5 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 17:
                time_of_day = "afternoon"
            elif 17 <= hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"

        # Generate comprehensive analysis
        result = self.generate(
            visual_elements=visual_description,
            detected_faces=faces_text,
            location_data=location_text,
            time_of_day=time_of_day
        )

        # Apply style if not balanced
        styled_caption = result.caption
        if self.style != "balanced":
            styled = self.style_adapter(
                caption=result.caption,
                target_style=self.style
            )
            styled_caption = styled.styled_caption

        return {
            "title": result.title,
            "caption": styled_caption,
            "scene_type": result.scene_type,
            "mood": result.mood,
            "hashtags": result.social_tags,
            "original_caption": result.caption
        }

    def learn_from_feedback(self, original: Dict, edited: str, rating: int):
        """
        Learn from user edits and ratings.

        Args:
            original: Original generated captions
            edited: User's edited version
            rating: User rating (1-5)
        """
        self.feedback_memory.append({
            "original": original,
            "edited": edited,
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        })

        # Could retrain with DSPy optimizers here
        if len(self.feedback_memory) >= 10:
            self._optimize_from_feedback()

    def _optimize_from_feedback(self):
        """Use feedback to improve generation."""
        # This would use DSPy's optimization capabilities
        # to improve based on user preferences
        pass


class PhotoBatchAnalyzer(dspy.Module):
    """
    Analyze batches of photos to find patterns and create stories.
    """

    def __init__(self):
        super().__init__()

        # Detect events/trips from photo batches
        self.event_detector = dspy.ChainOfThought(
            "photo_list, timestamps, locations -> event_groups, event_names"
        )

        # Generate travel story from grouped photos
        self.story_generator = dspy.ChainOfThought(
            "event_name, photo_captions, timeline -> travel_story"
        )

        # Suggest photo album organization
        self.album_organizer = dspy.Predict(
            "events, key_photos -> album_structure, cover_photo"
        )

    def analyze_trip(
        self,
        photos: List[Dict],
        generate_story: bool = True
    ) -> Dict:
        """
        Analyze a batch of photos from a trip.

        Args:
            photos: List of photo metadata and captions
            generate_story: Whether to generate a narrative

        Returns:
            Analysis results including events, story, organization
        """
        # Prepare photo data
        photo_list = [p.get("caption", "") for p in photos]
        timestamps = [p.get("timestamp", "") for p in photos]
        locations = [p.get("location", "") for p in photos]

        # Detect events
        events = self.event_detector(
            photo_list="; ".join(photo_list),
            timestamps=", ".join(str(t) for t in timestamps),
            locations=", ".join(locations)
        )

        results = {
            "events": events.event_groups,
            "event_names": events.event_names
        }

        # Generate story if requested
        if generate_story and events.event_groups:
            story = self.story_generator(
                event_name=events.event_names,
                photo_captions=photo_list,
                timeline=f"{timestamps[0]} to {timestamps[-1]}"
            )
            results["story"] = story.travel_story

        # Suggest album organization
        album = self.album_organizer(
            events=events.event_groups,
            key_photos=photo_list[:5]  # First 5 as candidates
        )
        results["album_structure"] = album.album_structure
        results["suggested_cover"] = album.cover_photo

        return results


class FaceRecognitionOptimizer(dspy.Module):
    """
    Optimize face recognition thresholds and decisions using DSPy.
    """

    def __init__(self):
        super().__init__()

        # Intelligent threshold determination
        self.threshold_optimizer = dspy.ChainOfThought(
            "distance, image_quality, lighting, angle -> should_match, confidence"
        )

        # Handle ambiguous cases
        self.ambiguity_resolver = dspy.ChainOfThought(
            "face_distances, context_clues, photo_metadata -> best_match, reasoning"
        )

    def optimize_match_decision(
        self,
        embedding_distance: float,
        face_quality: float,
        lighting_score: float,
        face_angle: str
    ) -> Tuple[bool, float, str]:
        """
        Make intelligent face matching decisions.

        Returns:
            Tuple of (should_match, confidence, reasoning)
        """
        result = self.threshold_optimizer(
            distance=str(embedding_distance),
            image_quality=str(face_quality),
            lighting=str(lighting_score),
            angle=face_angle
        )

        should_match = result.should_match.lower() == "yes"
        confidence = float(result.confidence)

        return should_match, confidence, result.completions.rationale

    def resolve_ambiguous_match(
        self,
        face_distances: Dict[str, float],
        photo_context: str,
        metadata: Dict
    ) -> Tuple[str, str]:
        """
        Resolve cases where multiple people have similar distances.

        Returns:
            Tuple of (best_match_name, reasoning)
        """
        distances_str = ", ".join(
            f"{name}: {dist:.3f}" for name, dist in face_distances.items()
        )

        result = self.ambiguity_resolver(
            face_distances=distances_str,
            context_clues=photo_context,
            photo_metadata=json.dumps(metadata)
        )

        return result.best_match, result.reasoning


# Example usage function
def integrate_dspy_with_travel_log(
    image_analysis: str,
    face_names: List[str],
    location: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    style: str = "balanced"
) -> Dict[str, str]:
    """
    Easy integration function for the Travel Log app.

    Args:
        image_analysis: Visual description from LLaVA
        face_names: Names from face recognition
        location: GPS or place name
        timestamp: Photo timestamp
        style: Caption style preference

    Returns:
        Optimized captions and metadata
    """
    generator = SmartTravelCaptionGenerator(style=style)
    return generator.forward(
        visual_description=image_analysis,
        face_names=face_names,
        location=location,
        timestamp=timestamp
    )


# Training function for optimization
def train_caption_generator(training_examples: List[Dict]) -> SmartTravelCaptionGenerator:
    """
    Train the caption generator with examples.

    Args:
        training_examples: List of dicts with 'input' and 'expected_output'

    Returns:
        Optimized generator
    """
    from dspy.teleprompt import BootstrapFewShot

    generator = SmartTravelCaptionGenerator()

    # Define metric for good captions
    def caption_quality_metric(example, prediction, trace=None):
        # Check if all required fields are present
        required = ["title", "caption", "mood", "hashtags"]
        if not all(field in prediction for field in required):
            return 0.0

        # Check quality criteria
        score = 0.0
        if len(prediction["title"].split()) <= 4:
            score += 0.25
        if 20 <= len(prediction["caption"].split()) <= 60:
            score += 0.25
        if prediction["mood"] in ["joyful", "serene", "adventurous", "contemplative"]:
            score += 0.25
        if "#" in prediction["hashtags"]:
            score += 0.25

        return score

    # Optimize with few-shot examples
    optimizer = BootstrapFewShot(
        metric=caption_quality_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4
    )

    optimized_generator = optimizer.compile(
        generator,
        trainset=training_examples
    )

    return optimized_generator


if __name__ == "__main__":
    # Example usage
    print("DSPy Caption Optimizer for Travel Log")
    print("-" * 50)

    # Test the generator
    generator = SmartTravelCaptionGenerator(style="casual")

    result = generator.forward(
        visual_description="A sunset beach scene with palm trees and waves",
        face_names=["John", "Sarah"],
        location="Maldives",
        timestamp=datetime(2024, 11, 8, 18, 30)
    )

    print("Generated Caption:")
    print(f"Title: {result['title']}")
    print(f"Caption: {result['caption']}")
    print(f"Mood: {result['mood']}")
    print(f"Tags: {result['hashtags']}")
#!/usr/bin/env python3
"""
Demo: Using DSPy with LLaVA for intelligent caption generation

This example shows how DSPy enhances LLaVA's visual understanding
with contextual reasoning for better captions.

Requirements:
    1. Ollama running (ollama serve)
    2. LLaVA model (ollama pull llava:7b)
    3. Llama3 model (ollama pull llama3)
    4. DSPy installed (pip install dspy-ai)
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_log.dspy_llava_integration import (
    generate_smart_captions,
    DSPyLLaVACaptionGenerator,
    SmartTravelPhotoCaptioner
)
from PIL import Image


def demo_basic_usage():
    """
    Demo 1: Basic caption generation with DSPy + LLaVA
    """
    print("\n" + "="*60)
    print("DEMO 1: Basic Smart Caption Generation")
    print("="*60)

    # Find a test image
    test_images = list(Path("~/personal_photos").expanduser().glob("*.jpg"))
    if not test_images:
        test_images = list(Path("~/personal_photos").expanduser().glob("*.jpeg"))

    if not test_images:
        print("❌ No test images found in ~/personal_photos")
        print("Please provide path to a photo:")
        return

    test_image = test_images[0]
    print(f"\n📸 Analyzing: {test_image.name}")

    # Generate captions
    try:
        captions = generate_smart_captions(
            image=test_image,
            face_names=["Sarah", "John"],  # Simulated face recognition
            location="Golden Gate Bridge, San Francisco",
            timestamp=datetime(2024, 11, 8, 15, 30)
        )

        print("\n📝 Generated Captions:")
        print(f"\n  Title: {captions['title']}")
        print(f"\n  Caption: {captions['caption']}")
        print(f"\n  Scene Type: {captions['scene_type']}")
        print(f"\n  Mood: {captions['mood']}")
        print(f"\n  Hashtags: {captions['hashtags']}")

        if captions.get('reasoning'):
            print(f"\n  🧠 DSPy Reasoning: {captions['reasoning']}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. Models are available: ollama list")
        print("  3. DSPy is installed: pip install dspy-ai")


def demo_comparison():
    """
    Demo 2: Compare LLaVA alone vs LLaVA + DSPy
    """
    print("\n" + "="*60)
    print("DEMO 2: LLaVA vs LLaVA+DSPy Comparison")
    print("="*60)

    test_images = list(Path("~/personal_photos").expanduser().glob("*.jpg"))
    if not test_images:
        print("❌ No test images found")
        return

    test_image = test_images[0]
    image = Image.open(test_image)

    print(f"\n📸 Analyzing: {test_image.name}\n")

    # Method 1: Just LLaVA
    print("─" * 60)
    print("METHOD 1: LLaVA Alone")
    print("─" * 60)

    try:
        from travel_log.caption_generator import CaptionGenerator

        basic_gen = CaptionGenerator()
        basic_result = basic_gen.generate_all(image)

        print(f"Title: {basic_result['title']}")
        print(f"Caption: {basic_result['caption']}")
        print(f"Travel: {basic_result['travel_caption']}")

        print("\n✓ Pure visual analysis")
        print("✗ No face name integration")
        print("✗ No location context")
        print("✗ No temporal reasoning")

    except Exception as e:
        print(f"Error with basic generator: {e}")

    # Method 2: LLaVA + DSPy
    print("\n" + "─" * 60)
    print("METHOD 2: LLaVA + DSPy")
    print("─" * 60)

    try:
        enhanced = generate_smart_captions(
            image=image,
            face_names=["Mom", "Dad"],
            location="Yosemite National Park",
            timestamp=datetime(2024, 7, 15, 14, 20)
        )

        print(f"Title: {enhanced['title']}")
        print(f"Caption: {enhanced['caption']}")
        print(f"Scene: {enhanced['scene_type']}")
        print(f"Mood: {enhanced['mood']}")
        print(f"Tags: {enhanced['hashtags']}")

        print("\n✓ Visual analysis (LLaVA)")
        print("✓ Personalized with names")
        print("✓ Location aware")
        print("✓ Time contextualized")
        print("✓ Multiple output formats")

    except Exception as e:
        print(f"Error with enhanced generator: {e}")

    print("\n" + "="*60)
    print("WINNER: LLaVA + DSPy")
    print("Reason: Combines vision with context for richer output")
    print("="*60)


def demo_adaptive_captions():
    """
    Demo 3: Adaptive captions based on scene type
    """
    print("\n" + "="*60)
    print("DEMO 3: Adaptive Captions by Scene Type")
    print("="*60)

    test_images = list(Path("~/personal_photos").expanduser().glob("*.jpg"))[:3]

    if not test_images:
        print("❌ No test images found")
        return

    captioner = SmartTravelPhotoCaptioner()

    for img_path in test_images:
        print(f"\n📸 {img_path.name}")
        print("─" * 40)

        image = Image.open(img_path)

        try:
            result = captioner.forward(
                image=image,
                face_names=["Alex"],
                metadata={
                    "location": "California Coast",
                    "time": "Evening"
                }
            )

            print(f"Scene Type: {result['type']}")
            print(f"Caption: {result['caption']}")
            print(f"Extra: {result['extra']}")

        except Exception as e:
            print(f"Error: {e}")


def demo_integration_with_face_recognition():
    """
    Demo 4: Full integration with face recognition system
    """
    print("\n" + "="*60)
    print("DEMO 4: Integration with Face Recognition")
    print("="*60)

    try:
        from travel_log import FaceDetector, FaceLabeler

        # Initialize face system
        detector = FaceDetector(detector_backend='mtcnn')
        labeler = FaceLabeler(
            database_path='./face_database',
            model_name='Facenet512'
        )

        # Get test image
        test_images = list(Path("~/personal_photos").expanduser().glob("*.jpg"))
        if not test_images:
            print("❌ No test images")
            return

        test_image = test_images[0]
        print(f"\n📸 Processing: {test_image.name}")

        # Detect and identify faces
        faces = detector.extract_faces(str(test_image))
        print(f"✓ Detected {len(faces)} faces")

        # Identify faces
        face_names = []
        if faces:
            matches = labeler.find_face(str(test_image))
            if matches and len(matches) > 0 and not matches[0].empty:
                for idx, match in matches[0].iterrows():
                    person = Path(match['identity']).parent.name
                    face_names.append(person)
                    print(f"  - {person} (confidence: {1-match['distance']:.2%})")

        # Generate smart captions with recognized names
        image = Image.open(test_image)
        captions = generate_smart_captions(
            image=image,
            face_names=face_names if face_names else None,
            location="Unknown",  # Could extract from EXIF
            timestamp=datetime.now()
        )

        print(f"\n📝 Smart Caption:")
        print(f"  {captions['caption']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure face database is set up")


def main():
    """Run all demos."""
    print("\n" + "🎯" * 30)
    print("DSPy + LLaVA Integration Demo")
    print("Travel Log - Smart Caption Generation")
    print("🎯" * 30)

    # Check prerequisites
    print("\n📋 Checking prerequisites...")

    try:
        import dspy
        print("✓ DSPy installed")
    except ImportError:
        print("❌ DSPy not installed")
        print("   Install with: pip install dspy-ai")
        return

    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✓ Ollama is running")
            models = response.json().get("models", [])
            model_names = [m['name'] for m in models]

            if any('llava' in m for m in model_names):
                print("✓ LLaVA model available")
            else:
                print("❌ LLaVA model not found")
                print("   Install with: ollama pull llava:7b")

            if any('llama3' in m for m in model_names):
                print("✓ Llama3 model available")
            else:
                print("⚠️  Llama3 not found (recommended)")
                print("   Install with: ollama pull llama3")
        else:
            print("❌ Ollama not responding")
    except:
        print("❌ Ollama not running")
        print("   Start with: ollama serve")
        return

    # Run demos
    print("\n" + "="*60)
    print("Running Demos...")
    print("="*60)

    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Comparison", demo_comparison),
        ("Adaptive Captions", demo_adaptive_captions),
        ("Face Recognition Integration", demo_integration_with_face_recognition)
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except KeyboardInterrupt:
            print("\n\n⏸️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Demo '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("✅ Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
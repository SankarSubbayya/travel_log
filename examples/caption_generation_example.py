#!/usr/bin/env python3
"""
Caption Generation Example with Ollama's LLaVA

Demonstrates how to use Ollama's LLaVA model to generate captions and titles for travel photos.

Before running this example:
1. Install Ollama from https://ollama.ai
2. Pull the LLaVA model: ollama pull llava:7b
3. Start Ollama: ollama serve
4. Then run this script in another terminal

This example shows:
1. Connecting to Ollama
2. Generating different types of captions (title, detailed caption, travel caption)
3. Processing multiple images
4. Saving captions to files
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_log.caption_generator import CaptionGenerator


def example_single_image():
    """Generate captions for a single image."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Image Caption Generation")
    print("=" * 70)

    # Initialize caption generator
    print("\nConnecting to Ollama and initializing caption generator...")
    try:
        generator = CaptionGenerator()
        print("✅ Connected to Ollama! Caption generator initialized!")
    except ConnectionError as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("\nTo use this example:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Pull the model: ollama pull llava:7b")
        print("  3. Start Ollama: ollama serve")
        print("  4. Then run this script")
        return

    # Example image path (you would use an actual image)
    image_path = Path("path/to/your/travel/photo.jpg")

    if not image_path.exists():
        print(f"⚠️  Example image not found at {image_path}")
        print("To test, replace 'path/to/your/travel/photo.jpg' with an actual image path")
        return

    print(f"\nProcessing image: {image_path.name}")

    # Generate a title
    print("\n📝 Generating title...")
    title = generator.generate_title(image_path)
    print(f"Title: {title}")

    # Generate a detailed caption
    print("\n📖 Generating detailed caption...")
    caption = generator.generate_caption(image_path)
    print(f"Caption: {caption}")

    # Generate a travel-specific caption
    print("\n✈️ Generating travel caption...")
    travel_caption = generator.generate_travel_caption(image_path)
    print(f"Travel Caption: {travel_caption}")

    # Generate all captions at once
    print("\n✨ Generating all captions...")
    all_captions = generator.generate_all(image_path)
    print("\n📋 All Captions:")
    for key, value in all_captions.items():
        print(f"\n  {key.replace('_', ' ').title()}:")
        print(f"  {value}")


def example_batch_processing():
    """Generate captions for multiple images in a directory."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Batch Processing Multiple Images")
    print("=" * 70)

    # Initialize caption generator
    print("\nConnecting to Ollama and initializing caption generator...")
    try:
        generator = CaptionGenerator()
        print("✅ Connected to Ollama! Caption generator initialized!")
    except ConnectionError as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return

    # Example directory
    photos_dir = Path("path/to/photos")

    if not photos_dir.exists():
        print(f"⚠️  Photos directory not found at {photos_dir}")
        print("To test, replace 'path/to/photos' with an actual directory of images")
        return

    # Get all image files
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.heic'}
    image_files = []
    for ext in supported_extensions:
        image_files.extend(photos_dir.glob(f'*{ext}'))
        image_files.extend(photos_dir.glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"No images found in {photos_dir}")
        return

    print(f"\nFound {len(image_files)} image(s)")

    # Process each image
    all_results = {}
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing {image_path.name}...")

        try:
            captions = generator.generate_all(image_path)
            all_results[image_path.name] = captions
            print(f"  ✅ Generated captions for {image_path.name}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

    # Display results
    print("\n" + "=" * 70)
    print("BATCH PROCESSING RESULTS")
    print("=" * 70)
    for image_name, captions in all_results.items():
        print(f"\n📸 {image_name}")
        print(f"  Title: {captions.get('title', 'N/A')}")
        print(f"  Caption: {captions.get('caption', 'N/A')[:80]}...")

    # Save results to file
    import json
    results_file = photos_dir / "captions_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")


def example_travel_log_manager():
    """Use caption generation with TravelLogFaceManager."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Integration with TravelLogFaceManager")
    print("=" * 70)

    from travel_log.face_manager import TravelLogFaceManager

    # Create a face manager with caption generation enabled
    print("\nInitializing TravelLogFaceManager with caption generation...")
    manager = TravelLogFaceManager(
        workspace_dir="./travel_log_workspace",
        enable_caption_generator=True
    )
    print("✅ Face manager initialized with caption support!")

    # Example: Process a photo with captions
    image_path = Path("path/to/your/travel/photo.jpg")

    if not image_path.exists():
        print(f"⚠️  Example image not found at {image_path}")
        print("To test, replace 'path/to/your/travel/photo.jpg' with an actual image path")
        return

    print(f"\nProcessing photo: {image_path.name}")
    result = manager.process_photo(
        image_path,
        extract_faces=True,
        identify_faces=False,
        generate_embeddings=False,
        generate_captions=True
    )

    print("\n📋 Processing Results:")
    print(f"  Faces detected: {result['num_faces']}")
    if 'captions' in result:
        print("  Captions generated:")
        for key, value in result['captions'].items():
            print(f"    - {key}: {value[:60]}...")
    if 'caption_file' in result:
        print(f"  Captions saved to: {result['caption_file']}")


def example_custom_prompts():
    """Generate captions with custom prompts using Ollama API."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Caption Parameters")
    print("=" * 70)

    print("\nConnecting to Ollama and initializing caption generator...")
    try:
        generator = CaptionGenerator()
        print("✅ Connected to Ollama! Caption generator initialized!")
    except ConnectionError as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return

    # Example image
    image_path = Path("path/to/your/travel/photo.jpg")

    if not image_path.exists():
        print(f"⚠️  Example image not found at {image_path}")
        print("To test, replace 'path/to/your/travel/photo.jpg' with an actual image path")
        print("\nFor this example, demonstrating with temperature variations...")
        from PIL import Image
        # Create a sample image
        image = Image.new("RGB", (400, 300), color=(100, 150, 200))
        print("Using sample blue image for demonstration")
    else:
        from PIL import Image
        image = Image.open(image_path)

    print(f"\nGenerating captions with different temperature settings...")
    temperatures = [0.3, 0.7, 1.0]

    for temp in temperatures:
        print(f"\n🌡️  Temperature: {temp} (0.3=deterministic, 1.0=creative)")
        try:
            caption = generator.generate_caption(image, temperature=temp)
            print(f"   Caption: {caption}")
        except Exception as e:
            print(f"   Error: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CAPTION GENERATION WITH OLLAMA - EXAMPLES")
    print("=" * 70)

    print("\nThis script demonstrates how to use Ollama's LLaVA model for generating")
    print("captions and titles for travel photos with true image analysis.")

    print("\n💡 Prerequisites:")
    print("   1. Install Ollama from https://ollama.ai")
    print("   2. Pull the model: ollama pull llava:7b")
    print("   3. Start Ollama: ollama serve")
    print("   4. Then run this script in another terminal")

    # Run examples
    try:
        example_single_image()
    except Exception as e:
        print(f"⚠️  Example 1 error: {e}")

    try:
        example_batch_processing()
    except Exception as e:
        print(f"⚠️  Example 2 error: {e}")

    try:
        example_travel_log_manager()
    except Exception as e:
        print(f"⚠️  Example 3 error: {e}")

    try:
        example_custom_prompts()
    except Exception as e:
        print(f"⚠️  Example 4 error: {e}")

    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

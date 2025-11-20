#!/usr/bin/env python3
"""
Test the updated CaptionGenerator with 8-bit quantization.
"""

import sys
from pathlib import Path
from PIL import Image
from travel_log.caption_generator import CaptionGenerator

def test_caption_generator(image_path: str = None):
    """Test the caption generator with a sample image."""
    
    print("=" * 80)
    print("Testing CaptionGenerator with 8-bit Quantized LLaVA")
    print("=" * 80)
    
    # Initialize the generator
    print("\nInitializing CaptionGenerator...")
    generator = CaptionGenerator()
    
    # Get test image
    if image_path:
        print(f"\nLoading image: {image_path}")
        test_image = Image.open(image_path)
    else:
        # Use a built-in test image
        print("\nUsing built-in test image (cherry blossom)...")
        from vllm.assets.image import ImageAsset
        test_image = ImageAsset("cherry_blossom").pil_image
    
    print("\n" + "=" * 80)
    print("Test 1: Generate Caption")
    print("=" * 80)
    caption = generator.generate_caption(test_image, max_tokens=100, temperature=0.7)
    print(f"Caption: {caption}")
    
    print("\n" + "=" * 80)
    print("Test 2: Generate Title")
    print("=" * 80)
    title = generator.generate_title(test_image, max_tokens=15, temperature=0.7)
    print(f"Title: {title}")
    
    print("\n" + "=" * 80)
    print("Test 3: Generate Travel Caption")
    print("=" * 80)
    travel_caption = generator.generate_travel_caption(test_image, max_tokens=120, temperature=0.7)
    print(f"Travel Caption: {travel_caption}")
    
    print("\n" + "=" * 80)
    print("Test 4: Generate All")
    print("=" * 80)
    all_results = generator.generate_all(test_image)
    print(f"Caption: {all_results['caption']}")
    print(f"\nTitle: {all_results['title']}")
    print(f"\nTravel Caption: {all_results['travel_caption']}")
    
    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)
    
    # Cleanup
    generator.cleanup()
    print("\n✓ Cleanup completed")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_caption_generator(image_path)


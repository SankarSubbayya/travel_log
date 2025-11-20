#!/usr/bin/env python3
"""
Test LLaVA-1.5-7B with custom images from your filesystem
"""

import os
import sys
from pathlib import Path
from PIL import Image

os.environ['VLLM_TORCH_COMPILE_LEVEL'] = '0'

from vllm import LLM, SamplingParams

def load_image(image_path: str) -> Image.Image:
    """Load an image from the filesystem."""
    img = Image.open(image_path)
    # Convert to RGB if needed (for PNG with alpha channel, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def analyze_image(llm: LLM, image_path: str, question: str = None):
    """Analyze an image using LLaVA."""
    
    # Load the image
    image = load_image(image_path)
    
    # Default question if none provided
    if question is None:
        question = "Describe this image in detail."
    
    # Create the prompt in LLaVA format
    prompt = {
        "prompt": f"USER: <image>\n{question}\nASSISTANT:",
        "multi_modal_data": {"image": image},
    }
    
    # Generate response
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=300
    )
    
    outputs = llm.generate(prompt, sampling_params)
    return outputs[0].outputs[0].text

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_llava_custom_image.py <image_path> [question]")
        print("\nExample:")
        print("  python test_llava_custom_image.py photo.jpg")
        print("  python test_llava_custom_image.py photo.jpg 'What people are in this photo?'")
        sys.exit(1)
    
    image_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("Loading LLaVA-1.5-7B with 8-bit quantization...")
    print("This may take a few minutes on first run...")
    print("=" * 80)
    
    # Initialize the model with 8-bit quantization
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        dtype="half",
    )
    
    print("\n✓ Model loaded successfully!\n")
    print("=" * 80)
    print(f"Analyzing image: {image_path}")
    if question:
        print(f"Question: {question}")
    print("=" * 80)
    print()
    
    # Analyze the image
    response = analyze_image(llm, image_path, question)
    
    print("Response:")
    print(response)
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()


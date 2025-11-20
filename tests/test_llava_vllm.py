#!/usr/bin/env python3
"""
Test script for LLaVA-1.5-7B with vLLM using 8-bit quantization
"""

import os
os.environ['VLLM_TORCH_COMPILE_LEVEL'] = '0'  # Disable compilation if python3.12-dev not installed

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

def main():
    print("=" * 80)
    print("Loading LLaVA-1.5-7B with 8-bit quantization...")
    print("=" * 80)
    
    # Initialize the model with 8-bit quantization
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        quantization="bitsandbytes",  # 8-bit quantization
        load_format="bitsandbytes",   # Load in bitsandbytes format
        enforce_eager=True,           # Use eager mode (if python3.12-dev not installed)
        max_model_len=2048,           # Reduce context length to save memory
        gpu_memory_utilization=0.90,  # Use up to 90% of GPU memory
        dtype="half",                 # Use float16
    )
    
    print("\n" + "=" * 80)
    print("Model loaded successfully!")
    print("=" * 80)
    
    # Test 1: Text-only prompt
    print("\n[Test 1] Text-only generation:")
    text_prompt = "USER: What is the capital of France?\nASSISTANT:"
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100
    )
    
    outputs = llm.generate(text_prompt, sampling_params)
    print(f"Prompt: {text_prompt}")
    print(f"Response: {outputs[0].outputs[0].text}")
    
    # Test 2: Image understanding (using built-in test image)
    print("\n" + "=" * 80)
    print("[Test 2] Image understanding with cherry blossom:")
    print("=" * 80)
    
    # Load a sample image from vLLM's assets
    image = ImageAsset("cherry_blossom").pil_image
    
    # Create a multimodal prompt
    # LLaVA uses a special format for image+text prompts
    prompt = {
        "prompt": "USER: <image>\nWhat is shown in this image?\nASSISTANT:",
        "multi_modal_data": {"image": image},
    }
    
    outputs = llm.generate(prompt, sampling_params)
    print(f"Response: {outputs[0].outputs[0].text}")
    
    # Test 3: More detailed image analysis
    print("\n" + "=" * 80)
    print("[Test 3] Detailed image analysis:")
    print("=" * 80)
    
    prompt = {
        "prompt": "USER: <image>\nDescribe this image in detail. What colors do you see? What season might it be?\nASSISTANT:",
        "multi_modal_data": {"image": image},
    }
    
    sampling_params_detailed = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=200
    )
    
    outputs = llm.generate(prompt, sampling_params_detailed)
    print(f"Response: {outputs[0].outputs[0].text}")
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nIf you see memory errors, try:")
        print("  1. Reducing max_model_len (currently 2048)")
        print("  2. Reducing gpu_memory_utilization (currently 0.90)")
        print("  3. Installing python3.12-dev for better optimization")
        raise


#!/usr/bin/env python3
"""
Batch image analysis for Travel Log using LLaVA-1.5-7B with 8-bit quantization

This script can analyze multiple images from your travel events and generate:
- Detailed descriptions
- Scene understanding
- Face/person detection
- Location/activity identification
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
from PIL import Image

os.environ['VLLM_TORCH_COMPILE_LEVEL'] = '0'

from vllm import LLM, SamplingParams

class LLaVAImageAnalyzer:
    """Analyze images using LLaVA vision-language model."""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        """Initialize the LLaVA model with 8-bit quantization."""
        print("=" * 80)
        print(f"Loading {model_name} with 8-bit quantization...")
        print("This may take 1-2 minutes on first run...")
        print("=" * 80)
        
        self.llm = LLM(
            model=model_name,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            enforce_eager=True,
            max_model_len=2048,
            gpu_memory_utilization=0.90,
            dtype="half",
        )
        print("\n✓ Model loaded successfully!\n")
    
    def load_image(self, image_path: Path) -> Image.Image:
        """Load and prepare an image."""
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    
    def analyze_single_image(
        self, 
        image_path: Path,
        question: str = "Describe this image in detail, including people, location, activities, and any notable features.",
        temperature: float = 0.7,
        max_tokens: int = 300
    ) -> Dict[str, Any]:
        """Analyze a single image and return structured results."""
        
        try:
            # Load the image
            image = self.load_image(image_path)
            
            # Create the prompt
            prompt = {
                "prompt": f"USER: <image>\n{question}\nASSISTANT:",
                "multi_modal_data": {"image": image},
            }
            
            # Generate response
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens
            )
            
            outputs = self.llm.generate(prompt, sampling_params)
            description = outputs[0].outputs[0].text.strip()
            
            return {
                "status": "success",
                "image_path": str(image_path),
                "description": description,
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "image_path": str(image_path),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_batch(
        self,
        image_paths: List[Path],
        questions: List[str] = None,
        output_file: str = None
    ) -> List[Dict[str, Any]]:
        """Analyze a batch of images."""
        
        results = []
        
        # Use default question if none provided
        if questions is None:
            questions = [
                "Describe this image in detail, including people, location, activities, and any notable features."
            ] * len(image_paths)
        elif len(questions) == 1:
            questions = questions * len(image_paths)
        
        print(f"\nAnalyzing {len(image_paths)} images...")
        print("=" * 80)
        
        for idx, (img_path, question) in enumerate(zip(image_paths, questions), 1):
            print(f"\n[{idx}/{len(image_paths)}] Processing: {img_path.name}")
            
            result = self.analyze_single_image(img_path, question)
            results.append(result)
            
            if result["status"] == "success":
                print(f"✓ Description: {result['description'][:100]}...")
            else:
                print(f"✗ Error: {result['error']}")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {output_file}")
        
        print("\n" + "=" * 80)
        print(f"Completed: {len([r for r in results if r['status'] == 'success'])}/{len(results)} successful")
        print("=" * 80)
        
        return results
    
    def ask_custom_questions(
        self,
        image_path: Path,
        questions: List[str]
    ) -> List[Dict[str, Any]]:
        """Ask multiple questions about a single image."""
        
        results = []
        print(f"\nAnalyzing: {image_path.name}")
        print("=" * 80)
        
        for idx, question in enumerate(questions, 1):
            print(f"\n[Question {idx}] {question}")
            result = self.analyze_single_image(image_path, question)
            results.append(result)
            
            if result["status"] == "success":
                print(f"Answer: {result['description']}")
            else:
                print(f"Error: {result['error']}")
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze travel photos using LLaVA vision AI"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image file(s) or directory to analyze"
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Custom question to ask about the image(s)",
        default="Describe this image in detail, including people, location, activities, and any notable features."
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    # Collect image paths
    image_paths = []
    for path_str in args.images:
        path = Path(path_str)
        if path.is_dir():
            # Add all images from directory
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_paths.extend(path.glob(ext))
        elif path.exists():
            image_paths.append(path)
        else:
            print(f"Warning: {path_str} not found, skipping...")
    
    if not image_paths:
        print("Error: No valid images found!")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} image(s) to analyze")
    
    # Initialize analyzer
    analyzer = LLaVAImageAnalyzer()
    
    # Analyze images
    results = analyzer.analyze_batch(
        image_paths,
        questions=[args.question],
        output_file=args.output
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for result in results:
        if result["status"] == "success":
            print(f"\n{Path(result['image_path']).name}:")
            print(f"  {result['description']}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Simple demo: Detect faces in a photo

Usage:
    python demo_face_detection.py path/to/photo.jpg
    
This will:
1. Detect all faces in the photo
2. Extract and save them
3. Show you the results
"""

import os
import sys
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

from travel_log import FaceDetector

def main():
    print("\n" + "="*70)
    print("FACE DETECTION DEMO")
    print("="*70)
    print("\n⚠️  Ignoring TensorFlow mutex warnings (they're harmless!)\n")
    
    # Check command line argument
    if len(sys.argv) < 2:
        print("Usage: python demo_face_detection.py <image_path>")
        print("\nExample:")
        print("  python demo_face_detection.py photos/group.jpg")
        print("\nNo image provided. Creating a test workspace instead...")
        
        # Create test workspace
        from travel_log import TravelLogFaceManager
        manager = TravelLogFaceManager("demo_workspace")
        print(f"\n✅ Created demo workspace at: demo_workspace/")
        print("\nTo test with your own photo:")
        print("  python demo_face_detection.py path/to/your/photo.jpg")
        return
    
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        print("\nMake sure the file exists and the path is correct.")
        return
    
    print(f"📷 Processing image: {image_path}")
    print(f"   File size: {image_path.stat().st_size / 1024:.1f} KB")
    
    # Create face detector
    print("\n🔍 Initializing face detector...")
    detector = FaceDetector(detector_backend='opencv')  # Fast for testing
    
    # Detect faces
    print(f"🔍 Detecting faces...")
    faces = detector.extract_faces(str(image_path))
    
    print(f"\n✅ Found {len(faces)} face(s)!")
    
    if len(faces) == 0:
        print("\nNo faces detected. This could mean:")
        print("  - The image doesn't contain visible faces")
        print("  - Faces are too small or unclear")
        print("  - Try a different detector: detector_backend='mtcnn'")
        return
    
    # Save extracted faces
    output_dir = Path("demo_workspace/faces")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving faces to: {output_dir}/")
    
    saved_paths = detector.save_extracted_faces(
        str(image_path),
        output_dir,
        prefix=image_path.stem
    )
    
    print(f"\n✅ Saved {len(saved_paths)} faces:")
    for i, path in enumerate(saved_paths, 1):
        print(f"   {i}. {path.name}")
    
    # Create annotated image
    annotated_path = output_dir.parent / f"{image_path.stem}_annotated.jpg"
    detector.annotate_image(
        str(image_path),
        str(annotated_path),
        color=(0, 255, 0),
        thickness=3
    )
    
    print(f"\n📸 Annotated image saved: {annotated_path}")
    
    print("\n" + "="*70)
    print("✅ SUCCESS!")
    print("="*70)
    print(f"\nResults:")
    print(f"  - Extracted faces: {output_dir}/")
    print(f"  - Annotated image: {annotated_path}")
    
    print("\n💡 Next steps:")
    print("  1. Check the extracted faces in demo_workspace/faces/")
    print("  2. Try face recognition: examples/face_labeling_example.py")
    print("  3. Generate embeddings: examples/face_embeddings_example.py")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()


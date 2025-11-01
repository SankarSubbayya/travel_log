#!/usr/bin/env python3
"""
Example: Basic Face Detection and Extraction

This example demonstrates how to:
1. Extract faces from a group photo
2. Save the extracted faces
3. Annotate the original image with face bounding boxes

Configuration:
    The script reads default paths from config.yaml:
    - images.default_test_image: Default image to process
    - images.output.extracted_faces: Where to save extracted faces
    - face_detection.default_backend: Which detection backend to use

Usage:
    python face_detection_example.py                    # Uses default from config.yaml
    python face_detection_example.py /path/to/photo.jpg # Uses command-line argument
"""

import sys
from pathlib import Path
from travel_log import FaceDetector, config, ensure_compatible_image, is_heic_file

def main():
    # Get configuration values
    default_backend = config.get('face_detection', {}).get('default_backend', 'mtcnn')
    default_image = config.get('images', {}).get('default_test_image', 'your_photo.jpg')
    output_dir = config.get('images', {}).get('output', {}).get('extracted_faces', 'extracted_faces')
    
    # Initialize the face detector using config
    # You can choose different backends: 'opencv', 'ssd', 'mtcnn', 'retinaface', 'dlib'
    detector = FaceDetector(detector_backend=default_backend)
    
    # Get image path from command line or use default from config
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    else:
        input_image = default_image
        print(f"ℹ️  Using default image from config: {input_image}")
    
    # Check if image exists
    if not Path(input_image).exists():
        print(f"❌ Error: Image file not found: {input_image}")
        print("\n💡 Usage:")
        print("   1. Update 'images.default_test_image' in config.yaml")
        print("   2. Or run: python face_detection_example.py /path/to/your/photo.jpg")
        print(f"\n   Example: python face_detection_example.py ~/Pictures/vacation.jpg")
        return
    
    # Handle HEIC conversion if needed
    converted = False
    if is_heic_file(input_image):
        print(f"📱 HEIC file detected, converting to JPEG...")
        try:
            input_image = ensure_compatible_image(input_image)
            converted = True
            print(f"✅ Converted to: {input_image}")
        except Exception as e:
            print(f"❌ Error converting HEIC: {e}")
            return
    
    # Extract and save faces
    print(f"Extracting faces from {input_image}...")
    saved_faces = detector.save_extracted_faces(
        image_path=input_image,
        output_dir=output_dir,
        prefix="face"
    )
    
    print(f"Extracted {len(saved_faces)} faces:")
    for face_path in saved_faces:
        print(f"  - {face_path}")
    
    # Create annotated image with bounding boxes
    print("\nCreating annotated image...")
    detector.annotate_image(
        image_path=input_image,
        output_path="annotated_photo.jpg",
        color=(0, 255, 0),  # Green boxes
        thickness=3
    )
    print("Saved annotated image to: annotated_photo.jpg")
    
    # Get face locations (if you just need coordinates)
    face_locations = detector.get_face_locations(input_image)
    print(f"\nFace locations:")
    for i, location in enumerate(face_locations):
        print(f"  Face {i+1}: x={location['x']}, y={location['y']}, "
              f"width={location['w']}, height={location['h']}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Example: Basic Face Detection and Extraction

This example demonstrates how to:
1. Extract faces from a group photo
2. Save the extracted faces
3. Annotate the original image with face bounding boxes
"""

from pathlib import Path
from travel_log import FaceDetector

def main():
    # Initialize the face detector
    # You can choose different backends: 'opencv', 'ssd', 'mtcnn', 'retinaface', 'dlib'
    detector = FaceDetector(detector_backend='mtcnn')
    
    # Path to your group photo
    input_image = "path/to/your/group_photo.jpg"
    
    # Directory to save extracted faces
    output_dir = "extracted_faces"
    
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


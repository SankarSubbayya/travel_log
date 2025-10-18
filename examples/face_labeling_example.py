#!/usr/bin/env python3
"""
Example: Face Recognition and Labeling

This example demonstrates how to:
1. Create a database of known people
2. Identify faces from extracted images
3. Label faces automatically
"""

from pathlib import Path
from travel_log import FaceLabeler

def main():
    # Initialize the face labeler with a database directory
    database_path = "face_database"
    labeler = FaceLabeler(
        database_path=database_path,
        model_name='Facenet512',  # High accuracy model
        detector_backend='mtcnn'
    )
    
    # Step 1: Add people to the database
    print("Step 1: Adding people to database...")
    
    # Add first person (provide multiple sample images for better accuracy)
    person1_samples = [
        "samples/alice/photo1.jpg",
        "samples/alice/photo2.jpg",
        "samples/alice/photo3.jpg"
    ]
    labeler.add_person("Alice", person1_samples)
    
    # Add second person
    person2_samples = [
        "samples/bob/photo1.jpg",
        "samples/bob/photo2.jpg"
    ]
    labeler.add_person("Bob", person2_samples)
    
    print(f"Database now contains: {labeler.list_known_people()}")
    
    # Step 2: Identify faces from extracted images
    print("\nStep 2: Identifying faces...")
    
    unknown_face = "extracted_faces/face_001.jpg"
    result = labeler.identify_face(unknown_face)
    
    if result:
        print(f"Face identified as: {result['name']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Distance: {result['distance']:.4f}")
    else:
        print("Face not recognized")
    
    # Step 3: Batch identification
    print("\nStep 3: Batch identification...")
    
    face_images = [
        "extracted_faces/face_001.jpg",
        "extracted_faces/face_002.jpg",
        "extracted_faces/face_003.jpg"
    ]
    
    results = labeler.identify_faces_batch(face_images)
    
    for img_path, result in zip(face_images, results):
        if result:
            print(f"{Path(img_path).name}: {result['name']} "
                  f"(confidence: {result['confidence']:.2%})")
        else:
            print(f"{Path(img_path).name}: Unknown")
    
    # Step 4: Create label mapping CSV
    print("\nStep 4: Creating label mapping...")
    
    df = labeler.create_label_mapping(
        face_images=face_images,
        output_path="face_labels.csv"
    )
    
    print(f"Saved label mapping to face_labels.csv")
    print(df)
    
    # Step 5: Verify if two faces are the same person
    print("\nStep 5: Face verification...")
    
    verification = labeler.verify_faces(
        "extracted_faces/face_001.jpg",
        "extracted_faces/face_002.jpg"
    )
    
    if verification['verified']:
        print("✓ Same person!")
    else:
        print("✗ Different people")
    print(f"Distance: {verification['distance']:.4f}")
    
    # Get database statistics
    stats = labeler.get_database_stats()
    print(f"\nDatabase Statistics:")
    print(f"Total people: {stats['total_people']}")
    print(f"Total images: {stats['total_images']}")
    for person, count in stats['people'].items():
        print(f"  {person}: {count} images")


if __name__ == "__main__":
    main()


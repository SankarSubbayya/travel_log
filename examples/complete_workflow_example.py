#!/usr/bin/env python3
"""
Example: Complete Travel Log Face Processing Workflow

This example demonstrates the complete workflow:
1. Process group photos from a trip
2. Extract all faces
3. Identify people automatically
4. Generate embeddings
5. Cluster similar faces
6. Export organized results
"""

from pathlib import Path
from travel_log import TravelLogFaceManager

def main():
    # Step 1: Initialize the face manager
    print("="*60)
    print("Travel Log Face Processing - Complete Workflow")
    print("="*60)
    
    workspace = "travel_log_workspace"
    manager = TravelLogFaceManager(
        workspace_dir=workspace,
        detector_backend='mtcnn',
        recognition_model='Facenet512'
    )
    
    print(f"\nWorkspace initialized at: {workspace}")
    
    # Step 2: Add known people to the database
    print("\n" + "="*60)
    print("Step 1: Building Face Database")
    print("="*60)
    
    # Add your travel companions to the database
    # You need sample photos of each person's face
    people_to_add = [
        ("Alice", ["samples/alice/face1.jpg", "samples/alice/face2.jpg"]),
        ("Bob", ["samples/bob/face1.jpg", "samples/bob/face2.jpg"]),
        ("Charlie", ["samples/charlie/face1.jpg"])
    ]
    
    for person_name, sample_images in people_to_add:
        result = manager.add_person_to_database(person_name, sample_images)
        print(f"Added {result['person_name']}: "
              f"{result['num_samples']} samples, "
              f"{result['embeddings_generated']} embeddings")
    
    # Step 3: Process a single group photo
    print("\n" + "="*60)
    print("Step 2: Processing Single Photo")
    print("="*60)
    
    photo_path = "photos/beach_group.jpg"
    result = manager.process_photo(
        image_path=photo_path,
        extract_faces=True,
        identify_faces=True,
        generate_embeddings=True
    )
    
    print(f"\nProcessed: {result['source_image']}")
    print(f"Found {result['num_faces']} faces:")
    
    for face in result['faces']:
        print(f"  Face {face['face_index']}:")
        print(f"    Identified as: {face.get('identified_as', 'N/A')}")
        if 'confidence' in face:
            print(f"    Confidence: {face['confidence']:.2%}")
        print(f"    Has embedding: {face.get('has_embedding', False)}")
    
    # Step 4: Process all photos in a directory
    print("\n" + "="*60)
    print("Step 3: Processing Photo Directory")
    print("="*60)
    
    photos_dir = "photos/trip_photos"
    batch_results = manager.process_directory(
        photos_dir=photos_dir,
        pattern="*.jpg"
    )
    
    print(f"\nProcessed {len(batch_results)} photos")
    
    total_faces = sum(r.get('num_faces', 0) for r in batch_results)
    print(f"Total faces detected: {total_faces}")
    
    # Step 5: Cluster similar faces
    print("\n" + "="*60)
    print("Step 4: Clustering Similar Faces")
    print("="*60)
    
    clusters = manager.get_face_clusters(threshold=0.6)
    
    print(f"\nFound {len(clusters)} face clusters:")
    for cluster_id, members in clusters.items():
        print(f"  {cluster_id}: {len(members)} faces")
        # Show first 3 members
        for member in members[:3]:
            print(f"    - {member}")
        if len(members) > 3:
            print(f"    ... and {len(members)-3} more")
    
    # Step 6: Generate summary report
    print("\n" + "="*60)
    print("Step 5: Generating Summary Report")
    print("="*60)
    
    summary = manager.generate_summary_report()
    
    print(f"\n📊 Summary Statistics:")
    print(f"  Workspace: {summary['workspace']}")
    print(f"  Extracted faces: {summary['extracted_faces']}")
    print(f"  Generated embeddings: {summary['generated_embeddings']}")
    print(f"  Identified faces: {summary['identified_faces']}")
    print(f"  Unknown faces: {summary['unknown_faces']}")
    print(f"\n  Database:")
    print(f"    Total people: {summary['database']['total_people']}")
    print(f"    Total reference images: {summary['database']['total_images']}")
    
    # Step 7: Export organized dataset
    print("\n" + "="*60)
    print("Step 6: Exporting Labeled Dataset")
    print("="*60)
    
    export_dir = "organized_faces"
    export_stats = manager.export_labeled_dataset(export_dir)
    
    print(f"\nExported {export_stats['total']} faces to {export_dir}")
    print("Breakdown by label:")
    for label, count in export_stats['by_label'].items():
        print(f"  {label}: {count} faces")
    
    # Final summary
    print("\n" + "="*60)
    print("✅ Workflow Complete!")
    print("="*60)
    print(f"\nResults saved in:")
    print(f"  - Extracted faces: {workspace}/extracted_faces/")
    print(f"  - Embeddings: {workspace}/embeddings/")
    print(f"  - Reports: {workspace}/results/")
    print(f"  - Face database: {workspace}/face_database/")
    print(f"  - Organized export: {export_dir}/")


def quick_process_example():
    """
    Quick example for processing photos with minimal setup
    """
    print("\n" + "="*60)
    print("Quick Processing Example")
    print("="*60)
    
    # Create a manager
    manager = TravelLogFaceManager("quick_workspace")
    
    # Process a single photo
    result = manager.process_photo("my_photo.jpg")
    
    print(f"Detected {result['num_faces']} faces")
    
    # That's it! All faces are extracted, and embeddings generated


if __name__ == "__main__":
    # Run the complete workflow
    main()
    
    # Uncomment to run quick example
    # quick_process_example()


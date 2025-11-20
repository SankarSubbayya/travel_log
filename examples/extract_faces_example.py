#!/usr/bin/env python3
"""
Example: Extract and Save Faces to Disk with Metadata

Demonstrates extracting individual face images and saving them with metadata.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_log.face_detector import detect_faces_from_file
from travel_log.face_manager import FaceManager
from travel_log.image_metadata import extract_metadata
from travel_log.face_extractor import FaceExtractor, extract_and_save_faces


def main():
    print("=" * 70)
    print("Face Extraction Example")
    print("=" * 70)

    # Get photo path
    if len(sys.argv) < 2:
        print("\n⚠️  Usage: python extract_faces_example.py <photo_path>")
        print("   Using sample photo from face_database...")

        sample_dirs = Path("face_database")
        if sample_dirs.exists():
            sample_photos = list(sample_dirs.rglob("*.jpg")) + list(sample_dirs.rglob("*.jpeg"))
            if sample_photos:
                photo_path = sample_photos[0]
                print(f"   Using: {photo_path}")
            else:
                print("   ❌ No sample photos found!")
                return
        else:
            print("   ❌ face_database directory not found!")
            return
    else:
        photo_path = Path(sys.argv[1])

    if not photo_path.exists():
        print(f"❌ Photo not found: {photo_path}")
        return

    print(f"\n📸 Processing photo: {photo_path.name}")

    # 1. Extract EXIF metadata
    print("\n1️⃣  Extracting EXIF metadata...")
    metadata = extract_metadata(str(photo_path))
    print(f"   ✓ Image size: {metadata.get('width')}×{metadata.get('height')}")
    if 'datetime_str' in metadata:
        print(f"   ✓ Date: {metadata['datetime_str']}")
    if 'latitude' in metadata:
        print(f"   ✓ GPS: {metadata['latitude']:.6f}, {metadata['longitude']:.6f}")

    # 2. Detect faces
    print("\n2️⃣  Detecting faces with DeepFace...")
    detected_faces = detect_faces_from_file(
        str(photo_path),
        detector_backend="retinaface",
        align=True
    )
    print(f"   ✓ Found {len(detected_faces)} face(s)")

    for i, face in enumerate(detected_faces):
        conf = face.get('confidence', 1.0)
        has_embedding = 'embedding' in face
        print(f"   Face {i}: confidence={conf:.2%}, has_embedding={has_embedding}")

    # 3. Identify faces
    print("\n3️⃣  Identifying faces...")
    face_manager = FaceManager(
        database_path="face_database",
        model_name="Facenet512",
        detector_backend="retinaface"
    )

    print("   Loading face database...")
    face_manager.initialize_database()
    print(f"   ✓ Loaded {len(face_manager.identities)} identities")

    face_identifications = {}
    for i, face in enumerate(detected_faces):
        if 'embedding' in face:
            result = face_manager.identify_face_embedding(
                face['embedding'],
                threshold=0.6
            )
            face_identifications[i] = result
            match_name = result.get('match', 'Unknown')
            conf = result.get('confidence', 0.0)
            print(f"   Face {i}: {match_name} (confidence: {conf:.2%})")

    # 4. Extract and save faces (by photo)
    print("\n4️⃣  Extracting and saving faces (organized by photo)...")
    saved_faces = extract_and_save_faces(
        photo_path=photo_path,
        detected_faces=detected_faces,
        face_identifications=face_identifications,
        exif_metadata=metadata,
        output_dir="extracted_faces",
        organize_by_person=False
    )

    print(f"   ✓ Saved {len(saved_faces)} faces")
    for face_info in saved_faces:
        print(f"   • {face_info['filename']}: {face_info['person_name']}")
        print(f"     Path: {face_info['saved_path']}")
        if 'metadata_path' in face_info:
            print(f"     Metadata: {face_info['metadata_path']}")
        if 'embedding_path' in face_info:
            print(f"     Embedding: {face_info['embedding_path']}")

    # 5. Extract and save faces (by person)
    print("\n5️⃣  Extracting and saving faces (organized by person)...")
    faces_by_person = extract_and_save_faces(
        photo_path=photo_path,
        detected_faces=detected_faces,
        face_identifications=face_identifications,
        exif_metadata=metadata,
        output_dir="extracted_faces_by_person",
        organize_by_person=True
    )

    print(f"   ✓ Organized faces for {len(faces_by_person)} people")
    for person, person_faces in faces_by_person.items():
        print(f"   • {person}: {len(person_faces)} face(s)")
        for face_info in person_faces[:2]:  # Show first 2
            print(f"     - {face_info['filename']}")

    # 6. Get statistics
    print("\n6️⃣  Extraction statistics...")
    extractor = FaceExtractor(output_dir="extracted_faces")
    stats = extractor.get_extraction_stats()

    print(f"   Output directory: {stats['output_directory']}")
    print(f"   Total faces: {stats.get('total_faces', 0)}")
    print(f"   Total people: {stats.get('total_people', 0)}")

    if stats.get('faces_by_person'):
        print("\n   Faces by person:")
        for person, count in stats['faces_by_person'].items():
            print(f"   • {person}: {count} faces")

    print("\n" + "=" * 70)
    print("✅ Face extraction completed!")
    print("=" * 70)
    print(f"\n📁 Extracted faces saved to:")
    print(f"   • extracted_faces/ (organized by photo)")
    print(f"   • extracted_faces_by_person/ (organized by person)")
    print("\n💡 Each face includes:")
    print("   • .jpg - Face image")
    print("   • .json - Metadata (person, bbox, confidence, EXIF context)")
    print("   • .npy - Face embedding (512D Facenet512)")


if __name__ == "__main__":
    main()

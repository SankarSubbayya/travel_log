#!/usr/bin/env python3
"""
Example: Individual Face Storage in Qdrant

Demonstrates storing and searching individual detected faces with their embeddings.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_log.qdrant_store import create_qdrant_store
from travel_log.face_detector import detect_faces_from_file
from travel_log.face_manager import FaceManager
from travel_log.image_metadata import extract_metadata


def main():
    print("=" * 70)
    print("Individual Face Storage Example")
    print("=" * 70)

    # 1. Initialize Qdrant
    print("\n1️⃣  Connecting to Qdrant...")
    store = create_qdrant_store()
    stats = store.get_statistics()
    print(f"   ✓ Connected!")
    print(f"   ✓ Photos: {stats['total_photos']}")
    print(f"   ✓ Faces: {stats['total_faces']}")

    # 2. Get photo path
    if len(sys.argv) < 2:
        print("\n⚠️  Usage: python face_storage_example.py <photo_path>")
        print("   Using sample photo from face_database...")

        # Try to find a sample photo
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

    print(f"\n2️⃣  Processing photo: {photo_path.name}")

    # 3. Extract EXIF metadata
    print("\n3️⃣  Extracting EXIF metadata...")
    metadata = extract_metadata(str(photo_path))
    print(f"   ✓ Image size: {metadata.get('width')}×{metadata.get('height')}")
    if 'datetime_str' in metadata:
        print(f"   ✓ Date: {metadata['datetime_str']}")
    if 'latitude' in metadata:
        print(f"   ✓ GPS: {metadata['latitude']:.6f}, {metadata['longitude']:.6f}")

    # 4. Detect faces
    print("\n4️⃣  Detecting faces with DeepFace...")
    detected_faces = detect_faces_from_file(
        str(photo_path),
        detector_backend="retinaface",
        align=True
    )
    print(f"   ✓ Found {len(detected_faces)} face(s)")

    # Display face info
    for i, face in enumerate(detected_faces):
        conf = face.get('confidence', 1.0)
        has_embedding = 'embedding' in face
        embedding_dim = len(face['embedding']) if has_embedding else 0
        print(f"   Face {i}: confidence={conf:.2%}, embedding={embedding_dim}D")

    # 5. Initialize face identification
    print("\n5️⃣  Identifying faces...")
    face_manager = FaceManager(
        database_path="face_database",
        model_name="Facenet512",
        detector_backend="retinaface"
    )

    # Initialize database
    print("   Loading face database...")
    face_manager.initialize_database()
    print(f"   ✓ Loaded {len(face_manager.identities)} identities")

    # Identify faces
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

    # 6. Store photo in Qdrant
    print("\n6️⃣  Storing photo in Qdrant...")
    photo_id = store.store_photo(
        photo_path=photo_path,
        face_embedding=None,
        exif_metadata=metadata,
        detected_faces=detected_faces,
        face_identifications=face_identifications,
        captions=None
    )
    print(f"   ✓ Photo stored! ID: {photo_id}")

    # 7. Store individual faces
    print("\n7️⃣  Storing individual faces in Qdrant...")
    face_ids = store.store_individual_faces(
        photo_id=photo_id,
        photo_path=photo_path,
        detected_faces=detected_faces,
        face_identifications=face_identifications,
        exif_metadata=metadata
    )
    print(f"   ✓ Stored {len(face_ids)} faces!")

    for i, face_id in enumerate(face_ids):
        person = face_identifications.get(i, {}).get('match', 'Unknown')
        print(f"   Face {i} ({person}): {face_id}")

    # 8. Query stored faces
    print("\n8️⃣  Querying stored faces...")

    # Get all faces
    all_faces = store.get_all_faces(limit=100)
    print(f"   ✓ Total faces in database: {len(all_faces)}")

    # Group by person
    from collections import defaultdict
    faces_by_person = defaultdict(list)
    for face in all_faces:
        person = face.get('person_name', 'Unknown')
        faces_by_person[person].append(face)

    print(f"\n   Face counts by person:")
    for person, person_faces in sorted(faces_by_person.items()):
        print(f"   • {person}: {len(person_faces)} faces")

    # 9. Search faces by person
    if face_identifications:
        # Get first identified person
        identified_people = [
            ident.get('match')
            for ident in face_identifications.values()
            if ident.get('match') not in ['Unknown', 'Error']
        ]

        if identified_people:
            search_person = identified_people[0]
            print(f"\n9️⃣  Searching for faces of '{search_person}'...")

            person_faces = store.search_faces_by_person(search_person, limit=20)
            print(f"   ✓ Found {len(person_faces)} faces of {search_person}")

            for face in person_faces[:5]:  # Show first 5
                print(f"   • {face['filename']} - Face #{face['face_index']}")
                if face.get('datetime'):
                    print(f"     Date: {face['datetime']}")
                if face.get('confidence'):
                    print(f"     Confidence: {face['confidence']:.2%}")

    # 10. Face similarity search
    if detected_faces and 'embedding' in detected_faces[0]:
        print(f"\n🔟 Searching for similar faces...")
        query_embedding = detected_faces[0]['embedding']

        similar_faces = store.search_faces_similar_to(
            query_embedding=query_embedding,
            limit=5,
            score_threshold=0.5
        )

        print(f"   ✓ Found {len(similar_faces)} similar faces")
        for face in similar_faces:
            print(f"   • {face['filename']} - Face #{face['face_index']}")
            print(f"     Person: {face['person_name']}")
            print(f"     Similarity: {face['score']:.2%}")

    # 11. Final statistics
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    final_stats = store.get_statistics()
    print(f"Total Photos: {final_stats['total_photos']}")
    print(f"Total Faces: {final_stats['total_faces']}")
    print(f"Collections: {final_stats['collection_name']}, {final_stats['faces_collection_name']}")
    print("\n✅ Face storage example completed!")


if __name__ == "__main__":
    main()

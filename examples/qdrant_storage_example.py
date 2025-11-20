#!/usr/bin/env python3
"""
Example: Storing Travel Photos in Qdrant

This script demonstrates how to store photos with all their metadata
(EXIF, faces, identifications, captions) in Qdrant for semantic search.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from travel_log import FaceDetector, FaceLabeler, get_complete_metadata
from travel_log.caption_generator import CaptionGenerator
from travel_log.qdrant_store import create_qdrant_store
from travel_log import config
from PIL import Image
import numpy as np


def store_photo_with_all_data(image_path: str):
    """
    Complete workflow: Detect, identify, caption, and store in Qdrant.

    Args:
        image_path: Path to photo
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(image_path).name}")
    print(f"{'='*60}\n")

    # Initialize components
    print("1️⃣  Initializing components...")
    detector = FaceDetector(detector_backend='mtcnn')
    labeler = FaceLabeler(
        database_path='./face_database',
        model_name='Facenet512'
    )
    try:
        caption_gen = CaptionGenerator()
        has_caption_gen = True
    except:
        print("   ⚠️  Caption generator not available (Ollama not running)")
        has_caption_gen = False

    qdrant_store = create_qdrant_store()

    # Extract EXIF metadata
    print("\n2️⃣  Extracting EXIF metadata...")
    metadata = get_complete_metadata(image_path)
    print(f"   ✓ Found metadata:")
    if 'datetime_str' in metadata:
        print(f"     - Date: {metadata['datetime_str']}")
    if 'latitude' in metadata and 'longitude' in metadata:
        print(f"     - GPS: {metadata['latitude']:.4f}, {metadata['longitude']:.4f}")
    if 'camera' in metadata and metadata['camera']:
        camera = metadata['camera']
        if 'camera_make' in camera:
            print(f"     - Camera: {camera.get('camera_make')} {camera.get('camera_model', '')}")

    # Detect faces
    print("\n3️⃣  Detecting faces...")
    faces = detector.extract_faces(image_path)
    print(f"   ✓ Detected {len(faces)} face(s)")

    # Get face embeddings for storage
    face_embedding = None
    if faces:
        try:
            # Get embeddings for all faces
            from deepface import DeepFace
            embeddings = []
            for face in faces:
                result = DeepFace.represent(
                    img_path=image_path,
                    model_name='Facenet512',
                    enforce_detection=False
                )
                if result:
                    embeddings.append(result[0]['embedding'])

            if embeddings:
                # Use average embedding for the photo
                face_embedding = np.mean(embeddings, axis=0)
                print(f"   ✓ Generated face embeddings (512D)")
        except Exception as e:
            print(f"   ⚠️  Could not generate embeddings: {e}")

    # Identify faces
    print("\n4️⃣  Identifying faces...")
    face_identifications = {}
    if faces:
        try:
            matches = labeler.find_face(image_path)
            if matches and len(matches) > 0 and not matches[0].empty:
                for idx, match in matches[0].iterrows():
                    person_name = Path(match['identity']).parent.name
                    distance = float(match['distance'])
                    confidence = max(0, 1 - (distance / 2.0))

                    face_identifications[idx] = {
                        'match': person_name,
                        'confidence': confidence,
                        'distance': distance
                    }
                    print(f"   ✓ Face {idx}: {person_name} ({confidence:.1%})")
            else:
                print(f"   ⚠️  No matches found in database")
        except Exception as e:
            print(f"   ⚠️  Identification error: {e}")

    # Generate captions
    print("\n5️⃣  Generating captions...")
    captions = None
    if has_caption_gen:
        try:
            image = Image.open(image_path)
            captions = caption_gen.generate_all(image)
            print(f"   ✓ Generated captions:")
            if 'title' in captions:
                print(f"     - Title: {captions['title']}")
            if 'caption' in captions:
                print(f"     - Caption: {captions['caption'][:100]}...")
        except Exception as e:
            print(f"   ⚠️  Caption generation error: {e}")
    else:
        print(f"   ⏭️  Skipped (Ollama not running)")

    # Store in Qdrant
    print("\n6️⃣  Storing in Qdrant...")
    try:
        point_id = qdrant_store.store_photo(
            photo_path=image_path,
            face_embedding=face_embedding,
            exif_metadata=metadata,
            detected_faces=faces,
            face_identifications=face_identifications,
            captions=captions
        )
        print(f"   ✓ Stored as: {point_id}")

        # Verify storage
        stored_photo = qdrant_store.get_photo(point_id)
        if stored_photo:
            print(f"   ✓ Verified storage")
            payload = stored_photo['metadata']
            print(f"\n📦 Stored Data Summary:")
            print(f"   - Filename: {payload.get('filename')}")
            print(f"   - EXIF fields: {len(payload.get('exif', {}))}")
            print(f"   - Detected faces: {payload.get('num_faces', 0)}")
            print(f"   - Identified people: {len(payload.get('people_names', []))}")
            print(f"   - Has captions: {bool(payload.get('captions'))}")
            print(f"   - Has GPS: {bool(payload.get('latitude') and payload.get('longitude'))}")

    except Exception as e:
        print(f"   ❌ Storage error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("✅ Complete!")
    print(f"{'='*60}\n")


def demo_search():
    """Demonstrate search functionality."""
    print("\n" + "="*60)
    print("DEMO: Searching Stored Photos")
    print("="*60 + "\n")

    qdrant_store = create_qdrant_store()

    # Get statistics
    stats = qdrant_store.get_statistics()
    print(f"📊 Database Statistics:")
    print(f"   Total photos: {stats.get('total_photos', 0)}")
    print(f"   Collection: {stats.get('collection_name')}")
    print(f"   Embedding dimension: {stats.get('embedding_dimension')}")

    if stats.get('total_photos', 0) == 0:
        print("\n⚠️  No photos in database yet. Store some photos first!")
        return

    # Search by person
    print(f"\n🔍 Search by Person:")
    photos_with_person = qdrant_store.search_by_person("sankar", limit=5)
    print(f"   Found {len(photos_with_person)} photos with 'sankar'")
    for photo in photos_with_person[:3]:
        print(f"   - {photo['filename']}: {photo.get('people', [])}")

    # Get all photos
    print(f"\n📸 All Photos:")
    all_photos = qdrant_store.get_all_photos(limit=10)
    for photo in all_photos[:5]:
        people_str = ", ".join(photo.get('people', [])) if photo.get('people') else "No IDs"
        print(f"   - {photo['filename']}: {people_str} ({photo.get('num_faces', 0)} faces)")


def main():
    """Run complete example."""
    import argparse

    parser = argparse.ArgumentParser(description="Store photos in Qdrant")
    parser.add_argument("image_path", nargs="?", help="Path to image file")
    parser.add_argument("--search", action="store_true", help="Demo search functionality")

    args = parser.parse_args()

    if args.search:
        demo_search()
        return

    # Get image path
    image_path = args.image_path
    if not image_path:
        # Try to use default from config
        image_path = config.get('images', {}).get('default_test_image')

    if not image_path or not Path(image_path).exists():
        print("❌ No valid image path provided")
        print("\nUsage:")
        print("  python examples/qdrant_storage_example.py <image_path>")
        print("  python examples/qdrant_storage_example.py --search")
        sys.exit(1)

    # Process and store photo
    store_photo_with_all_data(image_path)

    # Show search demo
    print("\n" + "─"*60)
    demo_search()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
View Qdrant Database Contents

Quick script to inspect what's stored in Qdrant.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from travel_log.qdrant_store import create_qdrant_store
from collections import defaultdict


def main():
    print("=" * 70)
    print("Qdrant Database Viewer")
    print("=" * 70)

    # Connect to Qdrant
    print("\n📊 Connecting to Qdrant...")
    try:
        store = create_qdrant_store()
        print("✅ Connected!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nMake sure Qdrant is running:")
        print("  docker start thirsty_kirch")
        return

    # Get statistics
    print("\n📈 Database Statistics")
    print("-" * 70)
    stats = store.get_statistics()
    print(f"Total Photos: {stats.get('total_photos', 0)}")
    print(f"Total Faces: {stats.get('total_faces', 0)}")
    print(f"Collections: {stats.get('collection_name')}, {stats.get('faces_collection_name')}")
    print(f"Embedding Dimension: {stats.get('embedding_dimension')}D")
    print(f"Qdrant URL: {stats.get('qdrant_url')}")

    # View all photos
    print("\n📸 Photos in Database")
    print("-" * 70)
    photos = store.get_all_photos(limit=100)

    if not photos:
        print("No photos stored yet.")
    else:
        for i, photo in enumerate(photos[:10], 1):  # Show first 10
            print(f"\n{i}. {photo['filename']}")
            print(f"   ID: {photo['id']}")
            if photo.get('datetime'):
                print(f"   Date: {photo['datetime']}")
            if photo.get('people'):
                print(f"   People: {', '.join(photo['people'])}")
            print(f"   Faces: {photo.get('num_faces', 0)}")

        if len(photos) > 10:
            print(f"\n... and {len(photos) - 10} more photos")

    # View all faces
    print("\n👥 Faces in Database")
    print("-" * 70)
    faces = store.get_all_faces(limit=100)

    if not faces:
        print("No faces stored yet.")
    else:
        # Group by person
        faces_by_person = defaultdict(list)
        for face in faces:
            person = face.get('person_name', 'Unknown')
            faces_by_person[person].append(face)

        print(f"Total faces: {len(faces)}")
        print(f"Unique people: {len(faces_by_person)}")
        print("\nFace counts by person:")

        for person, person_faces in sorted(faces_by_person.items(),
                                          key=lambda x: len(x[1]),
                                          reverse=True):
            print(f"  • {person}: {len(person_faces)} faces")

        # Show sample faces
        print("\nSample faces:")
        for face in faces[:5]:
            print(f"\n  Face from: {face['filename']}")
            print(f"    Person: {face.get('person_name', 'Unknown')}")
            if face.get('confidence'):
                print(f"    Confidence: {face['confidence']:.2%}")
            if face.get('datetime'):
                print(f"    Date: {face['datetime']}")

    # Search example
    if faces_by_person:
        print("\n🔍 Search Example")
        print("-" * 70)

        # Get most common person
        most_common_person = max(faces_by_person.items(), key=lambda x: len(x[1]))[0]

        if most_common_person != 'Unknown':
            print(f"Searching for faces of '{most_common_person}'...")
            person_faces = store.search_faces_by_person(most_common_person, limit=10)

            print(f"Found {len(person_faces)} faces:")
            for face in person_faces[:3]:
                print(f"  • {face['filename']} - Face #{face['face_index']}")
                if face.get('datetime'):
                    print(f"    Date: {face['datetime']}")

    print("\n" + "=" * 70)
    print("View complete!")
    print("\n💡 Tip: Open http://localhost:6333/dashboard to browse visually")
    print("=" * 70)


if __name__ == "__main__":
    main()

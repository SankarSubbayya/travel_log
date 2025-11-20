#!/usr/bin/env python3
"""Quick Qdrant database viewer"""
from qdrant_client import QdrantClient
import sys

# Connect to sapphire server
qdrant_url = sys.argv[1] if len(sys.argv) > 1 else "http://sapphire:6333"
client = QdrantClient(url=qdrant_url)

# Get stats
photos_info = client.get_collection("travel_photos")
faces_info = client.get_collection("detected_faces")

# Try to get reference faces collection
try:
    ref_faces_info = client.get_collection("reference_faces")
    has_ref_faces = True
except:
    has_ref_faces = False
    ref_faces_info = None

print("=" * 60)
print("Qdrant Database Summary")
print("=" * 60)
print(f"Photos: {photos_info.points_count}")
print(f"Detected Faces: {faces_info.points_count}")
if has_ref_faces:
    print(f"Reference Faces: {ref_faces_info.points_count}")

# Get all photos
photos = client.scroll(collection_name="travel_photos", limit=10)

if photos[0]:
    print("\n📸 Photos:")
    for p in photos[0]:
        print(f"  • {p.payload.get('filename')} - {p.payload.get('num_faces', 0)} faces")

# Get all faces
faces = client.scroll(collection_name="detected_faces", limit=10)

if faces[0]:
    print("\n👥 Detected Faces:")
    for f in faces[0]:
        person = f.payload.get('person_name', 'Unknown')
        print(f"  • {person} in {f.payload.get('filename')}")
else:
    print("\n👥 Detected Faces: None yet (upload and save a photo to store faces)")

# Show reference faces
if has_ref_faces:
    ref_faces = client.scroll(collection_name="reference_faces", limit=100)
    if ref_faces[0]:
        print("\n🔖 Reference Faces:")
        # Group by person
        people = {}
        for rf in ref_faces[0]:
            person = rf.payload.get('person_name', 'Unknown')
            if person not in people:
                people[person] = 0
            people[person] += 1

        for person, count in sorted(people.items()):
            print(f"  • {person}: {count} reference image(s)")

print(f"\n💡 View in browser: {qdrant_url}/dashboard")
print("=" * 60)

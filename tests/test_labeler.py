#!/usr/bin/env python3
"""Test face labeler with VGG-Face model"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from travel_log import FaceLabeler
from deepface import DeepFace
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 70)
print("Test Face Labeler with VGG-Face")
print("=" * 70)

# Initialize labeler with VGG-Face
print("\n1️⃣  Initializing FaceLabeler with VGG-Face...")
labeler = FaceLabeler(
    database_path="face_database",
    model_name="VGG-Face",
    distance_metric="cosine"
)
print(f"✓ Labeler initialized")
print(f"✓ Database: face_database/")
print(f"✓ Model: VGG-Face")

# Get test photo
photo_path = sys.argv[1] if len(sys.argv) > 1 else "~/personal_photos/IMG_0276_2.jpeg"
photo_path = Path(photo_path).expanduser()

if not photo_path.exists():
    print(f"\n❌ Photo not found: {photo_path}")
    sys.exit(1)

print(f"\n2️⃣  Testing with photo: {photo_path.name}")

# Detect faces
print("\n3️⃣  Detecting faces...")
faces = DeepFace.extract_faces(
    img_path=str(photo_path),
    detector_backend="retinaface",
    align=True,
    enforce_detection=False
)

print(f"✓ Detected {len(faces)} face(s)")

# Identify each face
print("\n4️⃣  Identifying faces...")
for i, face in enumerate(faces):
    print(f"\nFace {i}:")

    # Get embedding
    try:
        embedding_result = DeepFace.represent(
            img_path=face['face'],
            model_name="VGG-Face",
            enforce_detection=False
        )

        if embedding_result:
            embedding = embedding_result[0]['embedding']
            print(f"  ✓ Generated embedding: {len(embedding)}D")

            # Identify using labeler
            result = labeler.identify_face_embedding(
                embedding,
                threshold=0.20  # VGG-Face typical threshold
            )

            person = result.get('match', 'Unknown')
            distance = result.get('distance', 999)
            confidence = result.get('confidence', 0)

            if person not in ['Unknown', 'Error']:
                print(f"  ✓ Identified: {person}")
                print(f"  ✓ Distance: {distance:.4f}")
                print(f"  ✓ Confidence: {confidence:.2%}")
            else:
                print(f"  ✗ No match found (best distance: {distance:.4f})")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "=" * 70)

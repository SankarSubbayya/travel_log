#!/usr/bin/env python3
"""
Test the identify_faces function from app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qdrant_client import QdrantClient
from deepface import DeepFace
from PIL import Image
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Simulate the Streamlit session state
class FakeQdrantStore:
    def __init__(self):
        self.client = QdrantClient(url="http://sapphire:6333")

qdrant_store = FakeQdrantStore()

# Load test photo
photo_path = Path("~/personal_photos/IMG_0276_2.jpeg").expanduser()

print("=" * 70)
print("Test identify_faces() function logic")
print("=" * 70)

# Detect faces
print("\n1️⃣  Detecting faces...")
faces = DeepFace.extract_faces(
    img_path=str(photo_path),
    detector_backend="retinaface",
    align=True,
    enforce_detection=False
)
print(f"✓ Detected {len(faces)} faces")

# Create face_images list like Streamlit does
face_images = []
for i, face_data in enumerate(faces):
    face_array = (face_data['face'] * 255).astype('uint8')
    pil_image = Image.fromarray(face_array)
    face_images.append({'image': pil_image, 'bbox': face_data.get('facial_area', {})})

print(f"✓ Created {len(face_images)} PIL images")

# Now run the identification logic
print("\n2️⃣  Running identification logic...")

distance_threshold = 0.25
results = {}

for idx, face_data in enumerate(face_images):
    print(f"\nFace {idx}:")
    try:
        # Convert PIL image to numpy array
        face_array = np.array(face_data['image'])
        print(f"  Array shape: {face_array.shape}")

        # Generate embedding
        embedding_result = DeepFace.represent(
            img_path=face_array,
            model_name="VGG-Face",
            enforce_detection=False
        )

        if not embedding_result:
            print("  ✗ No embedding generated")
            results[idx] = {'match': 'Unknown', 'confidence': 0.0, 'status': 'no_embedding'}
            continue

        embedding = embedding_result[0]['embedding']
        print(f"  ✓ Generated {len(embedding)}D embedding")

        # Search in Qdrant
        search_results = qdrant_store.client.search(
            collection_name="reference_faces",
            query_vector=embedding,
            limit=3
        )

        if search_results:
            best_match = search_results[0]
            person_name = best_match.payload['person_name']
            score = best_match.score
            distance = 1 - score
            confidence = max(0, 1 - distance)

            print(f"  Top match: {person_name}")
            print(f"  Score: {score:.4f}, Distance: {distance:.4f}")

            if distance <= distance_threshold:
                results[idx] = {
                    'match': person_name,
                    'confidence': confidence,
                    'distance': distance,
                    'status': 'matched'
                }
                print(f"  ✅ MATCHED")
            else:
                results[idx] = {
                    'match': 'Unknown',
                    'confidence': confidence,
                    'distance': distance,
                    'status': 'low_confidence'
                }
                print(f"  ✗ Below threshold")
        else:
            print("  ✗ No search results")
            results[idx] = {'match': 'Unknown', 'confidence': 0.0, 'status': 'no_match'}

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results[idx] = {'match': 'Error', 'error': str(e)}

print(f"\n{'='*70}")
print("Results:")
for idx, result in results.items():
    print(f"  Face {idx}: {result.get('match', 'Unknown')} ({result.get('status', 'unknown')})")
print("=" * 70)

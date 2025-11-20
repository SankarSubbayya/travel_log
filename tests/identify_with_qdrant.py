#!/usr/bin/env python3
"""
Identify faces using Qdrant reference_faces collection
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

def identify_faces_with_qdrant(
    photo_path,
    qdrant_url="http://sapphire:6333",
    model_name="VGG-Face",
    detector="retinaface",
    threshold=0.20
):
    """Identify faces using Qdrant reference faces"""

    print("=" * 70)
    print("Face Identification using Qdrant")
    print("=" * 70)
    print(f"Photo: {Path(photo_path).name}")
    print(f"Model: {model_name}")
    print(f"Detector: {detector}")
    print(f"Threshold: {threshold}")

    # Connect to Qdrant
    print(f"\n1️⃣  Connecting to Qdrant ({qdrant_url})...")
    client = QdrantClient(url=qdrant_url)

    # Check reference faces collection
    try:
        ref_collection = client.get_collection("reference_faces")
        print(f"✓ Reference faces: {ref_collection.points_count} embeddings")
    except Exception as e:
        print(f"❌ Error accessing reference_faces collection: {e}")
        return

    # Detect faces in photo
    print(f"\n2️⃣  Detecting faces...")
    faces = DeepFace.extract_faces(
        img_path=str(photo_path),
        detector_backend=detector,
        align=True,
        enforce_detection=False
    )
    print(f"✓ Detected {len(faces)} face(s)")

    # Identify each face
    print(f"\n3️⃣  Identifying faces using Qdrant similarity search...")

    successful_matches = 0

    for idx, face_data in enumerate(faces):
        print(f"\nFace {idx}:")

        try:
            # Generate embedding for detected face
            embedding_result = DeepFace.represent(
                img_path=face_data['face'],
                model_name=model_name,
                enforce_detection=False
            )

            if not embedding_result:
                print("  ✗ Could not generate embedding")
                continue

            embedding = embedding_result[0]['embedding']
            print(f"  ✓ Generated {len(embedding)}D embedding")

            # Search in Qdrant
            search_results = client.search(
                collection_name="reference_faces",
                query_vector=embedding,
                limit=3  # Get top 3 matches
            )

            if search_results:
                best_match = search_results[0]
                person_name = best_match.payload['person_name']
                score = best_match.score

                # For cosine similarity, score is 1 - distance
                # Higher score = better match
                distance = 1 - score

                print(f"  Top match: {person_name}")
                print(f"    Score: {score:.4f}")
                print(f"    Distance: {distance:.4f}")

                # Check threshold
                if distance <= threshold:
                    print(f"  ✅ MATCHED: {person_name}")
                    successful_matches += 1

                    # Show runner-up matches
                    if len(search_results) > 1:
                        print(f"  Other possibilities:")
                        for i, result in enumerate(search_results[1:3], 1):
                            other_name = result.payload['person_name']
                            other_score = result.score
                            other_dist = 1 - other_score
                            print(f"    {i}. {other_name} (dist: {other_dist:.4f})")
                else:
                    print(f"  ✗ No match (distance {distance:.4f} > threshold {threshold})")
            else:
                print("  ✗ No matches found in Qdrant")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n{'='*70}")
    print(f"Results: {successful_matches}/{len(faces)} faces matched")

    if successful_matches == len(faces):
        print("✅ SUCCESS! All faces identified!")
    elif successful_matches > 0:
        print(f"⚠️  Partial success: {successful_matches} faces matched")
    else:
        print("❌ No faces matched")

    print("=" * 70)

    return successful_matches


if __name__ == "__main__":
    photo = sys.argv[1] if len(sys.argv) > 1 else "~/personal_photos/IMG_0276_2.jpeg"
    photo = str(Path(photo).expanduser())

    if not Path(photo).exists():
        print(f"❌ Photo not found: {photo}")
        sys.exit(1)

    identify_faces_with_qdrant(photo)

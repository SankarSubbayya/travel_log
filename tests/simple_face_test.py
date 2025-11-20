#!/usr/bin/env python3
"""
Simple face matching test using DeepFace directly.
Tests different configurations to find working settings.
"""

import sys
from pathlib import Path
from deepface import DeepFace
import numpy as np

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_face_matching(photo_path, face_db="face_database"):
    """Test face matching with different configurations."""

    print("=" * 70)
    print("Simple Face Matching Test")
    print("=" * 70)
    print(f"\nPhoto: {Path(photo_path).name}")
    print(f"Face Database: {face_db}\n")

    # Test configurations
    configs = [
        {"model": "Facenet512", "detector": "retinaface", "distance": "cosine"},
        {"model": "Facenet512", "detector": "mtcnn", "distance": "cosine"},
        {"model": "Facenet", "detector": "retinaface", "distance": "cosine"},
        {"model": "VGG-Face", "detector": "retinaface", "distance": "cosine"},
    ]

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['model']} + {config['detector']}")
        print("=" * 70)

        try:
            # Use DeepFace.find to match faces
            results = DeepFace.find(
                img_path=photo_path,
                db_path=face_db,
                model_name=config['model'],
                detector_backend=config['detector'],
                distance_metric=config['distance'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(results, list) and len(results) > 0:
                print(f"\n✅ Found {len(results)} face(s)!")

                for i, df in enumerate(results):
                    if not df.empty:
                        # Get the best match (first row)
                        match_path = df.iloc[0]['identity']
                        distance = df.iloc[0]['distance']

                        # Extract person name from path
                        person = Path(match_path).parent.name

                        print(f"\n  Face {i}:")
                        print(f"    Match: {person}")
                        print(f"    Distance: {distance:.4f}")
                        print(f"    File: {Path(match_path).name}")

                        # Show threshold info
                        if config['model'] == "Facenet512":
                            threshold = 0.4  # Typical threshold for Facenet512
                            if distance < threshold:
                                print(f"    ✓ Good match (< {threshold})")
                            else:
                                print(f"    ⚠️  Weak match (> {threshold})")
                    else:
                        print(f"  Face {i}: No match found")

                # Success! Save this config
                print(f"\n🎯 SUCCESS with this configuration!")
                print(f"   Model: {config['model']}")
                print(f"   Detector: {config['detector']}")
                break
            else:
                print("❌ No faces detected or matched")

        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    photo = sys.argv[1] if len(sys.argv) > 1 else "~/personal_photos/IMG_0276_2.jpeg"
    photo = str(Path(photo).expanduser())

    if not Path(photo).exists():
        print(f"❌ Photo not found: {photo}")
        sys.exit(1)

    test_face_matching(photo)

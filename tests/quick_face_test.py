#!/usr/bin/env python3
"""Quick face matching test"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from travel_log.face_manager import TravelLogFaceManager
from PIL import Image
import numpy as np

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

photo_path = sys.argv[1] if len(sys.argv) > 1 else "~/personal_photos/IMG_0276_2.jpeg"
photo_path = Path(photo_path).expanduser()

print("=" * 70)
print("Quick Face Matching Test")
print("=" * 70)
print(f"\nPhoto: {photo_path.name}\n")

# Test different configurations
configs = [
    ("Facenet512", "retinaface", 0.3),
    ("Facenet512", "retinaface", 0.4),
    ("Facenet512", "retinaface", 0.5),
    ("Facenet512", "mtcnn", 0.4),
    ("Facenet", "retinaface", 0.4),
]

for model, detector, threshold in configs:
    print(f"\n{'='*70}")
    print(f"Testing: {model} + {detector} (threshold={threshold})")
    print("=" * 70)

    try:
        # Initialize face manager
        fm = TravelLogFaceManager(
            workspace_dir=".",
            database_dir="face_database",
            detector_backend=detector,
            recognition_model=model
        )

        print("Initializing database...")
        fm.initialize_database()
        print(f"✓ Loaded {len(fm.known_faces)} people")

        # Identify faces
        print(f"\nIdentifying faces in {photo_path.name}...")
        results = fm.identify_faces_in_image(
            str(photo_path),
            threshold=threshold
        )

        if results:
            print(f"\n✅ Found {len(results)} faces:")
            for idx, result in results.items():
                match = result.get('match', 'Unknown')
                conf = result.get('confidence', 0)
                dist = result.get('distance', 999)

                if match not in ['Unknown', 'Error']:
                    print(f"   Face {idx}: {match} (conf={conf:.2%}, dist={dist:.3f}) ✓")
                else:
                    print(f"   Face {idx}: No match (dist={dist:.3f})")

            # Count successful matches
            matches = [r for r in results.values() if r.get('match') not in ['Unknown', 'Error']]
            if matches:
                print(f"\n🎯 SUCCESS! {len(matches)} match(es) found!")
                print(f"   Use: model={model}, detector={detector}, threshold={threshold}")
                break
        else:
            print("❌ No faces detected")

    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "=" * 70)

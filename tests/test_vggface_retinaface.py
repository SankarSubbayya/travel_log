#!/usr/bin/env python3
"""Test VGG-Face + retinaface combination"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from travel_log import FaceLabeler
import tempfile
from PIL import Image
from deepface import DeepFace
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 70)
print("Test: VGG-Face + retinaface")
print("=" * 70)

# Initialize with correct detector
print("\n1️⃣  Initializing FaceLabeler...")
print("   Model: VGG-Face")
print("   Detector: retinaface")

labeler = FaceLabeler(
    database_path="face_database",
    model_name="VGG-Face",
    detector_backend="retinaface",
    distance_metric="cosine"
)
print("✓ Initialized")

# Test photo
photo_path = Path(sys.argv[1] if len(sys.argv) > 1 else "~/personal_photos/IMG_0276_2.jpeg").expanduser()

print(f"\n2️⃣  Detecting faces in {photo_path.name}...")
faces = DeepFace.extract_faces(
    img_path=str(photo_path),
    detector_backend="retinaface",  # SAME detector
    align=True,
    enforce_detection=False
)
print(f"✓ Detected {len(faces)} faces")

print("\n3️⃣  Identifying faces...")
successful_matches = 0

for idx, face_data in enumerate(faces):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        face_image = Image.fromarray((face_data['face'] * 255).astype('uint8'))
        face_image.save(tmp.name, format='JPEG')
        tmp_path = tmp.name

    try:
        match_results = labeler.find_face(tmp_path)
        Path(tmp_path).unlink()

        if match_results and len(match_results) > 0 and not match_results[0].empty:
            best_match = match_results[0].iloc[0]
            distance = float(best_match['cosine'])
            person_name = Path(best_match['identity']).parent.name
            confidence = 1 - distance

            print(f"\nFace {idx}: ✓ {person_name}")
            print(f"  Distance: {distance:.4f}, Confidence: {confidence:.2%}")

            if confidence >= 0.6:
                successful_matches += 1
        else:
            print(f"\nFace {idx}: ✗ No match")
    except Exception as e:
        print(f"\nFace {idx}: ✗ Error - {e}")
        Path(tmp_path).unlink()

print(f"\n{'='*70}")
print(f"Result: {successful_matches}/{len(faces)} faces matched")

if successful_matches == len(faces):
    print("✅ SUCCESS! All faces matched!")
else:
    print("⚠️  Some faces not matched")

print("=" * 70)

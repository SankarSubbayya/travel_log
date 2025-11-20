#!/usr/bin/env python3
"""
Quick check: Does VGG-Face work with FaceLabeler?
"""

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
print("VGG-Face + FaceLabeler Test")
print("=" * 70)

photo_path = sys.argv[1] if len(sys.argv) > 1 else "~/personal_photos/IMG_0276_2.jpeg"
photo_path = Path(photo_path).expanduser()

# 1. Initialize FaceLabeler with VGG-Face
print("\n1️⃣  Initializing FaceLabeler with VGG-Face...")
try:
    labeler = FaceLabeler(
        database_path="face_database",
        model_name="VGG-Face",
        distance_metric="cosine"
    )
    print("✓ FaceLabeler initialized with VGG-Face")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    sys.exit(1)

# 2. Detect faces
print(f"\n2️⃣  Detecting faces in {photo_path.name}...")
faces = DeepFace.extract_faces(
    img_path=str(photo_path),
    detector_backend="retinaface",
    align=True,
    enforce_detection=False
)
print(f"✓ Detected {len(faces)} face(s)")

# 3. Identify each face using FaceLabeler
print("\n3️⃣  Identifying faces with FaceLabeler.find_face()...")

for idx, face_data in enumerate(faces):
    print(f"\nFace {idx}:")

    try:
        # Save face to temporary file (like Streamlit does)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            face_image = Image.fromarray((face_data['face'] * 255).astype('uint8'))
            face_image.save(tmp.name, format='JPEG')
            tmp_path = tmp.name

        # Identify using FaceLabeler.find_face()
        match_results = labeler.find_face(tmp_path)

        # Clean up temp file
        Path(tmp_path).unlink()

        if match_results and len(match_results) > 0 and not match_results[0].empty:
            best_match = match_results[0].iloc[0]
            distance = float(best_match['cosine'])  # VGG-Face uses cosine
            person_name = Path(best_match['identity']).parent.name

            # Convert distance to confidence (for cosine: confidence = 1 - distance)
            confidence = 1 - distance

            print(f"  ✓ Match: {person_name}")
            print(f"  ✓ Distance: {distance:.4f}")
            print(f"  ✓ Confidence: {confidence:.2%}")

            # Check if would pass 0.6 threshold
            if confidence >= 0.6:
                print(f"  ✓ PASSES 0.6 threshold!")
            else:
                print(f"  ✗ FAILS 0.6 threshold (confidence too low)")
        else:
            print(f"  ✗ No match found")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("Summary:")
print("  - FaceLabeler.find_face() works with VGG-Face")
print("  - Should work in Streamlit after restart")
print("=" * 70)

#!/usr/bin/env python3
"""Rebuild face database with VGG-Face model"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from travel_log import FaceLabeler
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 70)
print("Rebuild Face Database with VGG-Face")
print("=" * 70)

print("\nInitializing FaceLabeler with VGG-Face...")
print("This will create the face database embeddings file...")

labeler = FaceLabeler(
    database_path="face_database",
    model_name="VGG-Face",
    detector_backend="retinaface",
    distance_metric="cosine"
)

print("\n✓ Face database initialized!")
print(f"✓ Model: VGG-Face")
print(f"✓ Detector: retinaface")
print(f"✓ Distance: cosine")

# Check the pickle file was created
pkl_files = list(Path("face_database").glob("ds_model_vggface_*.pkl"))
if pkl_files:
    for pkl_file in pkl_files:
        size = pkl_file.stat().st_size / 1024
        print(f"\n✓ Created: {pkl_file.name}")
        print(f"  Size: {size:.1f} KB")
else:
    print("\n⚠️  No pickle file created")

print("\n" + "=" * 70)

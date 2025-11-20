#!/usr/bin/env python3
"""
Diagnose Face Matching Issues

Helps identify why faces aren't being matched.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from travel_log.face_detector import extract_faces_from_photo
from travel_log.face_manager import TravelLogFaceManager
from travel_log.image_metadata import extract_metadata
import numpy as np
from deepface import DeepFace


def main():
    print("=" * 70)
    print("Face Matching Diagnostic Tool")
    print("=" * 70)

    # Get photo path
    if len(sys.argv) < 2:
        print("\nUsage: python diagnose_face_matching.py <photo_path>")
        return

    photo_path = Path(sys.argv[1])
    if not photo_path.exists():
        print(f"❌ Photo not found: {photo_path}")
        return

    print(f"\n📸 Analyzing: {photo_path.name}\n")

    # 1. Check face database
    print("1️⃣  Checking Face Database...")
    print("-" * 70)

    face_db_path = Path("face_database")
    if not face_db_path.exists():
        print("❌ face_database directory not found!")
        return

    # Count reference images
    person_dirs = [d for d in face_db_path.iterdir() if d.is_dir()]
    print(f"✓ Found {len(person_dirs)} people in database:")

    for person_dir in person_dirs:
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg"))
        print(f"  • {person_dir.name}: {len(images)} reference image(s)")

    # Check pickle file
    pickle_file = face_db_path / "ds_model_facenet512_detector_mtcnn_aligned_normalization_base_expand_0.pkl"
    if pickle_file.exists():
        print(f"\n✓ Embeddings file found: {pickle_file.name}")
        print(f"  Size: {pickle_file.stat().st_size / 1024:.1f} KB")
    else:
        print("\n⚠️  No embeddings file found - database needs initialization")

    # 2. Detect faces in test photo
    print(f"\n2️⃣  Detecting Faces in Test Photo...")
    print("-" * 70)

    backends = ["retinaface", "mtcnn", "opencv"]
    detected_faces_per_backend = {}

    for backend in backends:
        try:
            print(f"\nTrying {backend}...")
            faces = DeepFace.extract_faces(
                img_path=str(photo_path),
                detector_backend=backend,
                align=True,
                enforce_detection=False
            )
            detected_faces_per_backend[backend] = faces
            print(f"✓ {backend}: Detected {len(faces)} face(s)")

            for i, face in enumerate(faces):
                conf = face.get('confidence', 1.0)
                has_emb = 'embedding' in face
                emb_dim = len(face['embedding']) if has_emb else 0
                print(f"  Face {i}: confidence={conf:.2%}, has_embedding={has_emb}")

        except Exception as e:
            print(f"✗ {backend}: {e}")

    if not detected_faces_per_backend:
        print("\n❌ No faces detected with any backend!")
        return

    # Use the backend that found the most faces
    best_backend = max(detected_faces_per_backend.items(), key=lambda x: len(x[1]))
    backend_name = best_backend[0]
    detected_faces = best_backend[1]

    print(f"\n✓ Using {backend_name} (found {len(detected_faces)} faces)")

    # Generate embeddings for detected faces
    print("\nGenerating embeddings...")
    for i, face in enumerate(detected_faces):
        try:
            # DeepFace.represent to get embeddings
            embedding = DeepFace.represent(
                img_path=face['face'],
                model_name="Facenet512",
                enforce_detection=False
            )
            if isinstance(embedding, list) and len(embedding) > 0:
                detected_faces[i]['embedding'] = np.array(embedding[0]['embedding'])
                print(f"✓ Face {i}: Generated 512D embedding")
        except Exception as e:
            print(f"✗ Face {i}: Could not generate embedding - {e}")

    # 3. Initialize face database with different settings
    print(f"\n3️⃣  Testing Face Identification...")
    print("-" * 70)

    test_configs = [
        {"model": "Facenet512", "detector": "retinaface", "threshold": 0.4},
        {"model": "Facenet512", "detector": "mtcnn", "threshold": 0.4},
        {"model": "Facenet512", "detector": "retinaface", "threshold": 0.6},
        {"model": "Facenet", "detector": "retinaface", "threshold": 0.4},
    ]

    for config in test_configs:
        print(f"\n📋 Config: model={config['model']}, detector={config['detector']}, threshold={config['threshold']}")

        try:
            face_manager = TravelLogFaceManager(
                database_path="face_database",
                model_name=config['model'],
                detector_backend=config['detector']
            )

            print("   Initializing database...")
            face_manager.initialize_database()
            print(f"   ✓ Loaded {len(face_manager.identities)} identities")

            # Try to match each detected face
            matches_found = 0
            for i, face in enumerate(detected_faces):
                if 'embedding' not in face:
                    print(f"   Face {i}: No embedding")
                    continue

                result = face_manager.identify_face_embedding(
                    face['embedding'],
                    threshold=config['threshold']
                )

                match_name = result.get('match', 'Unknown')
                confidence = result.get('confidence', 0.0)
                distance = result.get('distance', 999)

                if match_name not in ['Unknown', 'Error']:
                    matches_found += 1
                    print(f"   ✓ Face {i}: {match_name} (confidence: {confidence:.2%}, distance: {distance:.3f})")
                else:
                    print(f"   ✗ Face {i}: No match (best distance: {distance:.3f})")

            print(f"   Summary: {matches_found}/{len(detected_faces)} faces matched")

            if matches_found > 0:
                print(f"\n✅ SUCCESS! Found matches with this configuration:")
                print(f"   Model: {config['model']}")
                print(f"   Detector: {config['detector']}")
                print(f"   Threshold: {config['threshold']}")
                break

        except Exception as e:
            print(f"   ✗ Error: {e}")

    # 4. Recommendations
    print("\n" + "=" * 70)
    print("Recommendations")
    print("=" * 70)

    if matches_found == 0:
        print("""
⚠️  No matches found. Possible issues:

1. **Reference images quality**:
   - Make sure face_database contains clear, front-facing photos
   - One face per image
   - Good lighting

2. **Backend mismatch**:
   - Use the SAME detector for both reference and test photos
   - Recommended: retinaface or mtcnn

3. **Lower threshold**:
   - Try threshold=0.3 or 0.4 (default is 0.6)
   - Lower = more lenient matching

4. **Rebuild embeddings**:
   - Delete the .pkl file
   - Reinitialize the database

5. **Model selection**:
   - Try Facenet512 (recommended)
   - Or Facenet, VGG-Face

🔧 Fix in Streamlit:
   Go to Face Identification tab:
   - Detector: retinaface
   - Model: Facenet512
   - Threshold: 0.4
   - Click "Initialize Face Database"
""")
    else:
        print(f"""
✅ Matches found successfully!

Use these settings in Streamlit:
   - Model: {config['model']}
   - Detector: {config['detector']}
   - Threshold: {config['threshold']}
""")


if __name__ == "__main__":
    main()

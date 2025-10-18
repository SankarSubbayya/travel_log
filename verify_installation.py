#!/usr/bin/env python3
"""
Installation Verification Script

This script verifies that the face recognition modules are properly installed
and can be imported without errors.
"""

import sys

def verify_installation():
    """Verify all face recognition modules can be imported."""
    print("="*60)
    print("Verifying Travel Log Face Recognition Installation")
    print("="*60)
    
    errors = []
    
    # Test 1: Import main package
    print("\n✓ Test 1: Importing travel_log package...")
    try:
        import travel_log
        print("  ✓ travel_log imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import travel_log: {e}")
        errors.append(f"travel_log import: {e}")
    
    # Test 2: Import FaceDetector
    print("\n✓ Test 2: Importing FaceDetector...")
    try:
        from travel_log import FaceDetector
        print("  ✓ FaceDetector imported successfully")
        print(f"  ✓ Supported backends: {FaceDetector.SUPPORTED_BACKENDS}")
    except Exception as e:
        print(f"  ✗ Failed to import FaceDetector: {e}")
        errors.append(f"FaceDetector import: {e}")
    
    # Test 3: Import FaceEmbeddings
    print("\n✓ Test 3: Importing FaceEmbeddings...")
    try:
        from travel_log import FaceEmbeddings
        print("  ✓ FaceEmbeddings imported successfully")
        print(f"  ✓ Supported models: {', '.join(FaceEmbeddings.SUPPORTED_MODELS[:3])}...")
    except Exception as e:
        print(f"  ✗ Failed to import FaceEmbeddings: {e}")
        errors.append(f"FaceEmbeddings import: {e}")
    
    # Test 4: Import FaceLabeler
    print("\n✓ Test 4: Importing FaceLabeler...")
    try:
        from travel_log import FaceLabeler
        print("  ✓ FaceLabeler imported successfully")
        print(f"  ✓ Supported distance metrics: {FaceLabeler.DISTANCE_METRICS}")
    except Exception as e:
        print(f"  ✗ Failed to import FaceLabeler: {e}")
        errors.append(f"FaceLabeler import: {e}")
    
    # Test 5: Import TravelLogFaceManager
    print("\n✓ Test 5: Importing TravelLogFaceManager...")
    try:
        from travel_log import TravelLogFaceManager
        print("  ✓ TravelLogFaceManager imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import TravelLogFaceManager: {e}")
        errors.append(f"TravelLogFaceManager import: {e}")
    
    # Test 6: Import convenience functions
    print("\n✓ Test 6: Importing convenience functions...")
    try:
        from travel_log import (
            extract_faces_from_photo,
            generate_face_embedding,
            identify_faces_in_photos,
            create_face_manager
        )
        print("  ✓ All convenience functions imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import convenience functions: {e}")
        errors.append(f"Convenience functions import: {e}")
    
    # Test 7: Check DeepFace
    print("\n✓ Test 7: Checking DeepFace installation...")
    try:
        import deepface
        print(f"  ✓ DeepFace version: {deepface.__version__}")
    except Exception as e:
        print(f"  ✗ DeepFace not available: {e}")
        errors.append(f"DeepFace: {e}")
    
    # Test 8: Check OpenCV
    print("\n✓ Test 8: Checking OpenCV installation...")
    try:
        import cv2
        print(f"  ✓ OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"  ✗ OpenCV not available: {e}")
        errors.append(f"OpenCV: {e}")
    
    # Test 9: Check NumPy
    print("\n✓ Test 9: Checking NumPy installation...")
    try:
        import numpy as np
        print(f"  ✓ NumPy version: {np.__version__}")
    except Exception as e:
        print(f"  ✗ NumPy not available: {e}")
        errors.append(f"NumPy: {e}")
    
    # Test 10: Check Pillow
    print("\n✓ Test 10: Checking Pillow installation...")
    try:
        import PIL
        print(f"  ✓ Pillow version: {PIL.__version__}")
    except Exception as e:
        print(f"  ✗ Pillow not available: {e}")
        errors.append(f"Pillow: {e}")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        print("❌ INSTALLATION INCOMPLETE")
        print("="*60)
        print(f"\nFound {len(errors)} error(s):")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
        print("\nPlease run: uv sync")
        return False
    else:
        print("✅ INSTALLATION VERIFIED SUCCESSFULLY")
        print("="*60)
        print("\nAll face recognition modules are ready to use!")
        print("\nNext steps:")
        print("1. Read: FACE_RECOGNITION_QUICKSTART.md")
        print("2. Try: examples/face_detection_example.py")
        print("3. Explore: docs/face-recognition-guide.md")
        return True


if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)


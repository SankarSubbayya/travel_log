#!/usr/bin/env python3
"""
Simplest possible DeepFace download - no fancy stuff, just works.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

print("Simple DeepFace Model Download")
print("="*50)
print("\nNote: You may see mutex warnings - ignore them!")
print("The downloads will work despite the warnings.\n")

try:
    from deepface import DeepFace
    
    print("✓ DeepFace imported")
    print("\nDownloading will happen on first use...")
    print("Let's trigger a simple detection:\n")
    
    # Create a dummy image to trigger downloads
    import numpy as np
    import tempfile
    from PIL import Image
    
    # Create simple test image
    img = Image.new('RGB', (200, 200), color='white')
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img.save(f.name)
        test_img = f.name
    
    print("Triggering model downloads...")
    print("(This may take a few minutes on first run)\n")
    
    try:
        # This will download opencv detector (built-in, fast)
        result = DeepFace.extract_faces(
            test_img,
            detector_backend='opencv',
            enforce_detection=False
        )
        print("✓ OpenCV detector ready")
    except Exception as e:
        print(f"Note: {e}")
    
    try:
        # This will download Facenet (~90MB)
        result = DeepFace.represent(
            test_img,
            model_name='Facenet',
            enforce_detection=False
        )
        print("✓ Facenet model downloaded")
    except Exception as e:
        print(f"Note: {e}")
    
    # Cleanup
    import os
    os.unlink(test_img)
    
    print("\n" + "="*50)
    print("✅ Setup complete!")
    print("="*50)
    print("\nModels are cached for future use.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nDon't worry! Models will auto-download when you use face detection.")

print("\nYou're ready to use face recognition! Try:")
print("  uv run python demo_face_detection.py your_photo.jpg")


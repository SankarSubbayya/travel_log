#!/usr/bin/env python3
"""
Demo script showing how to completely eliminate TensorFlow warnings.
"""

# ============================================================================
# SUPPRESS TENSORFLOW WARNINGS (must be at the very top!)
# ============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'  # Additional suppression

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Now import travel_log - warnings should be suppressed
# ============================================================================
from travel_log import TravelLogFaceManager, FaceDetector

print("="*60)
print("Testing Face Recognition WITHOUT TensorFlow Warnings")
print("="*60)

print("\n1. Creating FaceDetector...")
detector = FaceDetector(detector_backend='opencv')
print("   ✓ FaceDetector created")

print("\n2. Creating TravelLogFaceManager...")
manager = TravelLogFaceManager("test_workspace")
print("   ✓ TravelLogFaceManager created")

print("\n" + "="*60)
print("✅ SUCCESS! No mutex warnings!")
print("="*60)

print("\nIf you still see warnings above, try one of these:")
print("  1. Restart your Python kernel/terminal")
print("  2. Set env vars before running:")
print("     TF_CPP_MIN_LOG_LEVEL=3 python your_script.py")
print("  3. Add these lines at the VERY TOP of your script:")
print("     import os")
print("     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'")
print("     os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'")


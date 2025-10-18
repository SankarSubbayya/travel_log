#!/usr/bin/env python3
"""
Template script with TensorFlow warnings completely suppressed.

Copy this template for your own scripts to avoid the mutex warning.
"""

# ============================================================================
# STEP 1: SUPPRESS WARNINGS (must be at the very top!)
# ============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'  # Additional suppression

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 2: NOW import travel_log modules
# ============================================================================
from travel_log import (
    TravelLogFaceManager,
    FaceDetector,
    FaceEmbeddings,
    FaceLabeler
)

# ============================================================================
# STEP 3: Your code here - no warnings!
# ============================================================================

def main():
    """Your main code here."""
    
    print("="*60)
    print("Face Recognition Example - No Warnings!")
    print("="*60)
    
    # Example: Process a photo
    manager = TravelLogFaceManager("my_workspace")
    print("\n✓ TravelLogFaceManager created successfully")
    print("✓ No TensorFlow mutex warnings!")
    
    # Add your code here
    # result = manager.process_photo("my_photo.jpg")
    # print(f"Found {result['num_faces']} faces")
    
    print("\nReady to process your photos!")


if __name__ == "__main__":
    main()


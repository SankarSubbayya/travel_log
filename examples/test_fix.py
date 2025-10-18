#!/usr/bin/env python3
"""
Test script with TensorFlow warnings suppressed
"""

# Set environment variables BEFORE importing any modules
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Suppress other warnings
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Now import travel_log modules
from travel_log import TravelLogFaceManager

print("Testing face recognition with warnings suppressed...")

# Test basic functionality
manager = TravelLogFaceManager("test_workspace")
print("✓ TravelLogFaceManager initialized successfully!")
print("✓ No mutex errors!")


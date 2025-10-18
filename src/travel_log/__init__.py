#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#   This code is part of the training material and, therefore, part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

# IMPORTANT: Suppress TensorFlow warnings BEFORE any imports that use TensorFlow
import os
import warnings

# Set TensorFlow environment variables to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'  # Additional verbose logging suppression

# Suppress Python warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Now proceed with regular imports
from svlearn.config.configuration import ConfigurationMixin

from dotenv import load_dotenv
load_dotenv()

config = ConfigurationMixin().load_config()

# Face recognition modules - TensorFlow warnings already suppressed above
from .face_detector import FaceDetector, extract_faces_from_photo
from .face_embeddings import FaceEmbeddings, generate_face_embedding
from .face_labeler import FaceLabeler, identify_faces_in_photos
from .face_manager import TravelLogFaceManager, create_face_manager

__all__ = [
    'config',
    'FaceDetector',
    'FaceEmbeddings',
    'FaceLabeler',
    'TravelLogFaceManager',
    'extract_faces_from_photo',
    'generate_face_embedding',
    'identify_faces_in_photos',
    'create_face_manager',
]

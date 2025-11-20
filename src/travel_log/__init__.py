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
from .caption_generator import CaptionGenerator
from .image_utils import (
    convert_heic_to_jpg,
    is_heic_file,
    ensure_compatible_image,
    get_supported_image_extensions,
    is_supported_image,
    HEIC_SUPPORTED
)
from .exif_utils import (
    get_exif_data,
    get_datetime,
    get_gps_coordinates,
    get_camera_info,
    get_image_dimensions,
    get_image_orientation,
    get_complete_metadata,
    format_gps_for_maps,
    has_gps_data,
    has_datetime
)

__all__ = [
    'config',
    'FaceDetector',
    'FaceEmbeddings',
    'FaceLabeler',
    'TravelLogFaceManager',
    'CaptionGenerator',
    'extract_faces_from_photo',
    'generate_face_embedding',
    'identify_faces_in_photos',
    'create_face_manager',
    # Image utilities
    'convert_heic_to_jpg',
    'is_heic_file',
    'ensure_compatible_image',
    'get_supported_image_extensions',
    'is_supported_image',
    'HEIC_SUPPORTED',
    # EXIF utilities
    'get_exif_data',
    'get_datetime',
    'get_gps_coordinates',
    'get_camera_info',
    'get_image_dimensions',
    'get_image_orientation',
    'get_complete_metadata',
    'format_gps_for_maps',
    'has_gps_data',
    'has_datetime',
]

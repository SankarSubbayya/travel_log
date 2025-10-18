"""
Setup utility to completely suppress TensorFlow warnings.

USAGE:
    Import this BEFORE importing travel_log:
    
    import setup_suppress_warnings  # Import this first!
    from travel_log import TravelLogFaceManager  # Now import travel_log
    
OR:
    Call the setup function at the very start of your script:
    
    import setup_suppress_warnings
    setup_suppress_warnings.suppress_all()
    
    from travel_log import TravelLogFaceManager
"""

import os
import sys
import warnings
import logging


def suppress_all():
    """
    Completely suppress TensorFlow and other framework warnings.
    Call this at the very beginning of your script.
    """
    
    # Suppress TensorFlow logging (must be set before TensorFlow import)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    
    # Additional TensorFlow settings
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode (avoids some GPU warnings)
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    
    # Suppress specific warning categories
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Suppress TensorFlow logging after import (if already imported)
    try:
        import tensorflow as tf
        tf.get_logger().setLevel(logging.ERROR)
        tf.autograph.set_verbosity(0)
    except ImportError:
        pass  # TensorFlow not yet imported
    
    # Suppress other framework warnings
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    
    print("✓ All warnings suppressed")


# Automatically suppress on import
suppress_all()


if __name__ == "__main__":
    print("Warnings suppression setup complete!")
    print("\nTo use in your scripts:")
    print("  import setup_suppress_warnings")
    print("  from travel_log import TravelLogFaceManager")


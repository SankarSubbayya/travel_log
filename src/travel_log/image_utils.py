"""
Image Utilities Module

Handles various image formats including HEIC (Apple's High Efficiency Image Container).
Converts images to formats compatible with face detection.
"""

import os
from pathlib import Path
from typing import Union
from PIL import Image
import tempfile

# Register HEIF/HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False


def convert_heic_to_jpg(heic_path: Union[str, Path], output_path: Union[str, Path] = None) -> str:
    """
    Convert HEIC/HEIF image to JPEG format.
    
    Args:
        heic_path: Path to HEIC/HEIF file
        output_path: Optional output path for JPEG. If None, creates temp file.
        
    Returns:
        Path to the converted JPEG file
        
    Raises:
        ValueError: If HEIC support not available
        FileNotFoundError: If input file doesn't exist
    """
    if not HEIC_SUPPORTED:
        raise ValueError(
            "HEIC support not available. Install pillow-heif: pip install pillow-heif"
        )
    
    heic_path = Path(heic_path)
    if not heic_path.exists():
        raise FileNotFoundError(f"HEIC file not found: {heic_path}")
    
    # Open HEIC file
    img = Image.open(heic_path)
    
    # Convert to RGB if necessary (some HEICs are in different color modes)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    # Determine output path
    if output_path is None:
        # Create temporary file
        fd, output_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
    else:
        output_path = str(output_path)
    
    # Save as JPEG
    img.save(output_path, 'JPEG', quality=95)
    
    return output_path


def is_heic_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is HEIC/HEIF format based on extension.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file has HEIC/HEIF extension
    """
    ext = Path(file_path).suffix.lower()
    return ext in ['.heic', '.heif']


def ensure_compatible_image(image_path: Union[str, Path]) -> str:
    """
    Ensure image is in a format compatible with face detection.
    Converts HEIC to JPEG if necessary.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Path to compatible image (original or converted)
        
    Note:
        If conversion is done, returns path to temporary file.
        Caller is responsible for cleanup if needed.
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # If HEIC, convert to JPEG
    if is_heic_file(image_path):
        if not HEIC_SUPPORTED:
            raise ValueError(
                f"HEIC file detected but support not available. "
                f"Install pillow-heif: pip install pillow-heif"
            )
        return convert_heic_to_jpg(image_path)
    
    # Return original path for other formats
    return str(image_path)


def get_supported_image_extensions() -> list:
    """
    Get list of supported image file extensions.
    
    Returns:
        List of supported extensions (with dots, lowercase)
    """
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    if HEIC_SUPPORTED:
        extensions.extend(['.heic', '.heif'])
    
    return extensions


def is_supported_image(file_path: Union[str, Path]) -> bool:
    """
    Check if file is a supported image format.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file extension is supported
    """
    ext = Path(file_path).suffix.lower()
    return ext in get_supported_image_extensions()


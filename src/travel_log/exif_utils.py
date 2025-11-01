"""
EXIF Metadata Utilities

Extract EXIF metadata from images including:
- Timestamp (when photo was taken)
- GPS coordinates (where photo was taken)
- Camera information
- Other metadata
"""

from pathlib import Path
from typing import Union, Dict, Optional, Tuple
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def get_exif_data(image_path: Union[str, Path]) -> Dict:
    """
    Extract all EXIF data from an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary of EXIF tags and values
    """
    try:
        image = Image.open(image_path)
        exif_data = {}
        
        # Get raw EXIF data
        exif = image.getexif()
        
        if exif is None:
            return {}
        
        # Convert numeric tags to readable names
        for tag_id, value in exif.items():
            tag_name = TAGS.get(tag_id, tag_id)
            exif_data[tag_name] = value
        
        return exif_data
        
    except Exception as e:
        logger.error(f"Error reading EXIF from {image_path}: {e}")
        return {}


def get_datetime(image_path: Union[str, Path]) -> Optional[datetime]:
    """
    Get the date and time when the photo was taken.
    
    Args:
        image_path: Path to image file
        
    Returns:
        datetime object or None if not available
        
    Example:
        >>> dt = get_datetime("photo.jpg")
        >>> print(dt.strftime("%Y-%m-%d %H:%M:%S"))
        2024-03-15 14:30:45
    """
    exif = get_exif_data(image_path)
    
    # Try different datetime tags
    datetime_tags = ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']
    
    for tag in datetime_tags:
        if tag in exif:
            try:
                # Parse EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
                dt_str = exif[tag]
                dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                return dt
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse datetime from {tag}: {e}")
                continue
    
    return None


def _convert_to_degrees(value: Tuple) -> float:
    """
    Convert GPS coordinates to decimal degrees.
    
    Args:
        value: Tuple of (degrees, minutes, seconds)
        
    Returns:
        Decimal degrees
    """
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)


def get_gps_coordinates(image_path: Union[str, Path]) -> Optional[Dict[str, float]]:
    """
    Extract GPS coordinates from image EXIF data.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with 'latitude', 'longitude', 'altitude' (if available)
        or None if GPS data not found
        
    Example:
        >>> coords = get_gps_coordinates("photo.jpg")
        >>> print(f"Lat: {coords['latitude']}, Lon: {coords['longitude']}")
        Lat: 37.7749, Lon: -122.4194
    """
    try:
        image = Image.open(image_path)
        exif = image.getexif()
        
        if not exif:
            return None
        
        # Get GPS IFD (Image File Directory)
        gps_ifd = exif.get_ifd(0x8825)  # GPS IFD tag
        
        if not gps_ifd:
            return None
        
        # Parse GPS tags
        gps_data = {}
        for tag_id, value in gps_ifd.items():
            tag_name = GPSTAGS.get(tag_id, tag_id)
            gps_data[tag_name] = value
        
        # Extract coordinates
        if 'GPSLatitude' not in gps_data or 'GPSLongitude' not in gps_data:
            return None
        
        # Convert to decimal degrees
        lat = _convert_to_degrees(gps_data['GPSLatitude'])
        lon = _convert_to_degrees(gps_data['GPSLongitude'])
        
        # Apply N/S and E/W directions
        if gps_data.get('GPSLatitudeRef') == 'S':
            lat = -lat
        if gps_data.get('GPSLongitudeRef') == 'W':
            lon = -lon
        
        result = {
            'latitude': lat,
            'longitude': lon
        }
        
        # Add altitude if available
        if 'GPSAltitude' in gps_data:
            altitude = float(gps_data['GPSAltitude'])
            if gps_data.get('GPSAltitudeRef') == 1:
                altitude = -altitude
            result['altitude'] = altitude
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting GPS from {image_path}: {e}")
        return None


def get_camera_info(image_path: Union[str, Path]) -> Dict[str, str]:
    """
    Get camera and photo settings information.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with camera make, model, and photo settings
    """
    exif = get_exif_data(image_path)
    
    info = {}
    
    # Camera information
    if 'Make' in exif:
        info['camera_make'] = exif['Make']
    if 'Model' in exif:
        info['camera_model'] = exif['Model']
    
    # Photo settings
    if 'ISOSpeedRatings' in exif:
        info['iso'] = exif['ISOSpeedRatings']
    if 'FNumber' in exif:
        info['aperture'] = f"f/{exif['FNumber']}"
    if 'ExposureTime' in exif:
        exp = exif['ExposureTime']
        if isinstance(exp, tuple):
            info['shutter_speed'] = f"{exp[0]}/{exp[1]}s"
        else:
            info['shutter_speed'] = f"{exp}s"
    if 'FocalLength' in exif:
        focal = exif['FocalLength']
        if isinstance(focal, tuple):
            info['focal_length'] = f"{focal[0]/focal[1]}mm"
        else:
            info['focal_length'] = f"{focal}mm"
    
    return info


def get_image_dimensions(image_path: Union[str, Path]) -> Tuple[int, int]:
    """
    Get image width and height.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (width, height)
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        logger.error(f"Error getting dimensions from {image_path}: {e}")
        return (0, 0)


def get_image_orientation(image_path: Union[str, Path]) -> str:
    """
    Get image orientation (portrait, landscape, or square).
    
    Args:
        image_path: Path to image file
        
    Returns:
        'portrait', 'landscape', or 'square'
    """
    width, height = get_image_dimensions(image_path)
    
    if width > height:
        return 'landscape'
    elif height > width:
        return 'portrait'
    else:
        return 'square'


def get_complete_metadata(image_path: Union[str, Path]) -> Dict:
    """
    Get all available metadata from an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with all metadata including datetime, GPS, camera info
        
    Example:
        >>> metadata = get_complete_metadata("photo.jpg")
        >>> print(f"Taken at: {metadata['datetime']}")
        >>> print(f"Location: {metadata['gps']}")
        >>> print(f"Camera: {metadata['camera']['camera_make']}")
    """
    metadata = {
        'file_path': str(image_path),
        'file_name': Path(image_path).name,
    }
    
    # Get datetime
    dt = get_datetime(image_path)
    if dt:
        metadata['datetime'] = dt
        metadata['datetime_str'] = dt.strftime("%Y-%m-%d %H:%M:%S")
        metadata['date'] = dt.date()
        metadata['time'] = dt.time()
    
    # Get GPS coordinates
    gps = get_gps_coordinates(image_path)
    if gps:
        metadata['gps'] = gps
        metadata['latitude'] = gps['latitude']
        metadata['longitude'] = gps['longitude']
        if 'altitude' in gps:
            metadata['altitude'] = gps['altitude']
    
    # Get camera info
    camera = get_camera_info(image_path)
    if camera:
        metadata['camera'] = camera
    
    # Get dimensions
    width, height = get_image_dimensions(image_path)
    metadata['width'] = width
    metadata['height'] = height
    metadata['orientation'] = get_image_orientation(image_path)
    
    return metadata


def format_gps_for_maps(latitude: float, longitude: float) -> Dict[str, str]:
    """
    Format GPS coordinates for use with mapping services.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        
    Returns:
        Dictionary with formatted URLs for various map services
    """
    return {
        'google_maps': f"https://www.google.com/maps?q={latitude},{longitude}",
        'apple_maps': f"http://maps.apple.com/?q={latitude},{longitude}",
        'openstreetmap': f"https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}&zoom=15",
        'coordinates': f"{latitude:.6f}, {longitude:.6f}"
    }


def has_gps_data(image_path: Union[str, Path]) -> bool:
    """
    Check if image has GPS data.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if GPS data exists, False otherwise
    """
    return get_gps_coordinates(image_path) is not None


def has_datetime(image_path: Union[str, Path]) -> bool:
    """
    Check if image has datetime information.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if datetime exists, False otherwise
    """
    return get_datetime(image_path) is not None



#!/usr/bin/env python3
"""
Example: Extract EXIF Metadata from Photos

This example demonstrates how to:
1. Get datetime (when photo was taken)
2. Get GPS coordinates (where photo was taken)
3. Get camera information
4. Get all metadata at once

Usage:
    python exif_metadata_example.py                    # Uses default from config
    python exif_metadata_example.py /path/to/photo.jpg # Uses command-line argument
"""

import sys
from pathlib import Path
from travel_log import (
    config,
    get_datetime,
    get_gps_coordinates,
    get_camera_info,
    get_complete_metadata,
    format_gps_for_maps,
    has_gps_data,
    has_datetime
)

def main():
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use default from config
        image_path = config.get('images', {}).get('default_test_image', 'your_photo.jpg')
        print(f"ℹ️  Using default image from config: {image_path}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        print("\n💡 Usage:")
        print("   python exif_metadata_example.py /path/to/your/photo.jpg")
        return
    
    print("=" * 70)
    print(f"EXIF Metadata Extraction")
    print("=" * 70)
    print(f"Image: {Path(image_path).name}")
    print()
    
    # Check what metadata is available
    print("📋 Metadata Availability:")
    print(f"   Datetime: {'✅ YES' if has_datetime(image_path) else '❌ NO'}")
    print(f"   GPS:      {'✅ YES' if has_gps_data(image_path) else '❌ NO'}")
    print()
    
    # Get datetime
    print("=" * 70)
    print("📅 DATE AND TIME")
    print("=" * 70)
    dt = get_datetime(image_path)
    if dt:
        print(f"   Photo taken: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Date: {dt.date()}")
        print(f"   Time: {dt.time()}")
        print(f"   Day of week: {dt.strftime('%A')}")
    else:
        print("   ℹ️  No datetime information in EXIF")
    print()
    
    # Get GPS coordinates
    print("=" * 70)
    print("📍 LOCATION (GPS)")
    print("=" * 70)
    gps = get_gps_coordinates(image_path)
    if gps:
        print(f"   Latitude:  {gps['latitude']:.6f}°")
        print(f"   Longitude: {gps['longitude']:.6f}°")
        if 'altitude' in gps:
            print(f"   Altitude:  {gps['altitude']:.1f}m")
        
        print()
        print("   📱 View on Maps:")
        maps = format_gps_for_maps(gps['latitude'], gps['longitude'])
        print(f"   Google Maps: {maps['google_maps']}")
        print(f"   Apple Maps:  {maps['apple_maps']}")
        print(f"   Coordinates: {maps['coordinates']}")
    else:
        print("   ℹ️  No GPS information in EXIF")
        print("   (Location services may have been off)")
    print()
    
    # Get camera info
    print("=" * 70)
    print("📷 CAMERA INFORMATION")
    print("=" * 70)
    camera = get_camera_info(image_path)
    if camera:
        if 'camera_make' in camera:
            print(f"   Make:  {camera['camera_make']}")
        if 'camera_model' in camera:
            print(f"   Model: {camera['camera_model']}")
        
        if any(k in camera for k in ['iso', 'aperture', 'shutter_speed', 'focal_length']):
            print()
            print("   Photo Settings:")
            if 'iso' in camera:
                print(f"   ISO:           {camera['iso']}")
            if 'aperture' in camera:
                print(f"   Aperture:      {camera['aperture']}")
            if 'shutter_speed' in camera:
                print(f"   Shutter Speed: {camera['shutter_speed']}")
            if 'focal_length' in camera:
                print(f"   Focal Length:  {camera['focal_length']}")
    else:
        print("   ℹ️  No camera information in EXIF")
    print()
    
    # Get complete metadata
    print("=" * 70)
    print("📊 COMPLETE METADATA")
    print("=" * 70)
    metadata = get_complete_metadata(image_path)
    
    print(f"   File: {metadata['file_name']}")
    
    if 'width' in metadata and 'height' in metadata:
        print(f"   Dimensions: {metadata['width']} x {metadata['height']} pixels")
        print(f"   Orientation: {metadata['orientation']}")
    
    if 'datetime_str' in metadata:
        print(f"   Taken: {metadata['datetime_str']}")
    
    if 'latitude' in metadata and 'longitude' in metadata:
        print(f"   Location: {metadata['latitude']:.6f}, {metadata['longitude']:.6f}")
    
    if 'camera' in metadata and 'camera_make' in metadata['camera']:
        print(f"   Camera: {metadata['camera'].get('camera_make', '')} {metadata['camera'].get('camera_model', '')}")
    
    print()
    print("=" * 70)
    print("✅ Extraction Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()



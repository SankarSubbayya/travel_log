#!/usr/bin/env python3
"""
Test HEIC Image Support

This script demonstrates HEIC/HEIF image conversion and face detection.

Usage:
    python test_heic.py /path/to/image.heic
"""

import sys
from pathlib import Path
from travel_log import (
    HEIC_SUPPORTED,
    convert_heic_to_jpg,
    is_heic_file,
    ensure_compatible_image,
    get_supported_image_extensions
)

def main():
    print("=" * 70)
    print("HEIC Image Support Test")
    print("=" * 70)
    print()
    
    # Check HEIC support
    print(f"HEIC Support Available: {'✅ YES' if HEIC_SUPPORTED else '❌ NO'}")
    
    if not HEIC_SUPPORTED:
        print("\n⚠️  HEIC support not available.")
        print("Install with: uv add pillow-heif")
        return
    
    print()
    print("Supported image extensions:")
    extensions = get_supported_image_extensions()
    print(f"  {', '.join(extensions)}")
    print()
    
    # Check if image path provided
    if len(sys.argv) < 2:
        print("💡 Usage: python test_heic.py /path/to/image.heic")
        print()
        print("Example:")
        print("  python test_heic.py ~/Photos/IMG_1234.HEIC")
        return
    
    image_path = sys.argv[1]
    print(f"Testing with: {image_path}")
    print()
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        return
    
    # Check if HEIC
    if is_heic_file(image_path):
        print("✅ HEIC file detected")
        
        # Convert to JPEG
        print("🔄 Converting HEIC to JPEG...")
        try:
            jpg_path = convert_heic_to_jpg(image_path, f"{Path(image_path).stem}_converted.jpg")
            print(f"✅ Converted successfully to: {jpg_path}")
            
            # Check file size
            original_size = Path(image_path).stat().st_size / (1024 * 1024)
            converted_size = Path(jpg_path).stat().st_size / (1024 * 1024)
            print(f"   Original: {original_size:.2f} MB")
            print(f"   Converted: {converted_size:.2f} MB")
            
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return
    else:
        print("ℹ️  Not a HEIC file (will use as-is)")
    
    # Test ensure_compatible_image
    print()
    print("🔄 Testing ensure_compatible_image()...")
    try:
        compatible_path = ensure_compatible_image(image_path)
        print(f"✅ Compatible image path: {compatible_path}")
        
        if compatible_path != str(image_path):
            print("   (HEIC was converted to temporary JPEG)")
        else:
            print("   (Original format already compatible)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print()
    print("=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)
    print()
    print("Now you can use HEIC files with:")
    print("  • python examples/face_detection_example.py your_photo.heic")
    print("  • python demo_face_detection.py your_photo.heic")
    print("  • Streamlit app: ./run_app.sh (upload HEIC files)")
    print()

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Debug: Show the exact structure of photos in Qdrant.
"""

from travel_log.qdrant_store import create_qdrant_store
import json

def main():
    print("=== Photo Data Structure in Qdrant ===\n")

    # Connect to Qdrant
    try:
        store = create_qdrant_store(url="http://sapphire:6333")
        print("✓ Connected to Qdrant\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        return

    # Get all photos
    photos = store.get_all_photos(limit=10)
    print(f"Found {len(photos)} photos\n")

    if not photos:
        print("No photos found!")
        return

    # Show structure of first photo
    print("=" * 80)
    print("FIRST PHOTO STRUCTURE:")
    print("=" * 80)
    photo = photos[0]

    print(json.dumps(photo, indent=2, default=str))

    print("\n" + "=" * 80)
    print("KEYS PRESENT IN PHOTO:")
    print("=" * 80)
    for key in sorted(photo.keys()):
        value = photo[key]
        value_type = type(value).__name__
        print(f"  {key}: {value_type} = {value}")

    print("\n" + "=" * 80)
    print("GPS CHECK:")
    print("=" * 80)

    for i, photo in enumerate(photos, 1):
        filename = photo.get('filename', 'Unknown')
        print(f"\nPhoto {i}: {filename}")

        # Check all possible GPS field names
        gps_fields = [
            'latitude', 'longitude',
            'lat', 'lon',
            'gps_latitude', 'gps_longitude',
            'location', 'coordinates',
            'exif_metadata'
        ]

        found_gps = False
        for field in gps_fields:
            if field in photo:
                print(f"  ✓ {field}: {photo[field]}")
                found_gps = True

        # Check if GPS is nested in exif_metadata
        if 'exif_metadata' in photo and photo['exif_metadata']:
            exif = photo['exif_metadata']
            print(f"  EXIF metadata type: {type(exif)}")
            if isinstance(exif, dict):
                if 'latitude' in exif:
                    print(f"  ✓ exif_metadata.latitude: {exif['latitude']}")
                    found_gps = True
                if 'longitude' in exif:
                    print(f"  ✓ exif_metadata.longitude: {exif['longitude']}")
                    found_gps = True

        if not found_gps:
            print(f"  ✗ No GPS data found in any field")

if __name__ == "__main__":
    main()

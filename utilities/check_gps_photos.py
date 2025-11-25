#!/usr/bin/env python3
"""
Check which photos in Qdrant have GPS data.
"""

from travel_log.qdrant_store import create_qdrant_store

def main():
    print("=== Checking Photos for GPS Data ===\n")

    # Connect to Qdrant
    try:
        store = create_qdrant_store(url="http://sapphire:6333")
        print("✓ Connected to Qdrant\n")
    except Exception as e:
        print(f"✗ Error connecting to Qdrant: {e}")
        return

    # Get all photos
    try:
        photos = store.get_all_photos(limit=100)
        print(f"Total photos in database: {len(photos)}\n")

        if not photos:
            print("❌ No photos found in database!")
            print("\nTo add photos:")
            print("1. Start the app: ./run_app.sh")
            print("2. Go to 'Face Detection' tab")
            print("3. Upload a photo")
            print("4. Go to 'Qdrant Storage' tab")
            print("5. Click '💾 Save to Qdrant'")
            return

        # Check which have GPS
        photos_with_gps = []
        photos_without_gps = []

        for photo in photos:
            has_gps = 'latitude' in photo and 'longitude' in photo
            if has_gps and photo['latitude'] and photo['longitude']:
                photos_with_gps.append(photo)
            else:
                photos_without_gps.append(photo)

        print(f"Photos WITH GPS: {len(photos_with_gps)}")
        print(f"Photos WITHOUT GPS: {len(photos_without_gps)}\n")

        if photos_with_gps:
            print("✓ Photos with GPS coordinates:")
            for photo in photos_with_gps:
                filename = photo.get('filename', 'Unknown')
                lat = photo.get('latitude', 0)
                lon = photo.get('longitude', 0)
                datetime = photo.get('datetime', 'Unknown')
                print(f"  - {filename}")
                print(f"    GPS: {lat:.6f}, {lon:.6f}")
                print(f"    Date: {datetime}")
                print()
        else:
            print("❌ No photos with GPS data found!\n")
            print("Your photos don't have GPS coordinates. You have 2 options:\n")
            print("Option 1: Add GPS manually in the app")
            print("  1. Go to 'Qdrant Storage' tab")
            print("  2. Expand '📍 Add Location Manually'")
            print("  3. Enter latitude and longitude")
            print("  4. Click '✅ Add Location to Photo'")
            print("  5. Save to Qdrant\n")
            print("Option 2: Upload photos with GPS EXIF data")
            print("  - Take photos with your phone (GPS enabled)")
            print("  - Or use photos from a GPS-enabled camera")
            print("  - Check EXIF data: exiftool photo.jpg | grep GPS\n")

        if photos_without_gps:
            print("Photos WITHOUT GPS:")
            for photo in photos_without_gps:
                filename = photo.get('filename', 'Unknown')
                datetime = photo.get('datetime', 'Unknown')
                print(f"  - {filename} (Date: {datetime})")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

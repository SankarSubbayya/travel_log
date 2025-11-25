#!/usr/bin/env python3
"""
Check raw Qdrant data to see GPS coordinates.
"""

from qdrant_client import QdrantClient

def main():
    print("=== Raw Qdrant Data Check ===\n")

    client = QdrantClient(url="http://sapphire:6333")

    # Get all points
    results = client.scroll(
        collection_name="travel_photos",
        limit=10,
        with_payload=True,
        with_vectors=False
    )

    photos = results[0]
    print(f"Found {len(photos)} photos\n")

    for i, point in enumerate(photos, 1):
        print(f"Photo {i}:")
        print(f"  ID: {point.id}")
        print(f"  Filename: {point.payload.get('filename', 'Unknown')}")

        lat = point.payload.get('latitude')
        lon = point.payload.get('longitude')

        if lat is not None and lon is not None:
            print(f"  ✓ GPS: {lat}, {lon}")
        elif lat is None and lon is None:
            print(f"  ✗ GPS: Both None (not set)")
        else:
            print(f"  ⚠ GPS: lat={lat}, lon={lon} (partial)")

        print()

if __name__ == "__main__":
    main()

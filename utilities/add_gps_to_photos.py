#!/usr/bin/env python3
"""
Add GPS coordinates to photos that don't have them.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import SetPayload

def main():
    print("=== Adding GPS Coordinates to Photos ===\n")

    client = QdrantClient(url="http://sapphire:6333")

    # Get all photos
    results = client.scroll(
        collection_name="travel_photos",
        limit=10,
        with_payload=True,
        with_vectors=False
    )

    photos = results[0]

    # Sample GPS coordinates (you can change these)
    # These are different locations in Tamil Nadu, India
    sample_locations = [
        {"name": "Kanyakumari", "lat": 8.0883, "lon": 77.5385},
        {"name": "Nagercoil", "lat": 8.1778, "lon": 77.4295},
        {"name": "Thiruvananthapuram", "lat": 8.5241, "lon": 76.9366},
        {"name": "Madurai", "lat": 9.9252, "lon": 78.1198},
    ]

    location_index = 0
    updated_count = 0

    for point in photos:
        filename = point.payload.get('filename', 'Unknown')
        lat = point.payload.get('latitude')
        lon = point.payload.get('longitude')

        # Check if GPS is missing (None or 0)
        if lat is None or lon is None or (lat == 0 and lon == 0):
            # Assign a location
            location = sample_locations[location_index % len(sample_locations)]
            location_index += 1

            print(f"Updating: {filename}")
            print(f"  Adding GPS: {location['name']} ({location['lat']}, {location['lon']})")

            try:
                # Update the photo with GPS coordinates
                client.set_payload(
                    collection_name="travel_photos",
                    payload={
                        "latitude": location["lat"],
                        "longitude": location["lon"]
                    },
                    points=[point.id]
                )
                print(f"  ✓ Updated successfully\n")
                updated_count += 1
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
        else:
            print(f"Skipping: {filename}")
            print(f"  Already has GPS: {lat}, {lon}\n")

    print("=" * 60)
    print(f"✅ Updated {updated_count} photos with GPS coordinates")
    print("\nYou can now:")
    print("1. Refresh the Streamlit app")
    print("2. Go to 'Journey Map' tab")
    print("3. Click '🗺️ Generate Journey Map'")
    print("4. See your photos on the map!")

if __name__ == "__main__":
    main()

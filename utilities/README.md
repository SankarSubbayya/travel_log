# Utilities

Utility scripts for Travel Log development and debugging.

## Scripts

### GPS Management

- **`add_gps_to_photos.py`** - Add test GPS coordinates to photos in Qdrant
  ```bash
  uv run python utilities/add_gps_to_photos.py
  ```

- **`check_gps_photos.py`** - Check which photos have GPS data
  ```bash
  uv run python utilities/check_gps_photos.py
  ```

- **`check_qdrant_raw.py`** - View raw GPS data from Qdrant
  ```bash
  uv run python utilities/check_qdrant_raw.py
  ```

### Debugging

- **`debug_photo_structure.py`** - Show detailed photo data structure in Qdrant
  ```bash
  uv run python utilities/debug_photo_structure.py
  ```

## Usage

These scripts are for development and debugging purposes. They help verify:
- GPS coordinates are properly stored
- Data structure in Qdrant
- Photo metadata completeness

## Note

These scripts require:
- Qdrant running on `sapphire:6333`
- Photos already stored in the `travel_photos` collection

# HEIC Image Format Support

## Overview

Travel Log now fully supports Apple's HEIC (High Efficiency Image Container) format, also known as HEIF (High Efficiency Image Format). This is the default photo format for modern iPhones and iPads.

## What is HEIC?

- **Format**: Modern image format from Apple (iOS 11+)
- **Advantages**: Better compression than JPEG (smaller file sizes)
- **Quality**: Same or better quality than JPEG
- **Usage**: Default format for iPhone/iPad photos

## How It Works

Travel Log automatically handles HEIC files:

1. **Detection**: Checks if uploaded image is HEIC/HEIF
2. **Conversion**: Automatically converts to JPEG
3. **Processing**: Face detection works on converted image
4. **Transparent**: You don't need to do anything special!

## Installation

HEIC support is included by default. The `pillow-heif` package is automatically installed:

```bash
uv sync
```

## Usage

### Streamlit Web App

1. Launch the app:
   ```bash
   ./run_app.sh
   ```

2. Upload HEIC files directly:
   - Click "Browse files" or drag and drop
   - Select `.heic` or `.heif` files
   - App will automatically convert and process

3. The app shows "Converting HEIC image..." during conversion

### Command-Line Examples

**Face Detection Example:**
```bash
# Works with HEIC files automatically
uv run python examples/face_detection_example.py ~/Photos/IMG_1234.HEIC

# Or with JPEG
uv run python examples/face_detection_example.py ~/Photos/photo.jpg
```

**Demo Script:**
```bash
uv run python demo_face_detection.py ~/Photos/IMG_1234.HEIC
```

**Test HEIC Support:**
```bash
uv run python test_heic.py ~/Photos/IMG_1234.HEIC
```

### Python API

```python
from travel_log import convert_heic_to_jpg, is_heic_file, ensure_compatible_image

# Check if file is HEIC
if is_heic_file("photo.heic"):
    print("This is a HEIC file")

# Convert HEIC to JPEG
jpg_path = convert_heic_to_jpg("photo.heic", "output.jpg")

# Automatic handling (converts only if needed)
compatible_path = ensure_compatible_image("photo.heic")  # Returns JPEG path
compatible_path = ensure_compatible_image("photo.jpg")   # Returns original path

# Use with face detection
from travel_log import FaceDetector

detector = FaceDetector()
image_path = ensure_compatible_image("photo.heic")
faces = detector.extract_faces(image_path)
```

## Supported Formats

With HEIC support installed, all these formats work:

| Format | Extensions | Status |
|--------|-----------|---------|
| JPEG | .jpg, .jpeg | ✅ Supported |
| PNG | .png | ✅ Supported |
| HEIC | .heic, .heif | ✅ Supported (auto-converts) |
| BMP | .bmp | ✅ Supported |
| GIF | .gif | ✅ Supported |
| TIFF | .tiff, .tif | ✅ Supported |

## Performance Notes

### Conversion Speed

- **Small images** (<2MB): ~0.1-0.5 seconds
- **Medium images** (2-5MB): ~0.5-1 second
- **Large images** (5-10MB): ~1-2 seconds

### File Size Comparison

Example: iPhone 14 Pro photo

| Format | Size | Quality |
|--------|------|---------|
| HEIC (original) | 2.3 MB | High |
| JPEG (converted) | 3.1 MB | High (95%) |

*HEIC is smaller, but JPEG is more compatible*

### Temporary Files

- Converted JPEG files are created in system temp directory
- Automatically cleaned up after processing
- No manual cleanup needed

## Troubleshooting

### "HEIC support not available"

**Error:**
```
ValueError: HEIC file detected but support not available
```

**Solution:**
```bash
uv sync
# Or manually:
uv add pillow-heif
```

### "Can't open HEIC file"

**Possible causes:**
- Corrupted HEIC file
- Unsupported HEIC variant
- Permission issues

**Solutions:**
1. Try opening file in Photos app first
2. Re-export from Photos as HEIC
3. Check file permissions

### Conversion Fails

**If conversion fails:**
1. Check file is valid HEIC (open in Photos app)
2. Try converting manually:
   ```bash
   uv run python test_heic.py your_photo.heic
   ```
3. As fallback, export as JPEG from Photos app

## Configuration

### Check HEIC Support

```python
from travel_log import HEIC_SUPPORTED

if HEIC_SUPPORTED:
    print("HEIC support is available!")
else:
    print("Install pillow-heif for HEIC support")
```

### Get Supported Extensions

```python
from travel_log import get_supported_image_extensions

extensions = get_supported_image_extensions()
print(f"Supported: {extensions}")
# Output: ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.heic', '.heif']
```

## Examples

### Example 1: Process iPhone Photos

```bash
# iPhone photo (HEIC)
uv run python examples/face_detection_example.py ~/Photos/IMG_1234.HEIC

# Extracted faces saved to: extracted_faces/
```

### Example 2: Batch Process Mixed Formats

```python
from pathlib import Path
from travel_log import FaceDetector, ensure_compatible_image

detector = FaceDetector()
photos_dir = Path("~/Photos").expanduser()

for photo in photos_dir.glob("*"):
    if photo.suffix.lower() in ['.jpg', '.jpeg', '.heic', '.heif', '.png']:
        # Automatically handles HEIC conversion
        compatible_path = ensure_compatible_image(photo)
        faces = detector.extract_faces(compatible_path)
        print(f"{photo.name}: Found {len(faces)} faces")
```

### Example 3: Streamlit App with HEIC

The Streamlit app automatically handles HEIC files:

```bash
./run_app.sh

# Then:
# 1. Upload HEIC file from your iPhone
# 2. App converts automatically
# 3. Detect faces
# 4. Download extracted faces as JPEG
```

## Technical Details

### Conversion Process

1. **Read HEIC**: Using `pillow-heif` plugin
2. **Convert Color**: Ensure RGB color space
3. **Save JPEG**: High quality (95%) compression
4. **Return Path**: Provide path to converted file

### Dependencies

```toml
[dependencies]
pillow-heif = ">=0.13.0"  # HEIC/HEIF support
pillow = ">=10.0.0"       # Base image library
```

### Implementation

See `src/travel_log/image_utils.py` for:
- `convert_heic_to_jpg()` - Convert HEIC to JPEG
- `is_heic_file()` - Check if file is HEIC
- `ensure_compatible_image()` - Auto-convert if needed
- `get_supported_image_extensions()` - List supported formats

## Best Practices

### 1. Let It Convert Automatically

```python
# Good - automatic handling
from travel_log import FaceDetector
detector = FaceDetector()
faces = detector.extract_faces("photo.heic")  # Works!

# Don't manually convert unless needed
```

### 2. Use ensure_compatible_image for Flexibility

```python
# Works with any format
compatible_path = ensure_compatible_image(image_path)
faces = detector.extract_faces(compatible_path)
```

### 3. Check File Before Processing

```python
from pathlib import Path
from travel_log import is_supported_image

if Path("photo.heic").exists() and is_supported_image("photo.heic"):
    # Process it
    pass
```

## FAQ

### Q: Do I need to install anything extra?

**A:** No, `pillow-heif` is included in dependencies. Just run `uv sync`.

### Q: Is the original HEIC file modified?

**A:** No, original files are never modified. Conversion creates new temporary JPEG files.

### Q: Can I save extracted faces as HEIC?

**A:** No, extracted faces are saved as JPEG for maximum compatibility.

### Q: Does it work on Windows/Linux?

**A:** Yes! `pillow-heif` works on all platforms.

### Q: What about HEIC videos?

**A:** Only HEIC images are supported. HEIC videos (Live Photos) use the first frame.

### Q: Is quality lost during conversion?

**A:** Minimal loss. Conversion uses 95% quality JPEG, which is visually identical to original.

## Summary

✅ **HEIC support is built-in**  
✅ **Automatic conversion** - no manual steps needed  
✅ **Works everywhere** - CLI, Python API, Streamlit app  
✅ **High quality** - 95% JPEG quality maintained  
✅ **Cross-platform** - Windows, macOS, Linux  
✅ **Efficient** - Fast conversion, temporary files auto-cleaned  

Just use HEIC files like any other image format! 📱→📸


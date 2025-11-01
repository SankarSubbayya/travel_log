# DeepFace Model Downloads Guide

## Overview

DeepFace automatically downloads machine learning models to `~/.deepface/weights/` on first use. This is normal behavior and only happens once per model.

## Download Location

**Default path:**
```
~/.deepface/weights/
```

On your system:
```
/home/sankar/.deepface/weights/
```

## What Gets Downloaded

DeepFace downloads models **only when you first use them**:

### Detection Models

| Model | Size | When Downloaded |
|-------|------|-----------------|
| opencv | Built-in | Never (included with OpenCV) |
| ssd | ~2 MB | First use of `detector_backend='ssd'` |
| mtcnn | ~2 MB | First use of `detector_backend='mtcnn'` |
| retinaface | ~1 MB | First use of `detector_backend='retinaface'` |
| dlib | ~99 MB | First use of `detector_backend='dlib'` |

### Recognition Models

| Model | Size | When Downloaded |
|-------|------|-----------------|
| VGG-Face | ~500 MB | First use of `model_name='VGG-Face'` |
| Facenet | ~90 MB | First use of `model_name='Facenet'` |
| Facenet512 | ~90 MB | First use of `model_name='Facenet512'` |
| OpenFace | ~30 MB | First use of `model_name='OpenFace'` |
| DeepFace | ~150 MB | First use of `model_name='DeepFace'` |
| ArcFace | ~160 MB | First use of `model_name='ArcFace'` |

## Check Downloaded Models

### Using the Check Script

```bash
uv run python check_deepface_models.py
```

This shows:
- ✅ Which models are downloaded
- 📦 Model file sizes
- 📂 Storage location
- 💾 Total disk space used

### Manually Check Directory

```bash
# List all downloaded models
ls -lh ~/.deepface/weights/

# See total size
du -sh ~/.deepface/weights/

# Count model files
ls ~/.deepface/weights/ | wc -l
```

### In Python

```python
from pathlib import Path

deepface_dir = Path.home() / ".deepface" / "weights"

if deepface_dir.exists():
    models = list(deepface_dir.glob("*.h5"))
    print(f"Downloaded: {len(models)} models")
    for model in models:
        size = model.stat().st_size / (1024 * 1024)
        print(f"  {model.name}: {size:.1f} MB")
else:
    print("No models downloaded yet")
```

## Storage Requirements

### Minimal Setup (Fast & Small)
```
opencv detector + Facenet = ~90 MB
```

### Recommended Setup (Balanced)
```
mtcnn detector + Facenet512 = ~92 MB
```

### Maximum Setup (All Models)
```
All detectors + All recognition models = ~1.5 GB
```

## How Model Downloads Work

1. **DeepFace checks** `~/.deepface/weights/`
2. **If model not found**, downloads from GitHub/Google Drive
3. **Saves to** `~/.deepface/weights/`
4. **Uses cached version** on all subsequent calls

## Troubleshooting

### Models downloading every time

**Cause:** Download failing or permissions issue

**Solution:**
```bash
# Check permissions
ls -ld ~/.deepface/weights/

# Fix permissions if needed
chmod -R 755 ~/.deepface/

# Manually create directory
mkdir -p ~/.deepface/weights/
```

### Download fails

**Causes:**
- No internet connection
- Firewall blocking downloads
- GitHub/Drive rate limits

**Solutions:**
```bash
# Check internet connection
ping github.com

# Try again later
# Use VPN if network blocks downloads
```

### Taking up too much space

**Solution:**
```bash
# See what's using space
du -sh ~/.deepface/weights/*

# Remove specific models (they'll re-download if needed)
rm ~/.deepface/weights/vgg_face_weights.h5

# Or remove all models
rm -rf ~/.deepface/weights/
```

## Quick Commands

```bash
# Check downloads
ls -lh ~/.deepface/weights/

# See total size
du -sh ~/.deepface/weights/

# Verify with check script
uv run python check_deepface_models.py

# Test face detection (will download models if needed)
uv run python demo_face_detection.py your_photo.jpg
```

## Summary

- **Models download automatically** on first use
- **Stored at:** `~/.deepface/weights/`
- **Download once, use forever** (cached locally)
- **Check status:** `uv run python check_deepface_models.py`
- **Typical size:** 90-500 MB depending on models used

No manual intervention needed - just use the face detection and models will download automatically! 🎉

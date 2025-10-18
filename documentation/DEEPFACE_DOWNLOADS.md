# DeepFace Model Downloads - Complete Guide

## Your Question

You ran:
```bash
nohup uv run python -c 'from deepface import DeepFace' &
```

**Yes! DeepFace downloads models to `~/.deepface/weights/`**

## Where DeepFace Stores Models

### Default Location
```
~/.deepface/weights/
```

Full path on your system:
```
/Users/sankar/.deepface/weights/
```

### What Gets Downloaded

DeepFace downloads models **only when you first use them**:

| Model Type | Model Name | Size | When Downloaded |
|------------|------------|------|-----------------|
| **Detection** | opencv | Built-in | Never (already in OpenCV) |
| | ssd | ~2 MB | First use of `detector_backend='ssd'` |
| | mtcnn | ~2 MB | First use of `detector_backend='mtcnn'` |
| | retinaface | ~1 MB | First use of `detector_backend='retinaface'` |
| | dlib | ~99 MB | First use of `detector_backend='dlib'` |
| **Recognition** | VGG-Face | ~500 MB | First use of `model_name='VGG-Face'` |
| | Facenet | ~90 MB | First use of `model_name='Facenet'` |
| | Facenet512 | ~90 MB | First use of `model_name='Facenet512'` |
| | OpenFace | ~30 MB | First use of `model_name='OpenFace'` |
| | DeepFace | ~150 MB | First use of `model_name='DeepFace'` |
| | ArcFace | ~160 MB | First use of `model_name='ArcFace'` |

## Check If Models Are Downloaded

### Method 1: Run the Check Script

```bash
python check_deepface_models.py
```

This shows:
- ✅ Which models are downloaded
- 📦 Model file sizes
- 📂 Storage location
- 💾 Total disk space used

### Method 2: Check Directory Manually

```bash
# List all downloaded models
ls -lh ~/.deepface/weights/

# See total size
du -sh ~/.deepface/weights/

# Count model files
ls ~/.deepface/weights/ | wc -l
```

### Method 3: In Python

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

## About Your nohup Command

### What It Does

```bash
nohup uv run python -c 'from deepface import DeepFace' &
```

- `nohup`: Run in background, doesn't stop if you close terminal
- `&`: Run as background job
- `from deepface import DeepFace`: Just imports DeepFace (doesn't download models yet!)

### Important: Import ≠ Download

**Importing DeepFace doesn't download models!** Models download when you:
- Call `DeepFace.extract_faces()` → downloads detector
- Call `DeepFace.represent()` → downloads recognition model
- Call `DeepFace.find()` → downloads both

### Check Your Background Process

```bash
# Check if still running
ps aux | grep deepface

# Check nohup output
cat nohup.out

# Check for running Python processes
ps aux | grep python
```

## Pre-Download All Models

If you want to download all models upfront (so there's no delay later):

### Method 1: Use the Pre-download Script

```bash
# Run in background
nohup python pre_download_models.py &

# Check progress
tail -f nohup.out

# Or run in foreground (see progress)
python pre_download_models.py
```

### Method 2: Download Specific Models

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deepface import DeepFace

# Download detector (e.g., mtcnn)
DeepFace.extract_faces("dummy.jpg", detector_backend='mtcnn', enforce_detection=False)

# Download recognition model (e.g., Facenet512)
DeepFace.represent("dummy.jpg", model_name='Facenet512', enforce_detection=False)
```

### Method 3: Download in Background (Correct Way)

```bash
# Download in background with progress output
nohup python pre_download_models.py > download.log 2>&1 &

# Monitor progress
tail -f download.log

# Check when done
ps aux | grep pre_download
```

## Why Pre-Download?

### Benefits

1. **No delays** during actual face detection
2. **Offline use** after download
3. **Predictable performance** - no waiting for downloads
4. **Better user experience** - fast from the start

### When to Pre-Download

- ✅ Before processing large batches of photos
- ✅ Before demos or presentations
- ✅ On a fast internet connection
- ✅ When you have time to wait

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

## Model Download Process

### What Happens When You First Use a Model

1. **DeepFace checks** `~/.deepface/weights/`
2. **If model not found**, downloads from:
   - GitHub releases
   - Google Drive (for some models)
   - Dropbox (for some models)
3. **Saves to** `~/.deepface/weights/`
4. **Uses cached version** on subsequent calls

### Download Times (Approximate)

| Connection | Small Model (90MB) | Large Model (500MB) |
|------------|-------------------|---------------------|
| Fast (100 Mbps) | ~10 seconds | ~45 seconds |
| Medium (25 Mbps) | ~40 seconds | ~3 minutes |
| Slow (5 Mbps) | ~3 minutes | ~15 minutes |

## Troubleshooting

### "Models downloading every time"

**Cause**: Download failing or permissions issue

**Solution**:
```bash
# Check permissions
ls -ld ~/.deepface/weights/

# Fix permissions if needed
chmod -R 755 ~/.deepface/

# Manually create directory
mkdir -p ~/.deepface/weights/
```

### "Download fails"

**Causes**:
- No internet connection
- Firewall blocking
- GitHub/Drive rate limits

**Solutions**:
```bash
# Try again later
# Use VPN if blocked
# Check internet: ping github.com
```

### "Taking up too much space"

**Solution**:
```bash
# See what's using space
du -sh ~/.deepface/weights/*

# Remove unused models
rm ~/.deepface/weights/vgg_face_weights.h5  # Example

# Or remove all (they'll re-download)
rm -rf ~/.deepface/weights/
```

## Checking Download Status

### Real-time Progress

```bash
# While downloading, watch the directory
watch -n 1 'ls -lh ~/.deepface/weights/ && du -sh ~/.deepface/weights/'

# Or monitor nohup output
tail -f nohup.out
```

### Verify Download Complete

```bash
# Check if process finished
jobs  # Shows background jobs
ps aux | grep python | grep deepface

# Check nohup output
tail -20 nohup.out
```

## Summary

### Your Setup

1. **Command you ran**: `nohup uv run python -c 'from deepface import DeepFace' &`
   - ✅ Imports DeepFace
   - ⚠️ Doesn't download models yet (models download on first use)

2. **Where models go**: `~/.deepface/weights/`

3. **To check**: `python check_deepface_models.py`

4. **To pre-download**: `python pre_download_models.py`

### Recommended Next Steps

```bash
# 1. Check current status
python check_deepface_models.py

# 2. Pre-download models if needed
python pre_download_models.py

# 3. Verify
ls -lh ~/.deepface/weights/

# 4. Test face detection
python demo_face_detection.py your_photo.jpg
```

### Quick Commands

```bash
# Check downloads
ls -lh ~/.deepface/weights/

# Pre-download in background
nohup python pre_download_models.py &

# Check progress
tail -f nohup.out

# Verify complete
python check_deepface_models.py
```

Now you know where everything is! 🎉


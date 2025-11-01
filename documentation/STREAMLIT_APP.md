# Streamlit Web App Guide

## Overview

The Travel Log Face Recognition Streamlit app provides an intuitive web interface for detecting and extracting faces from photos.

## Features

### 🎯 Core Features
- **Drag-and-drop image upload** - Easy file upload
- **Real-time face detection** - Instant results
- **Multiple detection backends** - Choose the best algorithm for your needs
- **Confidence filtering** - Filter out low-quality detections
- **Visual preview** - See all detected faces
- **Individual downloads** - Download each face separately
- **Bulk ZIP download** - Get all faces at once
- **Statistics** - View detection metrics

### 🔧 Configuration Options
- **Detection Backend**: opencv, ssd, mtcnn, retinaface, dlib
- **Confidence Threshold**: 0.5 - 1.0
- **GPU Acceleration**: Automatic detection

## Quick Start

### Launch the App

```bash
# Method 1: Using the helper script
./run_app.sh

# Method 2: Direct command
uv run streamlit run app.py

# Method 3: From examples directory
cd /home/sankar/travel_log
uv run streamlit run app.py
```

The app will open in your browser at: `http://localhost:8501`

### First-Time Setup

1. **Initialize Detector**
   - Open the sidebar (click `>` if collapsed)
   - Select a detection backend (default: mtcnn)
   - Click "🔄 Initialize Detector"
   - Wait for initialization (first time downloads models)

2. **Upload Image**
   - Click "Browse files" or drag and drop
   - Supported formats: JPG, JPEG, PNG

3. **Detect Faces**
   - Click "🔍 Detect Faces" button
   - Wait for processing
   - View results in the right panel

4. **Download Results**
   - Click "⬇️ Download" under each face
   - Or click "📦 Download All Faces" for ZIP

## User Interface

### Sidebar (Left)
- **Settings**
  - Detection Backend selector
  - Backend guide and descriptions
  - Confidence threshold slider
  - Initialize button
- **Statistics**
  - Number of faces detected
- **Info & Tips**

### Main Panel

#### Left Column
- **Upload Image** section
- Image preview
- "Detect Faces" button

#### Right Column
- **Detected Faces** grid
- Individual face previews
- Confidence scores
- Download buttons per face
- Bulk download option

### Expandable Sections (Bottom)
- **About Face Detection** - How it works
- **Technical Details** - System information

## Detection Backends

| Backend | Speed | Accuracy | Best For |
|---------|-------|----------|----------|
| **opencv** | ⚡⚡⚡ Fast | ⭐⭐ Good | Quick testing, real-time |
| **ssd** | ⚡⚡ Fast | ⭐⭐⭐ Good | Balanced performance |
| **mtcnn** | ⚡ Medium | ⭐⭐⭐⭐ High | Recommended default |
| **retinaface** | 🐌 Slow | ⭐⭐⭐⭐⭐ Highest | Maximum accuracy |
| **dlib** | 🐌 Slow | ⭐⭐⭐⭐ High | Alternative option |

**Recommendation:** Start with **MTCNN** for best balance.

## Confidence Threshold

The confidence threshold filters detections:

- **0.9 - 1.0** (High): Only very confident detections
- **0.7 - 0.9** (Medium): Balanced filtering
- **0.5 - 0.7** (Low): Include more potential faces

**Default:** 0.9 (recommended for most cases)

## Usage Examples

### Example 1: Quick Face Detection

1. Launch app: `./run_app.sh`
2. Click "Initialize Detector" (use default MTCNN)
3. Upload a group photo
4. Click "Detect Faces"
5. Download faces you want

### Example 2: High Accuracy Detection

1. Launch app
2. Select "retinaface" backend
3. Set confidence to 0.95
4. Click "Initialize Detector"
5. Upload image and detect

### Example 3: Fast Batch Processing

1. Launch app
2. Select "opencv" backend
3. Set confidence to 0.7
4. Initialize detector
5. Upload multiple images one by one
6. Download all faces as ZIP

## Performance

### First Run
- **Time**: 1-5 minutes
- **Reason**: Downloads ML models (~100-500MB)
- **After**: Models cached, much faster

### Subsequent Runs

| Setup | Detection Speed |
|-------|-----------------|
| GPU + MTCNN | 0.1-0.5s per face |
| CPU + MTCNN | 1-2s per face |
| GPU + RetinaFace | 0.5-1s per face |
| CPU + RetinaFace | 3-5s per face |

## Configuration

The app reads from `config.yaml`:

```yaml
face_detection:
  default_backend: mtcnn
  default_model: Facenet512

images:
  output:
    extracted_faces: extracted_faces
```

## Troubleshooting

### App Won't Start

**Problem:** `streamlit: command not found`

**Solution:**
```bash
uv sync
uv run streamlit run app.py
```

### "Initialize Detector First"

**Problem:** Tried to detect without initializing

**Solution:**
1. Click sidebar `>` to open
2. Click "🔄 Initialize Detector"
3. Wait for "✅ detector initialized"
4. Then click "Detect Faces"

### Slow Detection

**Problem:** Detection takes too long

**Solutions:**
- Use faster backend (opencv or ssd)
- Lower confidence threshold
- Use smaller images
- Check if GPU is being used

### No Faces Detected

**Problem:** 0 faces found in photo

**Reasons:**
- Photo doesn't contain faces
- Faces too small
- Poor lighting/quality
- Confidence threshold too high

**Solutions:**
- Lower confidence threshold to 0.7
- Try different backend (mtcnn or retinaface)
- Use better quality photo
- Ensure faces are clearly visible

### Memory Error

**Problem:** Out of memory during detection

**Solutions:**
- Use smaller images
- Use lighter backend (opencv)
- Close other applications
- Restart the app

### Models Keep Downloading

**Problem:** Downloads models every time

**Solution:**
```bash
# Check if models exist
ls -lh ~/.deepface/weights/

# If empty, models aren't saving properly
# Check permissions
chmod -R 755 ~/.deepface/
```

## Advanced Features

### Custom Configuration

Edit `config.yaml` before starting:

```yaml
face_detection:
  default_backend: retinaface  # Your preferred backend
  
images:
  output:
    extracted_faces: my_faces_folder
```

### Command-Line Options

```bash
# Run on different port
uv run streamlit run app.py --server.port 8502

# Run on specific address
uv run streamlit run app.py --server.address 0.0.0.0

# Disable file watcher
uv run streamlit run app.py --server.fileWatcherType none
```

### Batch Processing Workflow

For processing multiple photos:

1. Open app once
2. Initialize detector (one time)
3. Upload and process each photo
4. Download faces after each detection
5. Or download all faces at end

## Tips & Best Practices

### 1. Initialize Once
Initialize the detector once per session, not for each image.

### 2. Start with MTCNN
Best balance of speed and accuracy for most cases.

### 3. Adjust Confidence
- High confidence (0.9+): Fewer false positives
- Lower confidence (0.7+): Catch more faces

### 4. Use GPU
If you have a GPU, the app will use it automatically for faster processing.

### 5. Image Quality
Better quality photos = better detection:
- Good lighting
- Clear faces
- Not too small
- Not too blurry

### 6. Batch Downloads
Use "Download All Faces" for multiple faces instead of downloading individually.

## Keyboard Shortcuts

Streamlit built-in shortcuts:

- **`r`**: Rerun the app
- **`c`**: Clear cache
- **`?`**: Show shortcuts help

## FAQ

### Q: Can I process multiple images at once?
**A:** Currently one image at a time. Process them sequentially and use bulk ZIP download.

### Q: Where are faces saved?
**A:** Faces are not saved to disk automatically. Use download buttons to save them.

### Q: Can I add face recognition/labeling?
**A:** Current version focuses on detection. Use CLI tools for recognition features.

### Q: Does it work offline?
**A:** After first run (models downloaded), yes! No internet required.

### Q: What's the maximum image size?
**A:** No hard limit, but larger images take longer to process. Recommended: < 10MB

### Q: Can I use my webcam?
**A:** Not in current version. Upload photos from your camera roll.

## Integration with CLI Tools

The web app complements the CLI tools:

1. **Web App**: Quick face detection and extraction
2. **CLI `face_detection_example.py`**: Batch processing
3. **CLI `face_labeling_example.py`**: Face recognition
4. **CLI `complete_workflow_example.py`**: Full pipeline

### Workflow Example

```bash
# 1. Use web app for quick exploration
./run_app.sh
# Upload a few test photos, see what works

# 2. Use CLI for batch processing
uv run python examples/face_detection_example.py photo1.jpg
uv run python examples/face_detection_example.py photo2.jpg

# 3. Use CLI for face labeling
uv run python examples/face_labeling_example.py
```

## Deployment

### Local Network Access

Share with devices on your network:

```bash
uv run streamlit run app.py --server.address 0.0.0.0
```

Then access from other devices:
```
http://YOUR_IP:8501
```

### Docker Deployment (Advanced)

Create `Dockerfile`:
```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install uv
RUN uv sync
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## Support

For issues or questions:
- Check the [main documentation](README.md)
- Review [troubleshooting](#troubleshooting) section
- Bring questions to weekly check-ins with Chander and Asif

## Summary

- ✅ **Easy to use** - No coding required
- ✅ **Visual interface** - See results immediately
- ✅ **Multiple backends** - Choose what works best
- ✅ **Download options** - Individual or bulk
- ✅ **GPU accelerated** - Fast processing
- ✅ **No installation** - Works in browser

Launch the app and start detecting faces in seconds! 🎉

```bash
./run_app.sh
```


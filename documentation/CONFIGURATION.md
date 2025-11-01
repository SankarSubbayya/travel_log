# Configuration Guide

## Overview

Travel Log uses a `config.yaml` file to manage configuration settings. This allows you to avoid hardcoding paths and settings throughout the codebase.

## Configuration File Location

The main configuration file is located at:
```
/home/sankar/travel_log/config.yaml
```

## Configuration Structure

### Image Dataset Configuration

Configure paths to your photo collections and output directories:

```yaml
images:
  # Path to your personal photo collection
  personal_photos_dir: /home/sankar/personal_photos
  
  # Default test image (used by examples if no command-line arg provided)
  default_test_image: /home/sankar/personal_photos/IMG_0276_2.jpeg
  
  # Output directories (relative to project root)
  output:
    extracted_faces: extracted_faces
    annotated_images: annotated_images
    workspace: workspace
```

### Face Detection Configuration

Configure default detection and recognition models:

```yaml
face_detection:
  # Detection backend: opencv, ssd, mtcnn, retinaface, dlib
  default_backend: mtcnn
  
  # Recognition model: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, ArcFace
  default_model: Facenet512
```

## Using Configuration in Code

### In Python Scripts

```python
from travel_log import config

# Get image path
default_image = config.get('images', {}).get('default_test_image')

# Get detection backend
backend = config.get('face_detection', {}).get('default_backend', 'mtcnn')

# Get output directory
output_dir = config.get('images', {}).get('output', {}).get('extracted_faces')
```

### In Examples

The example scripts automatically read from config:

```bash
# Uses default_test_image from config.yaml
uv run python examples/face_detection_example.py

# Override with command-line argument
uv run python examples/face_detection_example.py /path/to/other/image.jpg
```

## Customizing Your Configuration

### Step 1: Copy Template (Optional)

A template file is provided at `config.yaml.template`:

```bash
# If you want to start fresh
cp config.yaml.template config.yaml
```

### Step 2: Update Paths

Edit `config.yaml` and update the paths:

```yaml
images:
  personal_photos_dir: /your/path/to/photos
  default_test_image: /your/path/to/test/image.jpg
```

### Step 3: Test Configuration

```bash
# Test that paths work
uv run python examples/face_detection_example.py
```

## Detection Backends

Choose the right backend for your needs:

| Backend | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| opencv | Fast | Good | Quick testing, real-time |
| ssd | Fast | Good | Balanced performance |
| mtcnn | Medium | High | Recommended default |
| retinaface | Slow | Highest | Maximum accuracy |
| dlib | Slow | High | Alternative to retinaface |

## Recognition Models

Choose the right model for your needs:

| Model | Size | Dimensions | Accuracy | Speed |
|-------|------|------------|----------|-------|
| OpenFace | Small | 128 | Good | Fast |
| Facenet | Medium | 128 | Good | Fast |
| Facenet512 | Medium | 512 | High | Medium |
| ArcFace | Medium | 512 | High | Medium |
| DeepFace | Large | 4096 | Highest | Slow |
| VGG-Face | Large | 4096 | Highest | Slow |

**Recommended:** `Facenet512` - best balance of speed and accuracy

## Configuration Best Practices

### 1. Use Absolute Paths

```yaml
# Good
default_test_image: /home/sankar/personal_photos/image.jpg

# Bad (relative paths may not work from all locations)
default_test_image: ../photos/image.jpg
```

### 2. Create Output Directories

The application will create output directories if they don't exist, but you can pre-create them:

```bash
mkdir -p extracted_faces annotated_images workspace
```

### 3. Keep Config Under Version Control

Add `config.yaml` to git with default/template values:

```bash
git add config.yaml
```

But consider using environment-specific configs for different machines:
- `config.yaml` - local development
- `config.prod.yaml` - production
- `config.test.yaml` - testing

### 4. Use Environment Variables (Advanced)

You can also use environment variables with a `.env` file:

```bash
# .env file
PERSONAL_PHOTOS_DIR=/home/sankar/personal_photos
DEFAULT_TEST_IMAGE=/home/sankar/personal_photos/IMG_0276_2.jpeg
```

Then in your code:
```python
import os
photos_dir = os.getenv('PERSONAL_PHOTOS_DIR', config.get('images', {}).get('personal_photos_dir'))
```

## Troubleshooting

### "Configuration file not found"

**Solution:** Make sure you're running commands from the project root where `config.yaml` exists:

```bash
cd /home/sankar/travel_log
uv run python examples/face_detection_example.py
```

### "Image file not found"

**Solution:** Update the path in `config.yaml`:

```yaml
images:
  default_test_image: /correct/path/to/your/image.jpg
```

Then verify:
```bash
ls -lh /correct/path/to/your/image.jpg
```

### Configuration Not Loading

**Check:**
1. File exists: `ls -lh config.yaml`
2. Valid YAML syntax (use a YAML validator)
3. Running from correct directory

## Example Complete Configuration

```yaml
cohort: Spring 2025

images:
  personal_photos_dir: /home/sankar/personal_photos
  default_test_image: /home/sankar/personal_photos/IMG_0276_2.jpeg
  
  output:
    extracted_faces: extracted_faces
    annotated_images: annotated_images
    workspace: workspace

face_detection:
  default_backend: mtcnn
  default_model: Facenet512
```

## Quick Commands

```bash
# View current config
cat config.yaml

# Edit config
nano config.yaml

# Test with config
uv run python examples/face_detection_example.py

# Override config with command-line arg
uv run python examples/face_detection_example.py /other/image.jpg
```

## Summary

- ✅ **Centralized configuration** in `config.yaml`
- ✅ **No hardcoded paths** in code
- ✅ **Easy to customize** for different environments
- ✅ **Template provided** at `config.yaml.template`
- ✅ **Examples use config** automatically
- ✅ **Command-line override** always available

Update `config.yaml` once and all examples work with your paths! 🎉


# Travel Log - Project Overview

A Python project for managing travel memories with advanced face recognition capabilities.

## 🚀 Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Verify installation
uv run python verify_installation.py

# 3. Process a photo
uv run python demo_face_detection.py path/to/your/photo.jpg
```

## 📁 Project Structure

```
travel_log/
├── src/travel_log/          # Core modules
│   ├── __init__.py
│   ├── face_detector.py     # Face detection & extraction
│   ├── face_embeddings.py   # Face embeddings generation
│   ├── face_labeler.py      # Face recognition & labeling
│   └── face_manager.py      # High-level orchestrator
│
├── examples/                 # Example scripts
│   ├── face_detection_example.py
│   ├── face_labeling_example.py
│   ├── face_embeddings_example.py
│   ├── complete_workflow_example.py
│   └── README.md
│
├── docs/                     # Documentation
│   ├── face-recognition-guide.md
│   └── ...
│
├── tests/                    # Unit tests (optional)
│
├── Utilities/               # Helper scripts
│   ├── demo_face_detection.py      # Simple face detection demo
│   ├── verify_installation.py      # Check if everything works
│   ├── test_installation.py        # Comprehensive tests
│   ├── check_deepface_models.py    # Check model downloads
│   ├── simple_download.py          # Safe model downloader
│   ├── template_script.py          # Template for your scripts
│   └── setup_suppress_warnings.py  # TensorFlow warning suppressor
│
└── Documentation/
    ├── README.md                      # This file
    ├── FACE_RECOGNITION_QUICKSTART.md # Quick start guide
    ├── IMPLEMENTATION_SUMMARY.md       # Implementation details
    ├── TESTING_GUIDE.md               # How to test
    ├── TENSORFLOW_WARNING_FIX.md      # Fix TensorFlow warnings
    ├── DEEPFACE_ALTERNATIVES.md       # Alternative libraries
    └── DEEPFACE_DOWNLOADS.md          # About model downloads
```

## 🎯 Main Features

### 1. Face Detection
Extract faces from group photos using multiple detection backends.

```python
from travel_log import FaceDetector

detector = FaceDetector(detector_backend='opencv')
faces = detector.save_extracted_faces("group_photo.jpg", "faces/")
```

### 2. Face Recognition
Identify people in photos automatically.

```python
from travel_log import FaceLabeler

labeler = FaceLabeler("face_database")
labeler.add_person("Alice", ["alice1.jpg", "alice2.jpg"])
result = labeler.identify_face("unknown.jpg")
```

### 3. Face Embeddings
Generate signature vectors for advanced analysis.

```python
from travel_log import FaceEmbeddings

embedder = FaceEmbeddings(model_name='Facenet512')
embedding = embedder.generate_embedding("face.jpg")
```

### 4. Complete Workflow
End-to-end photo processing pipeline.

```python
from travel_log import TravelLogFaceManager

manager = TravelLogFaceManager("workspace")
results = manager.process_directory("photos/")
summary = manager.generate_summary_report()
```

## 🛠️ Essential Utilities

### Check Installation
```bash
uv run python verify_installation.py
```

### Process Photos
```bash
uv run python demo_face_detection.py your_photo.jpg
```

### Check Model Downloads
```bash
uv run python check_deepface_models.py
```

### Download Models (Optional)
```bash
uv run python simple_download.py
```

## 📚 Documentation

- **[Quick Start](FACE_RECOGNITION_QUICKSTART.md)** - Get started in 5 minutes
- **[Testing Guide](TESTING_GUIDE.md)** - How to test the application
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[Comprehensive Guide](docs/face-recognition-guide.md)** - Full documentation
- **[TensorFlow Warnings](TENSORFLOW_WARNING_FIX.md)** - Fix common issues
- **[Alternatives](DEEPFACE_ALTERNATIVES.md)** - Other face recognition libraries

## 💡 Common Tasks

### Extract faces from photos
```bash
uv run python demo_face_detection.py photo.jpg
ls demo_workspace/faces/
```

### Build face database
```python
from travel_log import FaceLabeler

labeler = FaceLabeler("face_database")
labeler.add_person("Alice", ["samples/alice1.jpg", "samples/alice2.jpg"])
```

### Identify faces
```python
result = labeler.identify_face("unknown_face.jpg")
print(f"This is {result['name']} (confidence: {result['confidence']:.1%})")
```

## ⚠️ macOS Users

If you see TensorFlow mutex warnings:
- ✅ They're harmless - your code works fine
- ✅ Use OpenCV backend (recommended): `detector_backend='opencv'`
- ✅ See [TENSORFLOW_WARNING_FIX.md](TENSORFLOW_WARNING_FIX.md) for details

## 🆘 Troubleshooting

### No faces detected
- Try different detector: `detector_backend='mtcnn'`
- Check image quality
- Ensure faces are visible

### Import errors
```bash
uv sync  # Reinstall dependencies
```

### Slow first run
- Models download on first use (~100-500 MB)
- Subsequent runs are much faster
- Pre-download: `uv run python simple_download.py`

## 🤝 Weekly Check-ins

Use your 30-minute meetings with Chander and Asif to discuss:
- Face detection accuracy
- Recognition model tuning
- Database management
- Integration with other features
- Performance optimization

## 📦 Dependencies

- DeepFace >= 0.0.93
- OpenCV >= 4.8.0
- Pillow >= 10.0.0
- NumPy >= 1.24.0

All managed via `pyproject.toml` and installed with `uv sync`.

## 🧹 Project Maintenance

### Cleanup Script
```bash
./cleanup.sh  # Removes temporary test files
```

### Restore Deleted Files
```bash
ls .cleanup_backup/  # See backed up files
mv .cleanup_backup/filename.py .  # Restore if needed
```

### Permanent Cleanup
```bash
rm -rf .cleanup_backup/  # Delete backup permanently
```

## 📊 Project Status

✅ **Implemented:**
- Face detection and extraction
- Face recognition and labeling
- Face embeddings generation
- Complete workflow management
- Comprehensive documentation
- Example scripts

🎯 **Ready to Use:**
- All core modules working
- Examples documented
- Installation verified
- Troubleshooting guides included

## License

Copyright (c) 2016-2025. SupportVectors AI Lab

This code is part of the training material and, therefore, part of the intellectual property.
It may not be reused or shared without the explicit, written permission of SupportVectors.

Use is limited to the duration and purpose of the training at SupportVectors.

Author: SupportVectors AI Training Team


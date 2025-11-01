# Travel Log

A Python project for managing travel memories with advanced face recognition capabilities.

## Features

### 📱 HEIC Image Support (NEW!)

Full support for Apple's HEIC/HEIF image format:
- **Automatic conversion** to JPEG for processing
- **Command-line tools** support HEIC files
- **Streamlit app** can upload and process HEIC images
- **Works seamlessly** with all face detection features

### 🎭 Face Recognition

Travel Log now includes comprehensive face detection, recognition, and management capabilities:

- **Face Detection**: Automatically detect and extract faces from group photos
- **Face Recognition**: Identify people in your travel photos
- **Face Labeling**: Automatically label faces with names
- **Face Embeddings**: Generate signature vectors for advanced similarity search
- **Face Clustering**: Group similar faces together
- **Batch Processing**: Efficiently process entire photo collections

### Core Capabilities

1. **Face Detection & Extraction**
   - Multiple detection backends (MTCNN, RetinaFace, SSD, OpenCV, Dlib)
   - Automatic face extraction from group photos
   - Bounding box annotation

2. **Face Recognition & Labeling**
   - Build database of known people
   - Automatic face identification
   - Confidence scoring
   - Face verification (same person check)

3. **Face Embeddings**
   - Multiple recognition models (Facenet512, ArcFace, VGG-Face, etc.)
   - 128-4096 dimensional embeddings
   - Similarity computation (cosine, euclidean)
   - Embedding persistence

4. **Complete Workflow Management**
   - End-to-end photo processing pipeline
   - Organized workspace structure
   - Summary reports and statistics
   - Export organized datasets

## Installation

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

## 🌐 Web Interface (NEW!)

Launch the interactive Streamlit web app:

```bash
# Using the helper script
./run_app.sh

# Or directly
uv run streamlit run app.py
```

The app provides:
- 📤 **Drag-and-drop image upload**
- 🔍 **Real-time face detection**
- 👤 **Visual preview of detected faces**
- ⬇️ **Download extracted faces**
- ⚙️ **Configure detection settings**
- 📊 **Statistics and confidence scores**
- 📸 **EXIF metadata display** - Shows date, time, GPS location, camera info

**Troubleshooting:**
- Port in use? `./kill_streamlit.sh` then `./run_app.sh`
- Can't connect? Use `http://localhost:8501` (not external IP)
- See [Troubleshooting Guide](documentation/TROUBLESHOOTING.md) for complete guide

## Quick Start

### Basic Face Detection

```python
from travel_log import FaceDetector

detector = FaceDetector(detector_backend='mtcnn')
faces = detector.save_extracted_faces(
    image_path="group_photo.jpg",
    output_dir="extracted_faces"
)
print(f"Extracted {len(faces)} faces")
```

### Face Recognition

```python
from travel_log import FaceLabeler

labeler = FaceLabeler(database_path="face_database")
labeler.add_person("Alice", ["alice1.jpg", "alice2.jpg"])

result = labeler.identify_face("unknown_face.jpg")
print(f"Identified as: {result['name']} (confidence: {result['confidence']:.2%})")
```

### Complete Workflow

```python
from travel_log import TravelLogFaceManager

manager = TravelLogFaceManager(workspace_dir="my_trip")

# Add known people
manager.add_person_to_database("Alice", ["alice_samples/1.jpg"])

# Process all photos
results = manager.process_directory("trip_photos")

# Generate summary
summary = manager.generate_summary_report()
print(f"Processed {summary['extracted_faces']} faces")
```

## Documentation

- **[Quick Start Guide](documentation/FACE_RECOGNITION_QUICKSTART.md)** - Get started in 5 minutes
- **[Configuration Guide](documentation/CONFIGURATION.md)** - Configure paths and settings
- **[Complete Documentation Index](documentation/README.md)** - All guides and tutorials
- **[Face Recognition Guide](docs/face-recognition-guide.md)** - Comprehensive technical guide
- **[Examples](examples/README.md)** - Example scripts demonstrating all features

### Key Guides
- [Configuration Guide](documentation/CONFIGURATION.md) - Configure image paths and settings
- [Testing Guide](documentation/TESTING_GUIDE.md) - How to test the application
- [TensorFlow Warnings](documentation/TENSORFLOW_WARNING_FIX.md) - Fix common issues
- [Alternatives](documentation/DEEPFACE_ALTERNATIVES.md) - Other face recognition libraries

## Project Structure

```
travel_log/
├── src/travel_log/
│   ├── __init__.py
│   ├── face_detector.py      # Face detection & extraction
│   ├── face_embeddings.py    # Face embeddings generation
│   ├── face_labeler.py       # Face recognition & labeling
│   └── face_manager.py       # High-level orchestrator
├── examples/
│   ├── face_detection_example.py
│   ├── face_labeling_example.py
│   ├── face_embeddings_example.py
│   ├── complete_workflow_example.py
│   └── README.md
├── docs/
│   ├── face-recognition-guide.md
│   └── ...
└── tests/
```

## Examples

Check out the `examples/` directory for:

1. **face_detection_example.py** - Basic face detection and extraction
2. **face_labeling_example.py** - Face recognition and labeling
3. **face_embeddings_example.py** - Embeddings and similarity search
4. **complete_workflow_example.py** - End-to-end travel photo processing

## Requirements

- Python >= 3.12
- DeepFace >= 0.0.93
- OpenCV >= 4.8.0
- Pillow >= 10.0.0
- NumPy >= 1.24.0

All dependencies are managed via `pyproject.toml` and installed with `uv sync`.

## Development

### Running Tests

```bash
uv run pytest
```

### Building Documentation

```bash
./build_docs.sh
```

### Serving Documentation Locally

```bash
./serve_docs.sh
```

## Use Cases

### 1. Organize Trip Photos by Person

```python
manager = TravelLogFaceManager("paris_trip")
manager.add_person_to_database("Alice", ["alice_samples/1.jpg"])
manager.process_directory("paris_photos")
manager.export_labeled_dataset("organized_by_person")
```

### 2. Find All Photos of a Specific Person

```python
labeler = FaceLabeler("face_database")
results = labeler.identify_faces_batch(all_face_images)
alice_photos = [r['query_image'] for r in results if r and r['name'] == 'Alice']
```

### 3. Cluster Unknown Faces

```python
manager = TravelLogFaceManager("workspace")
manager.process_directory("photos")
clusters = manager.get_face_clusters(threshold=0.6)
```

## Weekly Project Checkins

We have 30-minute weekly meetings with Chander and Asif to discuss:

- Face detection accuracy and optimization
- Recognition model selection and tuning
- Database management best practices
- Integration with other travel log features
- Performance optimization for large photo collections
- Handling edge cases and challenging scenarios

## Support

For questions and clarifications:
- Weekly check-ins with Chander and Asif
- Review the [Face Recognition Guide](docs/face-recognition-guide.md)
- Check the [examples](examples/README.md)

## License

Copyright (c) 2016-2025. SupportVectors AI Lab

This code is part of the training material and, therefore, part of the intellectual property.
It may not be reused or shared without the explicit, written permission of SupportVectors.

Use is limited to the duration and purpose of the training at SupportVectors.

Author: SupportVectors AI Training Team

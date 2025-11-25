# Travel Log

An AI-powered travel photo management system with face recognition, journey mapping, and natural language search capabilities.

## Features

### 🆕 Latest Features (v2.0)

#### 🗺️ Journey Mapping
- **Interactive Maps**: Visualize your travel paths on Google Maps and Leaflet.js
- **Daily Breakdowns**: Group photos by day with separate route maps
- **Chronological Routes**: Automatic route generation through all photo locations
- **People Integration**: See who was at each location

#### 🔎 Semantic Search
- **Natural Language Queries**: Ask questions like "When, where and with whom did I see the turtles on the beach?"
- **Multi-modal Search**: Search across people, locations, dates, and photo content
- **Smart Ranking**: Relevance scoring with match explanations
- **Query Understanding**: Automatically extracts people, places, dates, and keywords

#### 🌍 Location Contextualization
- **Wikipedia Integration**: Automatic place descriptions and context
- **Reverse Geocoding**: GPS coordinates → place names
- **Location Intelligence**: Enriches photos with location history and facts

### 📱 HEIC Image Support

Full support for Apple's HEIC/HEIF image format:
- **Automatic conversion** to JPEG for processing
- **Command-line tools** support HEIC files
- **Streamlit app** can upload and process HEIC images
- **Works seamlessly** with all face detection features

### 🎭 Face Recognition

Comprehensive face detection, recognition, and management capabilities:

- **Face Detection**: Automatically detect and extract faces from group photos
- **Face Recognition**: Identify people in your travel photos using VGG-Face/DeepFace
- **Face Labeling**: Automatically label faces with names
- **Face Embeddings**: Generate 4096D signature vectors for similarity search
- **Qdrant Integration**: Vector database storage for semantic face search
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

## 🌐 Web Interface

Launch the interactive Streamlit web app:

```bash
# Using the helper script
./run_app.sh

# Or directly
uv run streamlit run app.py
```

The app provides **7 powerful tabs**:

1. **🔍 Face Detection** - Detect and extract faces from photos
2. **🎯 Face Identification** - Identify people using DeepFace + Qdrant
3. **📔 Travel Log** - View all photos with captions, locations, and people
4. **🗄️ Qdrant Storage** - Save photos to vector database with metadata
5. **💾 Face Database** - Manage known people for face recognition
6. **🗺️ Journey Map** - Visualize travel paths on interactive maps
7. **🔎 Search** - Natural language search across all your photos

**Key Features:**
- 📤 **Drag-and-drop image upload** with HEIC support
- 🔍 **Real-time face detection** with RetinaFace/MTCNN
- 👤 **Automatic face identification** via Qdrant vector search
- 🤖 **AI caption generation** with LLaVA vision model
- 📸 **EXIF metadata display** - Date, time, GPS location, camera info
- 🗺️ **Journey mapping** - Google Maps routes and interactive maps
- 🔎 **Semantic search** - "Find photos with Sarah at the beach"
- ⬇️ **Download extracted faces** individually or as ZIP

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

### New Features (v2.0)
- **[NEW_FEATURES.md](NEW_FEATURES.md)** - Complete guide to Journey Mapping, Search, and Location Context
- **[test_semantic_search.py](test_semantic_search.py)** - Test script for semantic search

### Core Documentation
- **[Quick Start Guide](documentation/FACE_RECOGNITION_QUICKSTART.md)** - Get started in 5 minutes
- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - Complete setup and usage guide
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current project status and features
- **[CLAUDE.md](CLAUDE.md)** - Development guide for Claude Code

### Technical Guides
- **[Configuration Guide](documentation/CONFIGURATION.md)** - Configure paths and settings
- **[Face Recognition Guide](docs/face-recognition-guide.md)** - Comprehensive technical guide
- **[Testing Guide](documentation/TESTING_GUIDE.md)** - How to test the application
- **[Examples](examples/README.md)** - Example scripts demonstrating all features

### Advanced Topics
- [Qdrant Integration](docs/guides/QDRANT_GUIDE.md) - Vector database guide
- [Caption Generation](docs/guides/CAPTION_GENERATION_GUIDE.md) - AI captions with LLaVA
- [Face Identification](docs/guides/FACE_IDENTIFICATION_GUIDE.md) - Qdrant-based face recognition
- [Batch Processing](docs/guides/BATCH_PROCESSING_GUIDE.md) - Process multiple photos
- [GPU Usage](docs/technical/GPU_USAGE.md) - GPU acceleration guide

## Project Structure

```
travel_log/
├── src/travel_log/
│   ├── __init__.py
│   ├── face_detector.py         # Face detection & extraction
│   ├── face_embeddings.py       # Face embeddings generation
│   ├── face_labeler.py          # Face recognition & labeling
│   ├── face_manager.py          # High-level orchestrator
│   ├── caption_generator.py     # LLaVA AI captions
│   ├── dspy_llava_integration.py # DSPy enhanced captions
│   ├── qdrant_store.py          # Vector database integration
│   ├── journey_mapper.py        # NEW: Journey mapping
│   ├── location_context.py      # NEW: Wikipedia integration
│   ├── semantic_search.py       # NEW: Natural language search
│   ├── image_utils.py           # HEIC/image utilities
│   └── exif_utils.py            # GPS/EXIF metadata
├── app.py                       # Streamlit web interface
├── store_reference_faces.py     # Populate Qdrant with known faces
├── test_semantic_search.py      # NEW: Test search functionality
├── examples/
│   ├── face_detection_example.py
│   ├── face_labeling_example.py
│   ├── face_embeddings_example.py
│   └── complete_workflow_example.py
├── docs/
│   ├── guides/                  # User guides
│   ├── technical/               # Technical documentation
│   └── archive/                 # Superseded docs
├── tests/                       # Test suite
└── face_database/               # Known people for face recognition
```

## Examples

Check out the `examples/` directory for:

1. **face_detection_example.py** - Basic face detection and extraction
2. **face_labeling_example.py** - Face recognition and labeling
3. **face_embeddings_example.py** - Embeddings and similarity search
4. **complete_workflow_example.py** - End-to-end travel photo processing

## Requirements

### Core Dependencies
- Python >= 3.12
- DeepFace >= 0.0.93 (face recognition)
- OpenCV >= 4.8.0 (image processing)
- Pillow >= 10.0.0 (image handling)
- NumPy >= 1.24.0 (numerical operations)
- Qdrant Client >= 1.15.1 (vector database)
- Streamlit >= 1.28.0 (web interface)

### New Dependencies (v2.0)
- sentence-transformers >= 5.1.2 (text embeddings for search)
- requests >= 2.31.0 (Wikipedia API)

### Optional
- DSPy AI >= 3.0.4 (enhanced captions)
- Ollama with llava:7b (AI caption generation)

All dependencies are managed via `pyproject.toml` and installed with `uv sync`.

### External Services
- **Qdrant**: Vector database (running on sapphire:6333)
- **Ollama**: Local LLM server for caption generation (optional)

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

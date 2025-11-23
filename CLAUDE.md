# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Travel Log is an AI-powered travel photo management system with face recognition, vector database storage (Qdrant), and AI caption generation (LLaVA). Features include:
- **Face Detection**: RetinaFace/MTCNN with 30% padding for better crops
- **Face Recognition**: DeepFace with VGG-Face (4096D embeddings) via Qdrant vector search
- **Photo Management**: Permanent storage with GPS location (manual or EXIF)
- **Travel Log**: View all photos with captions, locations, and identified people
- **AI Captions**: LLaVA vision model for intelligent photo descriptions
- **GPU Acceleration**: Auto-detected RTX 4090 support

## Development Setup

```bash
# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate

# Run tests
uv run pytest

# Start Streamlit app (localhost only)
./run_app.sh

# Or start with network access
./run_app_network.sh
```

## Architecture

### Core Components

1. **Face Detection & Recognition** (`src/travel_log/`)
   - `face_detector.py`: GPU-accelerated face detection using DeepFace (MTCNN, RetinaFace, SSD, OpenCV, Dlib)
   - `face_labeler.py`: Face identification against known database using VGG-Face/Facenet512
   - `face_embeddings.py`: Generate 512D or 4096D face embeddings for similarity search
   - `face_manager.py`: High-level orchestrator for complete face processing workflows

2. **Vector Database** (`src/travel_log/qdrant_store.py`)
   - Qdrant vector database integration for storing photos and faces
   - Three collections: `reference_faces` (known people), `travel_photos` (photo metadata), `detected_faces` (extracted faces)
   - Supports semantic search by face similarity, person name, GPS location, and timestamps

3. **AI Caption Generation** (`src/travel_log/caption_generator.py`, `dspy_llava_integration.py`)
   - LLaVA vision model integration via Ollama for image captioning
   - DSPy-enhanced captions with context (face names, GPS, timestamps)
   - Generates titles, detailed captions, travel descriptions, hashtags

4. **Streamlit Web App** (`app.py`)
   - 5 tabs: Face Detection, Face Identification, Travel Log, Qdrant Storage, Face Database
   - **Face Detection**: RetinaFace (default) or MTCNN backend with 0.6 confidence threshold
   - **Face Identification**: Automatic DeepFace/VGG-Face via Qdrant (no manual initialization)
   - **Travel Log**: View all photos with captions, locations, people, and delete functionality
   - **Qdrant Storage**: Save photos with manual GPS location input if EXIF missing
   - Session state management for detector, faces, identifications, captions, qdrant_store

5. **Image Utilities** (`src/travel_log/`)
   - `image_utils.py`: HEIC/HEIF image conversion to JPEG
   - `exif_utils.py`: EXIF metadata extraction (GPS, datetime, camera info)

### Key Design Patterns

- **Orchestrator Pattern**: `TravelLogFaceManager` coordinates all face operations
- **Component Initialization**: Each component (detector, labeler, embeddings) can be used independently or via manager
- **Session State**: Streamlit app maintains state across tabs (`st.session_state.detector`, `labeler`, `detected_faces`, etc.)
- **Lazy Loading**: Models download on first use (~100-500MB)

### Configuration

- Configuration file: `config.yaml`
- Access via: `from travel_log import config`
- Personal photos path: `config['images']['personal_photos_dir']`
- Default backend: `config['face_detection']['default_backend']` (mtcnn)
- Default model: `config['face_detection']['default_model']` (Facenet512)

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_qdrant_search.py

# Test face detection
python tests/simple_face_test.py

# Test Qdrant face identification
python tests/identify_with_qdrant.py

# View Qdrant database
python tests/quick_view_qdrant.py
```

## Important Implementation Details

### Face Recognition Pipeline

1. **Detection**: Extract faces using RetinaFace (default) or MTCNN
   - Confidence threshold: 0.6 (optimized for masked faces and challenging angles)
   - Face padding: 30% around detected face for better context
   - Handles masks, sunglasses, side profiles, and poor lighting

2. **Embedding**: Generate 4096D embeddings using DeepFace with VGG-Face model
   - GPU-accelerated on RTX 4090
   - Automatic model download on first use

3. **Identification**: Vector similarity search in Qdrant reference_faces collection
   - Distance threshold: ≤ 0.60 (very lenient for real-world photos)
   - Cosine similarity metric
   - Handles masked faces, different angles, and lighting variations

### Qdrant Integration

- **Server**: Running on `sapphire:6333` (configurable via `qdrant_url`)
- **Collections**:
  - `reference_faces`: Known people (4096D VGG-Face embeddings)
  - `travel_photos`: Photo metadata + captions + GPS + faces
  - `detected_faces`: Individual face embeddings from photos
- **Reference faces**: Stored via `store_reference_faces.py` script
- **Search API**: Uses `client.query_points()` (modern Qdrant API, not deprecated `search()`)
- **Photo Storage**: Permanent storage in `uploaded_photos/` directory (not `/tmp/`)

### GPU Usage

- **Auto-detection**: DeepFace automatically uses GPU if available (CUDA)
- **VGG-Face**: Recommended model for GPU (4096D embeddings)
- **Performance**: ~0.1-0.5s per face on GPU vs 1-3s on CPU
- See `docs/technical/GPU_USAGE.md` for details

### Streamlit App

- **Port**: 8501 (localhost) or network-accessible via `run_app_network.sh`
- **Session state**: All processing state stored in `st.session_state`
- **Face identification**: Automatic DeepFace + Qdrant (no model selection needed)
- **Photo storage**: `uploaded_photos/` with timestamp-based unique filenames
- **Manual GPS**: Add location coordinates if photo has no EXIF GPS data
- **Travel Log**: Browse all photos with captions, GPS, people, and delete capability

### TensorFlow Warnings Suppression

- All TF warnings suppressed in `src/travel_log/__init__.py` before imports
- Environment variables set: `TF_CPP_MIN_LOG_LEVEL=3`, `TF_ENABLE_ONEDNN_OPTS=0`

## Common Tasks

### Add a new person to face database

```bash
# 1. Add reference images to face_database/<person_name>/
mkdir -p "face_database/New Person"
cp photo1.jpg photo2.jpg photo3.jpg "face_database/New Person/"

# 2. Update Qdrant reference faces (regenerates ALL embeddings)
uv run python store_reference_faces.py

# 3. Verify the person was added
# The script will show all people and their image counts
```

**Tips for best reference photos:**
- Add 2-3 photos from different angles (front, side profile)
- Include photos with and without masks/sunglasses
- Use clear, well-lit photos
- Different expressions and settings help improve matching

### Process a directory of photos

```python
from travel_log import TravelLogFaceManager

manager = TravelLogFaceManager("workspace")
manager.add_person_to_database("Alice", ["alice1.jpg", "alice2.jpg"])
results = manager.process_directory("trip_photos")
summary = manager.generate_summary_report()
```

### Generate captions for an image

```python
from travel_log import CaptionGenerator
from PIL import Image

generator = CaptionGenerator(model_name="llava:7b")
image = Image.open("photo.jpg")
captions = generator.generate_all(image)  # Returns title, caption, travel_caption
```

### Use the Travel Log

**Web Interface (Recommended):**
1. Start the app: `./run_app.sh`
2. Upload photo in "Face Detection" tab
3. Click "Identify Faces" in "Face Identification" tab
4. View all photos in "Travel Log" tab
5. Delete unwanted photos with 🗑️ button

**Search photos by person via Python:**
```python
from travel_log.qdrant_store import create_qdrant_store

store = create_qdrant_store(url="http://sapphire:6333")
photos = store.search_by_person("Alice", limit=20)
```

### Add GPS Location Manually

If a photo doesn't have GPS data in EXIF:
1. Upload photo and save to Qdrant
2. In "Qdrant Storage" tab, expand "📍 Add Location Manually"
3. Enter latitude and longitude (find on Google Maps)
4. Click "✅ Add Location to Photo"
5. Save to Qdrant

**Example coordinates:**
- Chennai: `13.0827, 80.2707`
- Mumbai: `19.0760, 72.8777`
- Delhi: `28.6139, 77.2090`

## Dependencies

- **Core**: DeepFace (≥0.0.93), OpenCV (≥4.8.0), Pillow (≥10.0.0), NumPy (≥1.24.0)
- **ML**: TensorFlow/Keras (via tf-keras ≥2.20.0), PyTorch (≥2.0.0)
- **Vector DB**: qdrant-client (≥1.15.1)
- **AI**: transformers (≥4.30.0), dspy-ai (≥3.0.4)
- **UI**: Streamlit (≥1.28.0)
- **Images**: pillow-heif (≥0.13.0) for HEIC support
- **Framework**: svlearn-bootcamp (≥0.1.7) for config management

## Scripts

- `run_app.sh`: Start Streamlit app (localhost only)
- `run_app_network.sh`: Start with network access
- `kill_streamlit.sh`: Kill existing Streamlit processes
- `store_reference_faces.py`: Store face_database/ images to Qdrant
- `build_docs.sh`: Build documentation
- `serve_docs.sh`: Serve docs locally

## Documentation

- Main: `README.md`, `QUICK_START_GUIDE.md`, `PROJECT_STATUS.md`
- Guides: `docs/guides/` (7 guides for face extraction, Qdrant, Ollama, captions, batch processing)
- Technical: `docs/technical/` (SESSION_SUMMARY.md, FACE_MATCHING_FIX.md, GPU_USAGE.md, CONFIGURATION.md)
- Archive: `docs/archive/` (superseded docs)

## Weekly Check-ins

This is a training project with 30-minute weekly meetings with Chander and Asif to discuss face detection accuracy, model selection, database management, and performance optimization.

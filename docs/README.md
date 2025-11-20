# Travel Log Documentation

**Last Updated**: November 15, 2025
**Status**: ✅ Production Ready

## 📚 Quick Navigation

### 🚀 Getting Started
- **[Quick Start Guide](../QUICK_START_GUIDE.md)** - 5-minute setup
- **[Main README](../README.md)** - Complete project overview

### 📖 User Guides
- **[Face Extraction Guide](guides/FACE_EXTRACTION_GUIDE.md)** - Extract and save face images
- **[Qdrant Integration Guide](guides/QDRANT_INTEGRATION_GUIDE.md)** - Vector database setup
- **[Ollama Integration Guide](guides/OLLAMA_INTEGRATION_GUIDE.md)** - AI caption generation
- **[Caption Generator Guide](guides/CAPTION_GENERATOR_GUIDE.md)** - Detailed caption generation
- **[DSPy Streamlit Integration](guides/DSPY_STREAMLIT_INTEGRATION.md)** - DSPy integration
- **[Qdrant Streamlit Integration](guides/QDRANT_STREAMLIT_INTEGRATION.md)** - Streamlit + Qdrant
- **[Batch Processing Guide](guides/STREAMLIT_BATCH_PROCESSING.md)** - Batch face processing

### 🔧 Technical Documentation
- **[Session Summary](technical/SESSION_SUMMARY.md)** - Latest implementation details
- **[Face Matching Fix](technical/FACE_MATCHING_FIX.md)** - Fixed "no match found" issue
- **[Reference Faces in Qdrant](technical/REFERENCE_FACES_IN_QDRANT.md)** - Qdrant integration
- **[GPU Usage](technical/GPU_USAGE.md)** - GPU auto-detection and performance
- **[Configuration](technical/CONFIGURATION.md)** - Project configuration

### 📦 Archive
Old and superseded documentation: [archive/](archive/)

---

## 🎯 By Use Case

### I want to...

#### Get Started Quickly
→ [Quick Start Guide](../QUICK_START_GUIDE.md)

#### Fix "No match found" Issue
→ [Face Matching Fix](technical/FACE_MATCHING_FIX.md)

#### Understand GPU Usage
→ [GPU Usage Guide](technical/GPU_USAGE.md)

#### Setup Reference Faces
→ [Reference Faces in Qdrant](technical/REFERENCE_FACES_IN_QDRANT.md)

#### Extract Face Images
→ [Face Extraction Guide](guides/FACE_EXTRACTION_GUIDE.md)

#### Integrate with Qdrant
→ [Qdrant Integration Guide](guides/QDRANT_INTEGRATION_GUIDE.md)

#### Generate AI Captions
→ [Ollama Integration Guide](guides/OLLAMA_INTEGRATION_GUIDE.md)

---

## 📊 Current System Status

### Database Configuration
- **Qdrant URL**: `http://sapphire:6333`
- **Collections**:
  - `travel_photos` - Photo metadata (2 photos)
  - `detected_faces` - Individual face embeddings (0)
  - `reference_faces` - Known people (10 people)

### Face Recognition
- **Model**: VGG-Face (4096D embeddings)
- **Detector**: RetinaFace
- **Distance Metric**: Cosine similarity
- **Threshold**: 0.25

### GPU
- **Device**: NVIDIA GeForce RTX 4090
- **Memory**: 8GB allocated (of 24GB)
- **Framework**: TensorFlow 2.20.0
- **Status**: ✅ Auto-detected

### Ollama Models
- `llava:7b` - Image captioning (4.7 GB)
- `qwen2.5vl:7b` - Vision-language (6.0 GB)
- `llama3:latest` - Text generation (4.7 GB)

---

## 🗂️ Documentation Structure

```
docs/
├── README.md                    ← You are here
├── INDEX.md                     ← Detailed index (legacy)
│
├── guides/                      ← User guides
│   ├── FACE_EXTRACTION_GUIDE.md
│   ├── QDRANT_INTEGRATION_GUIDE.md
│   ├── OLLAMA_INTEGRATION_GUIDE.md
│   ├── CAPTION_GENERATOR_GUIDE.md
│   ├── DSPY_STREAMLIT_INTEGRATION.md
│   ├── QDRANT_STREAMLIT_INTEGRATION.md
│   └── STREAMLIT_BATCH_PROCESSING.md
│
├── technical/                   ← Technical documentation
│   ├── SESSION_SUMMARY.md       ← Latest session (Nov 14, 2025)
│   ├── FACE_MATCHING_FIX.md     ← Face matching troubleshooting
│   ├── REFERENCE_FACES_IN_QDRANT.md
│   ├── GPU_USAGE.md             ← GPU documentation
│   └── CONFIGURATION.md
│
└── archive/                     ← Outdated documentation
    ├── COMPLETE_WORKFLOW.md
    ├── FACE_STORAGE_GUIDE.md
    └── ... (old versions)
```

---

## 🔧 Key Scripts

### Database Management
```bash
# View database status
uv run python quick_view_qdrant.py

# Store reference faces in Qdrant
uv run python store_reference_faces.py

# Verify reference faces
uv run python store_reference_faces.py --verify-only
```

### Testing & Diagnostics
```bash
# Test face matching (Qdrant-based)
uv run python identify_with_qdrant.py ~/photos/IMG_0276.jpeg

# Test face matching (DeepFace-based)
uv run python simple_face_test.py ~/photos/IMG_0276.jpeg

# Check GPU detection
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Main Application
```bash
# Launch Streamlit app
uv run streamlit run app.py
```

---

## 📖 Documentation by Topic

### Face Recognition

| Topic | Document |
|-------|----------|
| Quick Start | [QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md) |
| Extraction | [Face Extraction Guide](guides/FACE_EXTRACTION_GUIDE.md) |
| Matching Fix | [Face Matching Fix](technical/FACE_MATCHING_FIX.md) |
| Reference Faces | [Reference Faces](technical/REFERENCE_FACES_IN_QDRANT.md) |

### Vector Database

| Topic | Document |
|-------|----------|
| Setup | [Qdrant Integration](guides/QDRANT_INTEGRATION_GUIDE.md) |
| Streamlit | [Qdrant Streamlit](guides/QDRANT_STREAMLIT_INTEGRATION.md) |
| Reference Faces | [Reference Faces](technical/REFERENCE_FACES_IN_QDRANT.md) |

### AI & Machine Learning

| Topic | Document |
|-------|----------|
| GPU Usage | [GPU Usage](technical/GPU_USAGE.md) |
| Captions | [Ollama Integration](guides/OLLAMA_INTEGRATION_GUIDE.md) |
| DSPy | [DSPy Streamlit](guides/DSPY_STREAMLIT_INTEGRATION.md) |

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "No match found" | [Face Matching Fix](technical/FACE_MATCHING_FIX.md) |
| Qdrant connection | [Session Summary](technical/SESSION_SUMMARY.md#troubleshooting) |
| GPU not detected | [GPU Usage](technical/GPU_USAGE.md#troubleshooting) |

---

## 🎓 Example Workflows

### Add New Person to Database
```bash
# 1. Create directory
mkdir face_database/NewPerson

# 2. Add reference photos
cp photo1.jpg photo2.jpg face_database/NewPerson/

# 3. Store in Qdrant
uv run python store_reference_faces.py

# 4. Verify
uv run python quick_view_qdrant.py
```

### Process Travel Photos
```bash
# 1. Launch app
uv run streamlit run app.py

# 2. In browser:
#    - Upload photo (Photo Upload tab)
#    - Detect faces (Face Detection tab)
#    - Identify faces (Face Identification tab)
#    - Save to Qdrant (click "💾 Save to Qdrant")
```

### Batch Process Directory
```bash
# Run batch processing example
uv run python examples/face_storage_example.py
```

---

## 🔍 Quick Reference

### Collection Schemas

**reference_faces** (4096D embeddings):
```json
{
  "person_name": "sankar",
  "image_filename": "sankar.jpg",
  "image_path": "/full/path/to/face_database/sankar/sankar.jpg",
  "is_reference": true,
  "model_name": "VGG-Face",
  "embedding_dimension": 4096,
  "stored_timestamp": "2025-11-14T01:51:45.123456"
}
```

**travel_photos** (512D embeddings):
```json
{
  "filename": "IMG_0276.jpeg",
  "num_faces": 5,
  "gps_latitude": 13.0827,
  "gps_longitude": 80.2707,
  "datetime_original": "2024-01-15T14:30:00",
  "camera_make": "Apple",
  "camera_model": "iPhone 12"
}
```

**detected_faces** (4096D embeddings):
```json
{
  "person_name": "sankar",
  "photo_id": "abc123",
  "face_index": 0,
  "confidence": 0.95,
  "distance": 0.0520,
  "face_image_path": "/path/to/extracted_faces/face_00.jpg"
}
```

### Distance Thresholds (VGG-Face)

| Distance | Match Quality |
|----------|---------------|
| < 0.10 | Excellent match |
| 0.10 - 0.15 | Good match |
| 0.15 - 0.20 | Acceptable match |
| 0.20 - 0.25 | Weak match |
| > 0.25 | No match |

**Current threshold**: 0.25 (in app.py line 366)

---

## 📝 Recent Changes

### November 14-15, 2025

#### Fixed
- ✅ Face matching now uses Qdrant instead of pickle files
- ✅ VGG-Face model as default (GPU-compatible)
- ✅ "No match found" issue resolved
- ✅ Reference faces stored in Qdrant (10 people)
- ✅ "Ganesh Sankar" duplicate directory merged

#### Added
- ✅ Qdrant-based face identification
- ✅ GPU usage documentation
- ✅ Comprehensive troubleshooting guides
- ✅ Diagnostic scripts (identify_with_qdrant.py, etc.)
- ✅ Reorganized documentation into docs/ directory

#### Updated
- ✅ app.py to use Qdrant for face matching
- ✅ Auto-initialize VGG-Face labeler on startup
- ✅ Documentation structure (docs/ directory)
- ✅ All internal documentation links

See: [Session Summary](technical/SESSION_SUMMARY.md)

---

## 💡 Tips & Best Practices

1. **Reference Photos**: Use clear, front-facing photos with good lighting
2. **Multiple Angles**: Add 2-3 photos per person for better accuracy
3. **Threshold Tuning**: Adjust threshold (0.20-0.30) based on your needs
4. **GPU Memory**: Automatically managed by TensorFlow
5. **Batch Processing**: Use scripts in examples/ for large collections
6. **Database Backup**: Regularly backup Qdrant data

---

## 🤝 Support & Contribution

### Getting Help
- Check relevant guide in [guides/](guides/)
- Review [technical documentation](technical/)
- Run diagnostic scripts
- Check [archive/](archive/) for historical context

### Reporting Issues
- Document the issue with screenshots
- Include error messages
- Note which guide you were following
- Provide system info (GPU, OS, etc.)

---

**Status**: ✅ Production Ready
**Last Session**: November 14, 2025
**Next Steps**: Upload more photos and populate `detected_faces` collection

---

Return to: [Main README](../README.md) | [Quick Start](../QUICK_START_GUIDE.md)

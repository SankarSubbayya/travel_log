# Travel Log - Documentation Index

**Last Updated**: November 14, 2025

## 📚 Quick Navigation

### Getting Started
- **[QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md)** - 5-minute setup guide
- **[README.md](../README.md)** - Complete project overview

### Current Session Documentation
- **[SESSION_SUMMARY.md](SESSION_SUMMARY.md)** - Complete implementation summary
- **[FACE_MATCHING_FIX.md](FACE_MATCHING_FIX.md)** - Fixed "no match found" issue
- **[REFERENCE_FACES_IN_QDRANT.md](REFERENCE_FACES_IN_QDRANT.md)** - Qdrant integration guide

### Technical Documentation
- **[GPU_USAGE.md](GPU_USAGE.md)** - GPU auto-detection and performance
- **[CONFIGURATION.md](CONFIGURATION.md)** - Project configuration

### Feature Guides
- **[FACE_EXTRACTION_GUIDE.md](../FACE_EXTRACTION_GUIDE.md)** - Face extraction documentation
- **[QDRANT_INTEGRATION_GUIDE.md](../QDRANT_INTEGRATION_GUIDE.md)** - Qdrant setup
- **[OLLAMA_INTEGRATION_GUIDE.md](../OLLAMA_INTEGRATION_GUIDE.md)** - AI caption generation

## 🎯 By Use Case

### I want to...

#### Get Started Quickly
→ [QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md)

#### Fix Face Matching Issues
→ [FACE_MATCHING_FIX.md](FACE_MATCHING_FIX.md)

#### Understand GPU Usage
→ [GPU_USAGE.md](GPU_USAGE.md)

#### Setup Reference Faces
→ [REFERENCE_FACES_IN_QDRANT.md](REFERENCE_FACES_IN_QDRANT.md)

#### Extract Face Images
→ [FACE_EXTRACTION_GUIDE.md](../FACE_EXTRACTION_GUIDE.md)

#### Integrate with Qdrant
→ [QDRANT_INTEGRATION_GUIDE.md](../QDRANT_INTEGRATION_GUIDE.md)

#### Generate AI Captions
→ [OLLAMA_INTEGRATION_GUIDE.md](../OLLAMA_INTEGRATION_GUIDE.md)

## 📊 System Overview

### Current Configuration

**Qdrant Database**:
- URL: `http://sapphire:6333`
- Collections: `travel_photos`, `detected_faces`, `reference_faces`
- Reference Faces: 10 people

**Face Recognition**:
- Model: VGG-Face (4096D embeddings)
- Detector: RetinaFace
- Distance Metric: Cosine similarity
- Threshold: 0.25

**GPU**:
- Device: NVIDIA GeForce RTX 4090
- Memory: 8GB allocated (of 24GB total)
- Framework: TensorFlow 2.20.0
- Auto-Detection: ✅ Enabled

**Ollama Models**:
- `llava:7b` - Image captioning (4.7 GB)
- `llama3:latest` - Text generation (4.7 GB)
- `qwen2.5vl:7b` - Vision-language model (6.0 GB)

## 🗂️ File Organization

```
travel_log/
├── documentation/
│   ├── INDEX.md                        ← You are here
│   ├── SESSION_SUMMARY.md              ← Latest session details
│   ├── FACE_MATCHING_FIX.md            ← Troubleshooting guide
│   ├── REFERENCE_FACES_IN_QDRANT.md    ← Qdrant integration
│   ├── GPU_USAGE.md                    ← GPU documentation
│   ├── CONFIGURATION.md                ← Project settings
│   └── archive/                        ← Old documentation
│
├── Root Documentation/
│   ├── README.md                       ← Main documentation
│   ├── QUICK_START_GUIDE.md            ← 5-min setup
│   ├── FACE_EXTRACTION_GUIDE.md        ← Face extraction
│   ├── QDRANT_INTEGRATION_GUIDE.md     ← Qdrant setup
│   └── OLLAMA_INTEGRATION_GUIDE.md     ← AI captions
│
├── Examples/
│   ├── face_storage_example.py
│   ├── qdrant_storage_example.py
│   └── caption_generation_example.py
│
└── Utility Scripts/
    ├── store_reference_faces.py        ← Store faces in Qdrant
    ├── identify_with_qdrant.py         ← Test face matching
    ├── quick_view_qdrant.py            ← View database stats
    └── simple_face_test.py             ← Diagnostic tool
```

## 🔧 Key Scripts

### Database Management
```bash
# View database
uv run python quick_view_qdrant.py

# Store reference faces
uv run python store_reference_faces.py

# Verify reference faces
uv run python store_reference_faces.py --verify-only
```

### Testing & Diagnostics
```bash
# Test face matching (Qdrant)
uv run python identify_with_qdrant.py photo.jpg

# Test face matching (DeepFace)
uv run python simple_face_test.py photo.jpg

# Check GPU
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Main Application
```bash
# Launch Streamlit app
uv run streamlit run app.py
```

## 📖 Documentation by Topic

### Face Recognition

| Topic | Document |
|-------|----------|
| Overview | [README.md](../README.md#face-recognition) |
| Extraction | [FACE_EXTRACTION_GUIDE.md](../FACE_EXTRACTION_GUIDE.md) |
| Matching Fix | [FACE_MATCHING_FIX.md](FACE_MATCHING_FIX.md) |
| Reference Faces | [REFERENCE_FACES_IN_QDRANT.md](REFERENCE_FACES_IN_QDRANT.md) |

### Vector Database

| Topic | Document |
|-------|----------|
| Setup | [QDRANT_INTEGRATION_GUIDE.md](../QDRANT_INTEGRATION_GUIDE.md) |
| Reference Faces | [REFERENCE_FACES_IN_QDRANT.md](REFERENCE_FACES_IN_QDRANT.md) |
| Quick View | Run `quick_view_qdrant.py` |

### AI & Machine Learning

| Topic | Document |
|-------|----------|
| GPU Usage | [GPU_USAGE.md](GPU_USAGE.md) |
| Caption Generation | [OLLAMA_INTEGRATION_GUIDE.md](../OLLAMA_INTEGRATION_GUIDE.md) |
| Models | [README.md](../README.md#features) |

### Troubleshooting

| Issue | Solution Document |
|-------|------------------|
| "No match found" | [FACE_MATCHING_FIX.md](FACE_MATCHING_FIX.md) |
| Qdrant connection | [SESSION_SUMMARY.md](SESSION_SUMMARY.md#troubleshooting) |
| GPU not detected | [GPU_USAGE.md](GPU_USAGE.md#troubleshooting) |

## 🎓 Learning Resources

### Understanding the Pipeline

1. **Photo Upload** → [README.md](../README.md#features)
2. **Face Detection** → [GPU_USAGE.md](GPU_USAGE.md#where-gpu-is-used)
3. **Face Identification** → [FACE_MATCHING_FIX.md](FACE_MATCHING_FIX.md#solution)
4. **Qdrant Storage** → [REFERENCE_FACES_IN_QDRANT.md](REFERENCE_FACES_IN_QDRANT.md#use-cases)

### Example Workflows

**Add New Person**:
1. Create directory: `mkdir face_database/NewPerson`
2. Add photos: `cp photo.jpg face_database/NewPerson/`
3. Store in Qdrant: `uv run python store_reference_faces.py`
4. Test: `uv run python identify_with_qdrant.py test_photo.jpg`

**Process Travel Photos**:
1. Launch app: `uv run streamlit run app.py`
2. Upload photo in "Photo Upload" tab
3. Detect faces in "Face Detection" tab
4. Identify faces in "Face Identification" tab
5. Save to Qdrant

## 🔍 Quick Reference

### Collection Schemas

**reference_faces** (4096D):
```json
{
  "person_name": "sankar",
  "image_path": "/path/to/face_database/sankar/sankar.jpg",
  "is_reference": true,
  "model_name": "VGG-Face",
  "embedding_dimension": 4096
}
```

**travel_photos** (512D):
```json
{
  "filename": "IMG_0276.jpeg",
  "num_faces": 5,
  "gps_latitude": 13.0827,
  "gps_longitude": 80.2707,
  "datetime_original": "2024-01-15T14:30:00"
}
```

**detected_faces** (4096D):
```json
{
  "person_name": "sankar",
  "photo_id": "abc123",
  "face_index": 0,
  "confidence": 0.95,
  "face_image_path": "/path/to/extracted_faces/face_00.jpg"
}
```

### Model Specifications

| Model | Dimension | Use Case | Speed |
|-------|-----------|----------|-------|
| VGG-Face | 4096D | Face recognition | Medium |
| Facenet512 | 512D | Face embeddings | Fast |
| LLaVA 7B | - | Image captions | Slow |

### Distance Thresholds

| Distance | Match Quality |
|----------|---------------|
| < 0.10 | Excellent |
| 0.10 - 0.15 | Good |
| 0.15 - 0.20 | Acceptable |
| 0.20 - 0.25 | Weak |
| > 0.25 | No match |

## 📝 Recent Changes

### November 14, 2025

- ✅ Fixed face matching (VGG-Face + Qdrant)
- ✅ Stored 10 reference faces in Qdrant
- ✅ Updated app.py to use Qdrant for identification
- ✅ Fixed "Ganesh Sankar" duplicate directory issue
- ✅ Created comprehensive documentation
- ✅ Verified GPU auto-detection (RTX 4090)

See: [SESSION_SUMMARY.md](SESSION_SUMMARY.md)

---

**Status**: ✅ Production Ready
**Last Session**: November 14, 2025
**Next Steps**: Upload photos and save to Qdrant

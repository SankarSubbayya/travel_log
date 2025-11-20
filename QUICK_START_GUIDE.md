# Travel Log - Quick Start Guide

**Last Updated**: November 14, 2025
**Status**: ✅ Production Ready

## 🎯 What is Travel Log?

An AI-powered photo management system with:
- **Face Recognition** (VGG-Face on GPU)
- **Vector Search** (Qdrant database)
- **EXIF Extraction** (GPS, datetime, camera)
- **AI Captions** (LLaVA via Ollama)

## ⚡ Quick Start (5 Minutes)

### Step 1: Install

```bash
cd /home/sankar/travel_log
uv sync
```

### Step 2: Add Reference Faces

```bash
# Create person directories
mkdir -p face_database/YourName

# Add photos (1+ per person)
cp your_photo.jpg face_database/YourName/

# Store in Qdrant
uv run python store_reference_faces.py
```

### Step 3: Launch App

```bash
uv run streamlit run app.py
```

Open: http://localhost:8501

### Step 4: Upload & Process Photos

1. **Upload Photo** tab → Upload image
2. **Face Detection** tab → Click "Detect Faces"
3. **Face Identification** tab → Click "🎯 Identify All Faces"
4. **Save to Qdrant** → Click "💾 Save to Qdrant"

Done! Your photos are now searchable by faces.

## 📊 Current Setup

### Database Status

```bash
uv run python tests/quick_view_qdrant.py
```

**Current State**:
- **Qdrant**: `http://sapphire:6333`
- **Reference Faces**: 10 people in database
- **Model**: VGG-Face (4096D embeddings)
- **GPU**: RTX 4090 (auto-detected)

### Collections

| Collection | Purpose | Count |
|------------|---------|-------|
| `reference_faces` | Known people | 10 |
| `travel_photos` | Photo metadata | 2 |
| `detected_faces` | Extracted faces | 0* |

*Will be populated when you save photos

## 🔧 Key Features

### 1. Face Identification (Qdrant-Powered)

**How it works**:
1. Detect faces in photo (RetinaFace detector)
2. Generate 4096D embeddings (VGG-Face model)
3. Search Qdrant `reference_faces` collection
4. Match with cosine similarity (threshold: 0.25)

**Test it**:
```bash
uv run python tests/identify_with_qdrant.py ~/personal_photos/IMG_0276_2.jpeg
```

### 2. Reference Face Management

**Add new person**:
```bash
mkdir face_database/NewPerson
cp photo.jpg face_database/NewPerson/
uv run python store_reference_faces.py
```

**Update existing**:
```bash
cp more_photos.jpg face_database/ExistingPerson/
uv run python store_reference_faces.py  # Rebuilds database
```

### 3. Batch Processing

Process entire directories:
```bash
uv run python examples/face_storage_example.py
```

## 🐛 Troubleshooting

### Issue: "No match found"

**Cause**: Qdrant not connected or reference faces not loaded

**Fix**:
```bash
# 1. Verify Qdrant
curl http://sapphire:6333/collections

# 2. Check reference faces
uv run python tests/quick_view_qdrant.py

# 3. Rebuild if needed
uv run python store_reference_faces.py

# 4. Restart Streamlit
uv run streamlit run app.py
```

### Issue: Qdrant connection failed

**Fix 1**: Check sapphire server
```bash
ssh sankar@sapphire "docker ps | grep qdrant"
```

**Fix 2**: Use SSH tunnel
```bash
ssh -L 6334:localhost:6333 sankar@sapphire -N
# Then update app.py to use localhost:6334
```

### Issue: GPU not detected

**Check**:
```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Expected**: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

## 📈 Performance

With RTX 4090 GPU:
- Face Detection: **~2-3s for 5 faces**
- Embedding Generation: **~0.3s per face**
- Qdrant Search: **<10ms**
- Full Pipeline: **~5-8s per photo**

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Complete project documentation |
| [Documentation Index](docs/README.md) | Complete documentation index |
| [Session Summary](docs/technical/SESSION_SUMMARY.md) | Latest implementation details |
| [Face Matching Fix](docs/technical/FACE_MATCHING_FIX.md) | Face matching troubleshooting |
| [Reference Faces in Qdrant](docs/technical/REFERENCE_FACES_IN_QDRANT.md) | Qdrant integration |

## 🔑 Key Scripts

```bash
# View database
uv run python tests/quick_view_qdrant.py

# Test face matching
uv run python tests/identify_with_qdrant.py photo.jpg

# Store reference faces
uv run python store_reference_faces.py

# Simple face test
uv run python tests/simple_face_test.py photo.jpg
```

## 💡 Tips

1. **Best Results**: Use clear, front-facing reference photos
2. **Multiple Photos**: Add 2-3 photos per person for better accuracy
3. **GPU Usage**: Automatic - no configuration needed
4. **Threshold**: Adjust in app.py (line 366) if needed (default: 0.25)
5. **Batch Processing**: Use examples/ scripts for large collections

## ✅ Verification

Test everything is working:

```bash
# 1. Check Qdrant
uv run python tests/quick_view_qdrant.py

# 2. Test face matching
uv run python tests/identify_with_qdrant.py ~/personal_photos/IMG_0276_2.jpeg

# 3. Expected output:
#    ✅ MATCHED: Appa
#    ✅ MATCHED: Ganesh Sankar
#    ✅ MATCHED: Meena
#    ✅ MATCHED: Amma
#    ✅ MATCHED: sankar
```

All 5 faces should match!

---

**Need Help?** Check [docs/](docs/) or run diagnostic scripts in [tests/](tests/).

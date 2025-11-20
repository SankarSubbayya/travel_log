# Travel Log - Tests & Diagnostics

Test scripts and diagnostic tools for Travel Log.

## 🧪 Test Scripts

### Face Recognition Tests

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_vggface_retinaface.py` | Test VGG-Face + RetinaFace combination | `uv run python tests/test_vggface_retinaface.py photo.jpg` |
| `simple_face_test.py` | Quick face matching test | `uv run python tests/simple_face_test.py photo.jpg` |
| `quick_face_test.py` | Test multiple model configurations | `uv run python tests/quick_face_test.py photo.jpg` |
| `test_labeler.py` | Test FaceLabeler class | `uv run python tests/test_labeler.py` |
| `diagnose_face_matching.py` | Comprehensive face matching diagnostics | `uv run python tests/diagnose_face_matching.py photo.jpg` |
| `test_streamlit_identify.py` | Test Streamlit identification logic | `uv run python tests/test_streamlit_identify.py` |

### Qdrant Tests

| Script | Purpose | Usage |
|--------|---------|-------|
| `identify_with_qdrant.py` | Test face identification using Qdrant | `uv run python tests/identify_with_qdrant.py photo.jpg` |
| `quick_view_qdrant.py` | View Qdrant database status | `uv run python tests/quick_view_qdrant.py` |
| `view_qdrant.py` | Detailed Qdrant database viewer | `uv run python tests/view_qdrant.py` |

### AI/Caption Tests

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_caption_generator.py` | Test caption generation | `uv run python tests/test_caption_generator.py` |
| `test_ollama_connection.py` | Test Ollama connection | `uv run python tests/test_ollama_connection.py` |
| `test_llava_custom_image.py` | Test LLaVA with custom images | `uv run python tests/test_llava_custom_image.py` |
| `test_llava_vllm.py` | Test LLaVA via vLLM | `uv run python tests/test_llava_vllm.py` |

### System Tests

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_installation.py` | Verify installation | `uv run python tests/test_installation.py` |
| `test_heic.py` | Test HEIC image support | `uv run python tests/test_heic.py` |
| `check_deepface_models.py` | Check available DeepFace models | `uv run python tests/check_deepface_models.py` |
| `check_vggface.py` | Check VGG-Face model | `uv run python tests/check_vggface.py` |

## 🔧 Utility Scripts

### Database Management

| Script | Purpose | Usage |
|--------|---------|-------|
| `rebuild_face_db.py` | Rebuild face database with VGG-Face | `uv run python tests/rebuild_face_db.py` |

### Other

| Script | Purpose | Usage |
|--------|---------|-------|
| `simple_download.py` | Simple download utility | `uv run python tests/simple_download.py` |

## 🚀 Quick Tests

### Verify Everything Works

```bash
# 1. Check installation
uv run python tests/test_installation.py

# 2. View Qdrant database
uv run python tests/quick_view_qdrant.py

# 3. Test face matching
uv run python tests/identify_with_qdrant.py ~/personal_photos/IMG_0276_2.jpeg
```

Expected output from test #3:
```
✅ MATCHED: Appa
✅ MATCHED: Ganesh Sankar
✅ MATCHED: Meena
✅ MATCHED: Amma
✅ MATCHED: sankar
```

### Check GPU Detection

```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 🐛 Troubleshooting Tests

### Face Matching Issues

**Run diagnostics**:
```bash
uv run python tests/diagnose_face_matching.py ~/photos/test.jpg
```

**Test different configurations**:
```bash
uv run python tests/quick_face_test.py ~/photos/test.jpg
```

**Simple test**:
```bash
uv run python tests/simple_face_test.py ~/photos/test.jpg
```

### Qdrant Issues

**View database**:
```bash
uv run python tests/quick_view_qdrant.py
```

**Detailed view**:
```bash
uv run python tests/view_qdrant.py
```

### Caption Generation Issues

**Test Ollama connection**:
```bash
uv run python tests/test_ollama_connection.py
```

**Test caption generation**:
```bash
uv run python tests/test_caption_generator.py
```

## 📊 Test Results Archive

All tests should pass with current configuration:
- **Model**: VGG-Face
- **Detector**: RetinaFace
- **Distance Threshold**: 0.25
- **Qdrant**: sapphire:6333

## 📝 Notes

- Most test scripts require a photo path as argument
- Default test photo: `~/personal_photos/IMG_0276_2.jpeg`
- Some tests may take 10-30 seconds (model loading)
- GPU will be used automatically for neural network operations

## 🔗 Related Documentation

- [Main README](../README.md)
- [Technical Docs](../docs/technical/)
- [Troubleshooting Guide](../docs/technical/FACE_MATCHING_FIX.md)

---

**Last Updated**: November 15, 2025

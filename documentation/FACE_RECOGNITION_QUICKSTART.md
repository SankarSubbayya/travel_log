# Face Recognition Quick Start Guide

## Overview

Your Travel Log project now has comprehensive face recognition capabilities powered by DeepFace. Here's everything you need to know to get started quickly.

## What's Been Built

### 4 Core Modules

1. **`face_detector.py`** - Extract faces from group photos
2. **`face_embeddings.py`** - Generate signature vectors for faces
3. **`face_labeler.py`** - Identify and label faces
4. **`face_manager.py`** - High-level orchestrator for complete workflows

### 4 Example Scripts

1. **`face_detection_example.py`** - Basic detection demo
2. **`face_labeling_example.py`** - Recognition demo
3. **`face_embeddings_example.py`** - Embeddings demo
4. **`complete_workflow_example.py`** - End-to-end demo

### Documentation

- **`docs/face-recognition-guide.md`** - Comprehensive guide (400+ lines)
- **`examples/README.md`** - Example documentation
- **`README.md`** - Updated with face recognition features

## Installation

```bash
# Install all dependencies (including DeepFace, OpenCV, etc.)
uv sync

# Verify installation
uv run python -c "from travel_log import TravelLogFaceManager; print('✓ Ready!')"
```

**Note**: If you see TensorFlow warnings like `[mutex.cc : 452] RAW: Lock blocking`, don't worry! These are harmless and already suppressed in the modules. See `TENSORFLOW_WARNING_FIX.md` if needed.

## 30-Second Demo

```python
from travel_log import TravelLogFaceManager

# Create manager
manager = TravelLogFaceManager("demo_workspace")

# Process a photo (extracts faces, generates embeddings)
result = manager.process_photo("my_photo.jpg")

# See results
print(f"Found {result['num_faces']} faces!")
```

## 5-Minute Setup

### Step 1: Extract Faces from Photos

```python
from travel_log import FaceDetector

detector = FaceDetector(detector_backend='mtcnn')
faces = detector.save_extracted_faces(
    "group_photo.jpg",
    "extracted_faces"
)
```

### Step 2: Build Face Database

```python
from travel_log import FaceLabeler

labeler = FaceLabeler("face_database")

# Add people (use 2-3 photos per person)
labeler.add_person("Alice", ["alice1.jpg", "alice2.jpg"])
labeler.add_person("Bob", ["bob1.jpg", "bob2.jpg"])
```

### Step 3: Identify Faces

```python
# Identify a face
result = labeler.identify_face("unknown_face.jpg")

if result:
    print(f"This is {result['name']}!")
    print(f"Confidence: {result['confidence']:.1%}")
else:
    print("Unknown person")
```

### Step 4: Generate Embeddings

```python
from travel_log import FaceEmbeddings

embedder = FaceEmbeddings(model_name='Facenet512')

# Generate embedding (512-dimensional vector)
emb = embedder.generate_embedding("face.jpg")
print(f"Generated {emb['dimension']}-d embedding")

# Compare faces
similarity = embedder.compute_similarity(
    emb1['embedding'],
    emb2['embedding'],
    metric='cosine'
)
print(f"Similarity: {similarity:.2f}")
```

## Complete Workflow

```python
from travel_log import TravelLogFaceManager

# Initialize
manager = TravelLogFaceManager("my_trip_workspace")

# Add known people to database
manager.add_person_to_database(
    "Alice",
    ["samples/alice/1.jpg", "samples/alice/2.jpg"]
)

# Process all photos in directory
results = manager.process_directory("trip_photos")

# Cluster similar faces
clusters = manager.get_face_clusters(threshold=0.6)

# Generate summary
summary = manager.generate_summary_report()
print(f"""
Trip Summary:
  - Extracted: {summary['extracted_faces']} faces
  - Identified: {summary['identified_faces']} faces
  - Unknown: {summary['unknown_faces']} faces
  - Database: {summary['database']['total_people']} people
""")

# Export organized by person
manager.export_labeled_dataset("organized_by_person")
```

## Key Concepts

### 1. Face Detection
**What**: Find faces in images and extract them
**Backend Options**: `mtcnn` (recommended), `retinaface`, `ssd`, `opencv`, `dlib`

### 2. Face Recognition
**What**: Identify who a person is
**Models**: `Facenet512` (recommended), `ArcFace`, `VGG-Face`, `OpenFace`

### 3. Face Embeddings
**What**: Convert face to numerical signature vector
**Dimensions**: 128-4096 depending on model
**Use**: Similarity search, clustering, matching

### 4. Face Labeling
**What**: Assign names to faces automatically
**Requires**: Database of known people with sample images

## File Structure After Processing

```
workspace/
├── extracted_faces/          # All extracted face images
│   ├── face_photo1_000.jpg
│   ├── face_photo1_001.jpg
│   └── ...
├── embeddings/               # Face embeddings (vectors)
│   ├── face_photo1_000.npy
│   ├── face_photo1_001.npy
│   └── database.pkl
├── face_database/            # Known people database
│   ├── alice/
│   │   ├── alice_0.jpg
│   │   └── alice_1.jpg
│   ├── bob/
│   │   └── bob_0.jpg
│   └── ...
└── results/                  # Reports and summaries
    ├── batch_report_20251017.json
    └── summary_20251017.json
```

## Common Tasks

### Task: Extract all faces from group photos

```python
detector = FaceDetector()
batch_results = detector.batch_extract_faces(
    ["photo1.jpg", "photo2.jpg", "photo3.jpg"],
    "extracted_faces"
)
```

### Task: Find all photos of a specific person

```python
labeler = FaceLabeler("face_database")
results = labeler.identify_faces_batch(all_face_images)
alice_photos = [
    r['query_image'] for r in results
    if r and r['name'] == 'Alice'
]
```

### Task: Check if two faces are the same person

```python
labeler = FaceLabeler("face_database")
result = labeler.verify_faces("face1.jpg", "face2.jpg")
if result['verified']:
    print("Same person!")
```

### Task: Find similar faces

```python
embedder = FaceEmbeddings()
emb1 = embedder.generate_embedding("query_face.jpg")

# Compare with candidates
embeddings = embedder.generate_embeddings_batch(candidate_faces)
similar = embedder.find_most_similar(
    emb1['embedding'],
    embeddings,
    top_k=5
)
```

### Task: Cluster unknown faces

```python
manager = TravelLogFaceManager("workspace")
manager.process_directory("photos")
clusters = manager.get_face_clusters(threshold=0.6)

# Clusters contain groups of similar faces
for cluster_id, faces in clusters.items():
    print(f"{cluster_id}: {len(faces)} faces")
```

## Performance Tips

### For Speed
- Use `detector_backend='ssd'` or `'opencv'`
- Use `model_name='OpenFace'` or `'SFace'`
- Process in batches
- Reduce image resolution

### For Accuracy
- Use `detector_backend='mtcnn'` or `'retinaface'`
- Use `model_name='Facenet512'` or `'ArcFace'`
- Add 3-5 sample images per person
- Use high-quality, well-lit photos

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No faces detected | Try different backend: `detector_backend='retinaface'` |
| Poor recognition | Add more sample images, try `model_name='ArcFace'` |
| Slow processing | Use `detector_backend='ssd'`, `model_name='SFace'` |
| Out of memory | Process in smaller batches, use lighter model |
| False matches | Increase recognition threshold, add more samples |

## Next Steps

1. **Try the examples**: Start with `examples/face_detection_example.py`
2. **Build your database**: Add 2-3 photos of each person you want to recognize
3. **Process test photos**: Try a small batch first
4. **Tune parameters**: Experiment with different backends and models
5. **Scale up**: Process your entire photo collection
6. **Read the guide**: Check `docs/face-recognition-guide.md` for details

## Weekly Checkins with Chander and Asif

Use your 30-minute weekly meetings to discuss:

✅ **Completed Implementation**
- ✓ Face detection from group photos
- ✓ Face separation and extraction
- ✓ Face labeling with confidence scores
- ✓ Face embedding generation (signatures)

🤔 **Questions to Discuss**
1. Which detection backend works best for your photo types?
2. How to improve recognition accuracy for challenging photos?
3. Best practices for database management (how many samples, quality)?
4. How to tune clustering thresholds for your specific use case?
5. Integration with other travel log features?
6. Handling edge cases (poor lighting, occlusions, profiles)?
7. Performance optimization for large collections?

## Resources

- **Comprehensive Guide**: `docs/face-recognition-guide.md`
- **Examples**: `examples/` directory
- **API Reference**: Check module docstrings
- **DeepFace Docs**: https://github.com/serengil/deepface

## Summary

You now have a complete face recognition system that can:
- ✅ Detect and extract faces from group photos
- ✅ Recognize and label people automatically
- ✅ Generate embeddings for advanced analysis
- ✅ Cluster similar faces
- ✅ Process entire photo collections
- ✅ Export organized results

**Start with**: `examples/complete_workflow_example.py`

**Read next**: `docs/face-recognition-guide.md`

**Ask Chander and Asif**: Any questions about implementation or usage!


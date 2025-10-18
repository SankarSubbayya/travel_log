# Face Recognition Examples

This directory contains example scripts demonstrating the face recognition capabilities of Travel Log.

## Examples Overview

### 1. `face_detection_example.py`
Basic face detection and extraction from group photos.

**What it demonstrates:**
- Extracting faces from a group photo
- Saving extracted faces to disk
- Annotating images with face bounding boxes
- Getting face location coordinates

**Run it:**
```bash
python examples/face_detection_example.py
```

### 2. `face_labeling_example.py`
Face recognition and automatic labeling.

**What it demonstrates:**
- Creating a face database with known people
- Identifying faces automatically
- Batch face identification
- Generating label mapping CSV files
- Face verification (comparing two faces)
- Database statistics

**Run it:**
```bash
python examples/face_labeling_example.py
```

### 3. `face_embeddings_example.py`
Face embeddings and similarity search.

**What it demonstrates:**
- Generating face embeddings (signature vectors)
- Computing similarity between faces
- Finding similar faces
- Saving and loading embeddings
- Creating embedding databases
- Pairwise similarity analysis

**Run it:**
```bash
python examples/face_embeddings_example.py
```

### 4. `complete_workflow_example.py`
Complete end-to-end workflow for processing travel photos.

**What it demonstrates:**
- Complete photo processing pipeline
- Building and managing face database
- Processing entire photo directories
- Face clustering
- Generating summary reports
- Exporting organized results

**Run it:**
```bash
python examples/complete_workflow_example.py
```

## Before Running

### 1. Install Dependencies

```bash
uv sync
```

### 2. Prepare Your Photos

Create a directory structure like this:

```
photos/
├── group_photo1.jpg
├── group_photo2.jpg
└── group_photo3.jpg

samples/
├── alice/
│   ├── photo1.jpg
│   └── photo2.jpg
├── bob/
│   ├── photo1.jpg
│   └── photo2.jpg
└── charlie/
    └── photo1.jpg
```

### 3. Update Paths

Edit the example scripts to point to your actual photo paths:

```python
# Update these paths
input_image = "path/to/your/group_photo.jpg"
photos_dir = "path/to/your/photos"
person1_samples = ["path/to/alice/photo1.jpg", "path/to/alice/photo2.jpg"]
```

## Common Patterns

### Quick Single Photo Processing

```python
from travel_log import TravelLogFaceManager

manager = TravelLogFaceManager("workspace")
result = manager.process_photo("my_photo.jpg")
print(f"Found {result['num_faces']} faces")
```

### Quick Face Recognition Setup

```python
from travel_log import FaceLabeler

labeler = FaceLabeler("face_database")
labeler.add_person("Alice", ["alice1.jpg", "alice2.jpg"])

result = labeler.identify_face("unknown_face.jpg")
print(f"Identified as: {result['name']}")
```

### Quick Embedding Generation

```python
from travel_log import generate_face_embedding

embedding = generate_face_embedding("face.jpg")
print(f"Generated {len(embedding)}-dimensional embedding")
```

## Example Outputs

### Face Detection Output
```
Extracting faces from: group_photo.jpg
Detected 5 face(s) in group_photo.jpg
Saved face to: extracted_faces/face_group_photo_000.jpg
Saved face to: extracted_faces/face_group_photo_001.jpg
...
```

### Face Labeling Output
```
Face identified as: Alice
Confidence: 87.5%
Distance: 0.1234

face_001.jpg: Alice (confidence: 87.5%)
face_002.jpg: Bob (confidence: 92.3%)
face_003.jpg: Unknown
```

### Embeddings Output
```
Generated embedding:
  Dimension: 512
  Model: Facenet512
  Vector shape: (512,)

Top 3 similar faces:
  1. face_002.jpg (similarity: 0.9234)
  2. face_005.jpg (similarity: 0.8765)
  3. face_008.jpg (similarity: 0.8456)
```

### Complete Workflow Output
```
============================================================
Travel Log Face Processing - Complete Workflow
============================================================

Workspace initialized at: travel_log_workspace

============================================================
Step 1: Building Face Database
============================================================
Added Alice: 2 samples, 2 embeddings
Added Bob: 2 samples, 2 embeddings

============================================================
Step 2: Processing Single Photo
============================================================
Processed: photos/beach_group.jpg
Found 4 faces:
  Face 0:
    Identified as: Alice
    Confidence: 89.23%
    Has embedding: True
  Face 1:
    Identified as: Bob
    Confidence: 91.45%
    Has embedding: True
  Face 2:
    Identified as: unknown
    Has embedding: True
  Face 3:
    Identified as: Alice
    Confidence: 85.67%
    Has embedding: True

============================================================
✅ Workflow Complete!
============================================================
```

## Tips

1. **Start Simple**: Begin with `face_detection_example.py`
2. **Build Database**: Add clear, frontal face photos to your database
3. **Test Small**: Process a few photos before scaling up
4. **Tune Parameters**: Experiment with different backends and models
5. **Check Results**: Review extracted faces and identifications
6. **Iterate**: Refine your database based on results

## Troubleshooting

### "No faces detected"
- Try a different detection backend
- Check photo quality and lighting
- Ensure faces are clearly visible

### "Poor recognition accuracy"
- Add more sample images per person
- Use better quality sample photos
- Try a different recognition model

### "Processing is slow"
- Use faster backend (ssd or opencv)
- Process in smaller batches
- Use faster model (OpenFace or SFace)

## Next Steps

1. Run the examples with your own photos
2. Build your face database with known people
3. Process a test collection of photos
4. Review the comprehensive guide in `docs/face-recognition-guide.md`
5. Integrate face recognition into your travel log workflows

## Support

For questions and clarifications:
- Use weekly check-ins with Chander and Asif
- Review the complete guide: `docs/face-recognition-guide.md`
- Check DeepFace documentation: https://github.com/serengil/deepface


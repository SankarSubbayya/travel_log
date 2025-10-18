# Face Recognition Guide for Travel Log

## Overview

The Travel Log project includes comprehensive face detection, recognition, labeling, and embedding capabilities powered by DeepFace. This guide will help you get started with processing faces in your travel photos.

## Features

1. **Face Detection & Extraction**: Automatically detect and extract faces from group photos
2. **Face Recognition**: Identify known people in photos
3. **Face Labeling**: Automatically label faces with names
4. **Face Embeddings**: Generate signature vectors for face similarity and clustering
5. **Face Clustering**: Group similar faces together
6. **Batch Processing**: Process entire photo collections efficiently

## Installation

The required dependencies are already included in `pyproject.toml`:

```bash
uv sync
```

This will install:
- `deepface>=0.0.93`: Core face recognition library
- `opencv-python>=4.8.0`: Image processing
- `pillow>=10.0.0`: Image handling
- `numpy>=1.24.0`: Numerical operations

## Quick Start

### 1. Basic Face Detection

Extract faces from a group photo:

```python
from travel_log import FaceDetector

# Initialize detector
detector = FaceDetector(detector_backend='mtcnn')

# Extract and save faces
faces = detector.save_extracted_faces(
    image_path="group_photo.jpg",
    output_dir="extracted_faces"
)

print(f"Extracted {len(faces)} faces")
```

### 2. Face Recognition

Identify people in photos:

```python
from travel_log import FaceLabeler

# Initialize labeler with your database
labeler = FaceLabeler(database_path="face_database")

# Add people to database
labeler.add_person("Alice", ["alice1.jpg", "alice2.jpg"])
labeler.add_person("Bob", ["bob1.jpg", "bob2.jpg"])

# Identify a face
result = labeler.identify_face("unknown_face.jpg")
if result:
    print(f"Identified as: {result['name']} (confidence: {result['confidence']:.2%})")
```

### 3. Generate Embeddings

Create signature vectors for faces:

```python
from travel_log import FaceEmbeddings

# Initialize embeddings generator
embedder = FaceEmbeddings(model_name='Facenet512')

# Generate embedding
embedding = embedder.generate_embedding("face.jpg")
print(f"Embedding dimension: {embedding['dimension']}")

# Compute similarity
similarity = embedder.compute_similarity(
    embedding1['embedding'],
    embedding2['embedding'],
    metric='cosine'
)
```

### 4. Complete Workflow

Process an entire trip's worth of photos:

```python
from travel_log import TravelLogFaceManager

# Initialize manager
manager = TravelLogFaceManager(workspace_dir="my_trip")

# Add known people
manager.add_person_to_database("Alice", ["alice_samples/1.jpg", "alice_samples/2.jpg"])
manager.add_person_to_database("Bob", ["bob_samples/1.jpg"])

# Process all photos in a directory
results = manager.process_directory("trip_photos")

# Generate summary
summary = manager.generate_summary_report()
print(f"Processed {summary['extracted_faces']} faces")
print(f"Identified {summary['identified_faces']} faces")

# Export organized results
manager.export_labeled_dataset("organized_faces")
```

## Core Modules

### FaceDetector

Detects and extracts faces from images.

**Key Methods:**
- `extract_faces(image_path)`: Extract all faces from an image
- `save_extracted_faces(image_path, output_dir)`: Extract and save faces to disk
- `get_face_locations(image_path)`: Get bounding box coordinates
- `annotate_image(image_path, output_path)`: Draw boxes around detected faces

**Supported Backends:**
- `opencv`: Fast, good for simple cases
- `ssd`: Faster than MTCNN, decent accuracy
- `mtcnn`: **Recommended** - Robust, handles various poses
- `retinaface`: High accuracy, slower
- `dlib`: Classic approach, reliable

### FaceEmbeddings

Generates face embeddings (signature vectors) for similarity and recognition.

**Key Methods:**
- `generate_embedding(image_path)`: Generate embedding for one face
- `generate_embeddings_batch(image_paths)`: Generate multiple embeddings
- `compute_similarity(emb1, emb2, metric)`: Compare two embeddings
- `find_most_similar(query, candidates, top_k)`: Find similar faces
- `save_embeddings(embeddings, path)`: Save embeddings to disk
- `load_embeddings(path)`: Load embeddings from disk

**Supported Models:**
- `VGG-Face`: Classic, reliable (2622-d)
- `Facenet`: Google's model, balanced (128-d)
- `Facenet512`: **Recommended** - Higher accuracy (512-d)
- `OpenFace`: Lightweight (128-d)
- `DeepFace`: Facebook's model (4096-d)
- `ArcFace`: State-of-the-art accuracy (512-d)
- `Dlib`: Classic approach (128-d)
- `SFace`: Optimized for speed (128-d)

### FaceLabeler

Identifies and labels faces by comparing against a database.

**Key Methods:**
- `add_person(name, image_paths)`: Add person to database
- `identify_face(image_path)`: Identify a single face
- `identify_faces_batch(image_paths)`: Identify multiple faces
- `verify_faces(img1, img2)`: Check if two faces are the same person
- `create_label_mapping(faces, output_path)`: Create CSV mapping of labels
- `list_known_people()`: List all people in database
- `get_database_stats()`: Get database statistics

**Database Structure:**
```
face_database/
├── alice/
│   ├── alice_0.jpg
│   ├── alice_1.jpg
├── bob/
│   ├── bob_0.jpg
│   └── bob_1.jpg
└── charlie/
    └── charlie_0.jpg
```

### TravelLogFaceManager

High-level orchestrator that combines all functionality.

**Key Methods:**
- `process_photo(image_path)`: Process a single photo end-to-end
- `process_photos_batch(image_paths)`: Process multiple photos
- `process_directory(photos_dir)`: Process all photos in a directory
- `add_person_to_database(name, samples)`: Add person with embeddings
- `get_face_clusters(threshold)`: Cluster similar faces
- `generate_summary_report()`: Get processing statistics
- `export_labeled_dataset(output_dir)`: Export organized results

**Workspace Structure:**
```
workspace/
├── extracted_faces/      # Extracted face images
├── embeddings/           # Face embeddings
├── face_database/        # Known people database
└── results/              # Reports and summaries
```

## Advanced Usage

### Custom Detection Parameters

```python
detector = FaceDetector(
    detector_backend='mtcnn',
    align=True,              # Align faces before extraction
    expand_percentage=10     # Expand face region by 10%
)
```

### Similarity Metrics

Different metrics for comparing embeddings:

```python
# Cosine similarity (0-1, higher = more similar)
cosine_sim = embedder.compute_similarity(emb1, emb2, metric='cosine')

# Euclidean distance (lower = more similar)
euclidean_dist = embedder.compute_similarity(emb1, emb2, metric='euclidean')

# Normalized Euclidean
euclidean_l2 = embedder.compute_similarity(emb1, emb2, metric='euclidean_l2')
```

### Face Clustering

Group similar faces automatically:

```python
manager = TravelLogFaceManager("workspace")

# Process photos first
manager.process_directory("photos")

# Get clusters (threshold: 0.6 for cosine similarity)
clusters = manager.get_face_clusters(threshold=0.6, metric='cosine')

for cluster_id, faces in clusters.items():
    print(f"{cluster_id}: {len(faces)} faces")
```

### Verification vs Recognition

**Verification**: Are these two images of the same person?
```python
result = labeler.verify_faces("image1.jpg", "image2.jpg")
if result['verified']:
    print("Same person!")
```

**Recognition**: Who is this person?
```python
result = labeler.identify_face("unknown.jpg")
print(f"This is {result['name']}")
```

## Performance Tips

### 1. Choose the Right Backend

- **For speed**: Use `opencv` or `ssd`
- **For accuracy**: Use `mtcnn` or `retinaface`
- **For balance**: Use `mtcnn` (recommended default)

### 2. Choose the Right Model

- **For speed**: Use `OpenFace` or `SFace`
- **For accuracy**: Use `Facenet512` or `ArcFace`
- **For balance**: Use `Facenet512` (recommended default)

### 3. Database Best Practices

- **Add 3-5 sample images per person** for better accuracy
- Use **diverse photos**: different lighting, angles, expressions
- Use **high-quality face images**: clear, well-lit, frontal
- **Update database** periodically with new photos

### 4. Batch Processing

Process multiple images efficiently:

```python
# Don't do this (slow)
for image in images:
    result = manager.process_photo(image)

# Do this instead (faster)
results = manager.process_photos_batch(images)
```

## Troubleshooting

### No Faces Detected

- Try a different backend: `detector_backend='retinaface'`
- Check image quality and lighting
- Ensure faces are visible and not too small

### Poor Recognition Accuracy

- Add more sample images to database
- Use higher-quality sample images
- Try a different model: `model_name='ArcFace'`
- Adjust recognition threshold

### Memory Issues

- Process images in smaller batches
- Use a lighter model: `model_name='OpenFace'`
- Reduce image resolution before processing

### Slow Processing

- Use faster backend: `detector_backend='ssd'`
- Use faster model: `model_name='SFace'`
- Process in parallel (multiple images)

## Example Use Cases

### 1. Organize Trip Photos

```python
manager = TravelLogFaceManager("trip_to_paris")

# Add travel companions
manager.add_person_to_database("Alice", ["alice_samples/1.jpg"])
manager.add_person_to_database("Bob", ["bob_samples/1.jpg"])

# Process all trip photos
manager.process_directory("paris_photos")

# Export organized by person
manager.export_labeled_dataset("organized_by_person")
```

### 2. Find All Photos of a Person

```python
labeler = FaceLabeler("face_database")

# Process and identify all faces
results = labeler.identify_faces_batch(all_face_images)

# Filter for specific person
alice_photos = [
    r['query_image'] for r in results
    if r and r['name'] == 'Alice'
]
```

### 3. Cluster Unknown Faces

```python
embedder = FaceEmbeddings()

# Generate embeddings for all unknown faces
embeddings = embedder.generate_embeddings_batch(unknown_faces)

# Find duplicates/similar faces
for i, emb1 in enumerate(embeddings):
    similar = embedder.find_most_similar(
        emb1['embedding'],
        embeddings[i+1:],
        top_k=5
    )
    # Show potential matches
```

## Integration with Travel Log

The face recognition features integrate seamlessly with the travel log:

```python
from travel_log import TravelLogFaceManager, config

# Use travel log configuration
workspace = Path(config.get('workspace', 'travel_log_workspace'))
manager = TravelLogFaceManager(workspace)

# Process photos from a trip
trip_dir = workspace / "trips" / "paris_2025"
results = manager.process_directory(trip_dir / "photos")

# Generate trip report with face statistics
summary = manager.generate_summary_report()
```

## Questions for Weekly Checkins

Use your 30-minute weekly meetings with Chander and Asif to discuss:

1. **Face Detection**: What detection backend works best for your photo types?
2. **Recognition Accuracy**: How to improve identification accuracy?
3. **Database Management**: Best practices for maintaining the face database?
4. **Clustering**: How to tune clustering thresholds for your use case?
5. **Integration**: How to integrate face recognition with other travel log features?
6. **Performance**: How to optimize processing for large photo collections?
7. **Edge Cases**: How to handle challenging scenarios (poor lighting, occlusions, etc.)?

## Additional Resources

- [DeepFace GitHub](https://github.com/serengil/deepface)
- [DeepFace Documentation](https://github.com/serengil/deepface/tree/master/deepface)
- [Face Recognition Papers](https://paperswithcode.com/task/face-recognition)

## Next Steps

1. **Try the examples**: Start with `examples/face_detection_example.py`
2. **Build your database**: Add photos of people you want to recognize
3. **Process a test set**: Try processing a small set of photos
4. **Tune parameters**: Experiment with different backends and models
5. **Scale up**: Process your full photo collection
6. **Integrate**: Add face features to your travel log workflows


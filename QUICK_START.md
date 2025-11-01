# Quick Start Guide - Face Identification

Get started with face identification in 5 minutes!

## Option 1: Web UI (Easiest)

### Step 1: Start the app
```bash
cd /home/sankar/travel_log
streamlit run app.py
```

### Step 2: Set up database
1. Go to **"💾 Database Management"** tab
2. Enter person's name (e.g., "John")
3. Upload 3-5 photos of that person
4. Click "Add to Database"
5. Repeat for other people

### Step 3: Detect faces
1. Go to **"🔍 Face Detection"** tab
2. Initialize detector (choose backend)
3. Upload a photo
4. Click "Detect Faces"

### Step 4: Identify faces
1. Go to **"🎯 Face Identification"** tab
2. Set database path: `./face_database`
3. Click "Initialize Identification"
4. Click "Identify All Faces"
5. View results with confidence scores!

## Option 2: Command Line (Fast)

### Step 1: Create database structure
```bash
mkdir -p face_database/{John,Jane}
# Add some photos of John
cp ~/photos/john*.jpg face_database/John/
# Add some photos of Jane
cp ~/photos/jane*.jpg face_database/Jane/
```

### Step 2: Verify database
```bash
python manage_face_database.py stats --db-dir ./face_database
```

### Step 3: Extract faces from a photo
```bash
python -c "
from travel_log import FaceDetector
detector = FaceDetector()
detector.save_extracted_faces('./group_photo.jpg', './extracted_faces')
"
```

### Step 4: Identify extracted faces
```bash
python process_extracted_faces.py \
  --faces-dir ./extracted_faces \
  --db-dir ./face_database
```

View results:
```bash
cat identification_results/face_identification_*.csv
```

## Option 3: Programmatic

```python
from travel_log import FaceDetector, FaceLabeler

# Detect faces
detector = FaceDetector()
faces = detector.extract_faces('photo.jpg')

# Identify them
labeler = FaceLabeler(database_path='./face_database')

for face in faces:
    result = labeler.identify_face(face_image_path)
    print(f"Match: {result['identity']}, Confidence: {result['confidence']}")
```

## Common Workflows

### Identify family in vacation photos
```bash
# 1. Setup database once
python manage_face_database.py add-person \
  --db-dir ./face_database \
  --name "Mom" \
  --images ./family_photos/mom*.jpg

python manage_face_database.py add-person \
  --db-dir ./face_database \
  --name "Dad" \
  --images ./family_photos/dad*.jpg

# 2. Process vacation photos
for photo in vacation/*.jpg; do
  python -c "
from travel_log import FaceDetector
detector = FaceDetector()
detector.save_extracted_faces('$photo', './extracted')
  "
done

# 3. Identify all
python process_extracted_faces.py \
  --faces-dir ./extracted \
  --db-dir ./face_database \
  --output ./results
```

### Organize a messy photo collection
```bash
# 1. Create database with known people
python manage_face_database.py add-person \
  --db-dir ./face_database \
  --name "Friend A" \
  --images ./known/friend_a*.jpg

# 2. Extract faces from all photos
for photo in photos/*.jpg; do
  python -c "
from travel_log import FaceDetector
detector = FaceDetector()
detector.save_extracted_faces('$photo', './all_faces')
  "
done

# 3. Identify them
python process_extracted_faces.py \
  --faces-dir ./all_faces \
  --db-dir ./face_database \
  --output ./identified
```

## Key Commands

### Database Management
```bash
# Add person
python manage_face_database.py add-person \
  --db-dir ./face_database \
  --name "Person Name" \
  --images ./photos/*.jpg

# List people
python manage_face_database.py list --db-dir ./face_database

# View stats
python manage_face_database.py stats --db-dir ./face_database

# Remove person
python manage_face_database.py remove-person \
  --db-dir ./face_database \
  --name "Person Name" \
  --confirm
```

### Face Identification
```bash
# Identify extracted faces
python process_extracted_faces.py \
  --faces-dir ./extracted_faces \
  --db-dir ./face_database

# With custom model (faster)
python process_extracted_faces.py \
  --faces-dir ./extracted_faces \
  --db-dir ./face_database \
  --model ArcFace

# With custom threshold
python process_extracted_faces.py \
  --faces-dir ./extracted_faces \
  --db-dir ./face_database \
  --confidence-threshold 0.5
```

## Recommended Setup

### First Time
1. **Start Streamlit app**: `streamlit run app.py`
2. **Add people to database**: Use Database Management tab
3. **Try detection**: Upload a photo and detect faces
4. **Try identification**: Identify faces against your database

### Batch Processing
1. **Organize photos** into folders by person
2. **Build database**: Use `manage_face_database.py`
3. **Extract faces**: Use `FaceDetector.save_extracted_faces()`
4. **Identify**: Use `process_extracted_faces.py`
5. **Review**: Check CSV/JSON results

## Tips

- ✅ Use 3-5 photos per person for best results
- ✅ Use clear, frontal face photos
- ✅ Ensure good lighting
- ✅ Use Facenet512 model (best accuracy)
- ✅ Set confidence threshold to 0.6 (balanced)

- ❌ Don't use blurry or small faces
- ❌ Don't use extreme angles or heavy obstruction
- ❌ Don't expect 100% accuracy
- ❌ Don't rely on single photo per person

## Troubleshooting

**No matches found?**
- Add more photos per person
- Lower confidence threshold to 0.4
- Try different model (ArcFace)

**App crashes?**
- Restart: `streamlit run app.py`
- Check database path exists

**Slow processing?**
- Use ArcFace model (faster)
- Use CPU if GPU is slow
- Batch process in smaller groups

**Accuracy issues?**
- Add more training photos
- Use Facenet512 model (most accurate)
- Check photo quality in database

## Next Steps

For detailed documentation, see [FACE_IDENTIFICATION_GUIDE.md](FACE_IDENTIFICATION_GUIDE.md)

Happy identifying! 🎉

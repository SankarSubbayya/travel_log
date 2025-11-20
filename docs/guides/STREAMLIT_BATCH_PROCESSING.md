# Streamlit App - Batch Processing Guide

## Overview

The enhanced Streamlit app now includes **batch processing** directly in the web interface. No need to use command-line tools anymore - you can process multiple extracted faces at once through the UI!

## New Feature: Batch Processing Tab

The **Face Identification** tab now has two sub-tabs:

### Tab 1: 📸 Single Photo
Process faces detected from a single photo uploaded in the Detection tab.

### Tab 2: 📁 Batch Processing (NEW)
Process a directory of extracted face images all at once.

---

## How to Use Batch Processing

### Step 1: Set Up Database
1. Go to **"💾 Database Management"** tab
2. Add people with their photos
3. This creates your face database

### Step 2: Extract Faces
You have two options:

**Option A: Use Detection Tab**
1. Go to **"🔍 Face Detection"** tab
2. Upload photos and click "Detect Faces"
3. Click "Download All Faces" to get a ZIP file
4. Extract the ZIP to a folder (e.g., `./extracted_faces`)

**Option B: Extract Programmatically**
```python
from travel_log import FaceDetector

detector = FaceDetector()
detector.save_extracted_faces('photo1.jpg', './extracted_faces')
detector.save_extracted_faces('photo2.jpg', './extracted_faces')
```

### Step 3: Initialize Identification Model
1. Go to **"🎯 Face Identification"** tab
2. Set **Database Path** to where your database is (e.g., `./face_database`)
3. Select **Recognition Model** (Facenet512 recommended)
4. Click **"⚙️ Initialize Identification"**
5. Wait for confirmation message

### Step 4: Run Batch Processing
1. Click the **"📁 Batch Processing"** sub-tab
2. Enter the **Extracted Faces Directory** path (e.g., `./extracted_faces`)
3. Adjust **Confidence Threshold** if needed (0.6 is default)
4. Click **"🚀 Start Batch Processing"**
5. Watch the progress as it processes each face

### Step 5: Review Results

The app displays:

**📊 Summary Metrics:**
- **Total Processed** - All faces processed
- **✅ Identified** - Faces matched with high confidence
- **⚠️ Low Confidence** - Matches below threshold
- **❌ No Match** - No suitable match found
- **⚠️ Errors** - Failed to process

**Filter & Analyze:**
- Filter by status (identified, low_confidence, no_match, error)
- Filter by minimum confidence level
- Download results as CSV

**View by Person:**
- Expanders show all faces identified for each person
- See confidence scores for each match

---

## Features

### 1. Progress Tracking
- Real-time progress bar during processing
- Current file being processed displayed
- Processes faces one by one with feedback

### 2. Detailed Results
- Results shown in interactive table
- Face filename, status, match, confidence, distance
- Sortable and filterable columns

### 3. Export Results
- Download results as CSV file
- Compatible with Excel, Google Sheets
- Includes all details for analysis

### 4. Results Grouping
- Results grouped by person name
- Quick overview of who was identified
- See all faces per person

### 5. Flexible Filtering
- Filter by status (success, low confidence, no match)
- Adjust confidence threshold dynamically
- Refilter without reprocessing

---

## Workflow Examples

### Example 1: Identify People in Vacation Photos

```
Step 1: Add family to database (Database tab)
   - Name: "Mom" → Upload 3-5 photos
   - Name: "Dad" → Upload 3-5 photos
   - Name: "Sister" → Upload 3-5 photos

Step 2: Extract faces from vacation photos (Detection tab)
   - Upload vacation_photo_1.jpg → Detect Faces
   - Download extracted faces

Step 3: Initialize identification (Identification tab)
   - Database Path: ./face_database
   - Model: Facenet512
   - Click Initialize

Step 4: Batch process (Batch Processing sub-tab)
   - Extracted Faces Dir: ./extracted_faces
   - Click Start Batch Processing
   - Review results grouped by person
```

### Example 2: Process All Photos from Event

```
Step 1: Extract faces from all event photos
   detector = FaceDetector()
   for photo in event_photos:
       detector.save_extracted_faces(photo, './event_faces')

Step 2: Set up database with attendees
   - Use Database Management tab to add all attendees

Step 3: Run batch identification
   - Point to ./event_faces directory
   - Process all at once
   - Download CSV with results
```

### Example 3: Find Specific Person

```
Step 1: Create database with just one person
   - Add only the person you're looking for

Step 2: Extract faces from a photo collection
   detector = FaceDetector()
   for photo in collection:
       detector.save_extracted_faces(photo, './faces')

Step 3: Run batch identification
   - Look for "Identified" entries in results
   - See all instances where this person appears
```

---

## Settings Explained

### Database Path
- Where your face database is located
- Default: `./face_database`
- Can be absolute or relative path

### Recognition Model
- **Facenet512** (default) - Best accuracy, balanced speed
- **Facenet** - Fast, good accuracy
- **ArcFace** - Very fast, good accuracy
- **VGG-Face** - Highest accuracy, slowest
- **DeepFace**, **OpenFace**, **DeepID**, **Dlib**, **SFace** - Other options

### Confidence Threshold
- Minimum confidence to accept a match
- **0.3-0.5** - Very relaxed, may have false positives
- **0.6** - Balanced (default)
- **0.7-0.8** - Strict, only high confidence matches
- **0.9+** - Very strict, only perfect matches

### Extracted Faces Directory
- Where your extracted face images are
- Default: `./extracted_faces`
- Can point to any folder with face images

---

## Understanding Results

### Status Types

**✅ Identified**
- Face matched in database
- Confidence above threshold
- Person name shown
- Ready to use

**⚠️ Low Confidence**
- Face has a match in database
- But confidence below threshold
- Might be similar but not same person
- Review manually if needed

**❌ No Match**
- No similar face found in database
- Confidence is 0
- Could be unknown person or poor quality

**⚠️ Error**
- Processing failed for this face
- Check file format
- Try re-extracting the face

### Confidence Score
- **0.0 to 1.0** - Probability of match
- **0.9+** - Very confident match
- **0.7-0.8** - High confidence
- **0.5-0.6** - Moderate confidence
- **Below 0.5** - Low confidence

### Distance
- **0.0** - Perfect match
- **0.1-0.3** - Very similar
- **0.3-0.5** - Similar
- **0.5-1.0** - Somewhat similar
- **1.0+** - Very different

---

## Tips & Tricks

### 1. Best Results
- Use 5+ photos per person in database
- Ensure good lighting in all photos
- Use frontal face photos when possible
- Keep database up to date

### 2. Improving Accuracy
- Try Facenet512 or VGG-Face for best accuracy
- Lower confidence threshold gradually
- Add more training photos
- Remove poor quality database photos

### 3. Speed Optimization
- Use ArcFace or Facenet for speed
- Process in batches if you have 1000+ faces
- Use CPU for small batches, GPU for large

### 4. Handling Results
- Export CSV for analysis in Excel
- Filter by person to check all instances
- Review low confidence matches manually
- Create separate databases for different contexts

---

## Keyboard Shortcuts & UI Tips

### Navigation
- Click "📸 Single Photo" to go back to single photo mode
- Click "📁 Batch Processing" to switch to batch mode
- Use expanders to expand/collapse person sections

### Filtering
- Select multiple statuses in filter
- Drag confidence slider to adjust threshold
- Table automatically updates with filters

### Download
- Click "Download CSV" button to save results
- File named: `batch_identification_results.csv`
- Can open in Excel, Google Sheets, Python

---

## Common Issues & Solutions

### "Initialize identification model first!"
**Problem:** You haven't initialized the identification model yet
**Solution:**
1. Set database path
2. Select model
3. Click "⚙️ Initialize Identification"

### "Directory not found"
**Problem:** The path to extracted faces directory is wrong
**Solution:**
1. Verify the directory exists
2. Use absolute path if relative path doesn't work
3. Check spelling

### "No face images found in directory"
**Problem:** The directory is empty or has no image files
**Solution:**
1. Check directory path
2. Make sure faces are extracted (using Detection tab or FaceDetector)
3. Verify file format (.jpg, .png, .bmp, etc.)

### "Low identification accuracy"
**Problem:** Many faces marked as "No Match" or "Low Confidence"
**Solution:**
1. Add more training photos (5+ per person)
2. Improve photo quality in database
3. Lower confidence threshold
4. Try different recognition model (Facenet512, VGG-Face)

### "Processing is slow"
**Problem:** Batch processing takes too long
**Solution:**
1. Use ArcFace or Facenet model (faster)
2. Process smaller batches
3. Check GPU availability
4. Lower the number of faces to process

---

## Comparison: CLI vs Web UI

| Feature | Command-Line | Web UI |
|---------|--------------|--------|
| Ease of Use | Moderate | Easy |
| No Installation | No | Yes |
| Progress Feedback | Minimal | Real-time |
| Result Visualization | Terminal | Interactive |
| Batch Processing | Yes | Yes |
| Export Options | JSON/CSV | CSV (in-browser) |
| Database Management | Full CLI | Full UI |
| Model Selection | Command flag | Dropdown |
| Flexibility | High | Medium |

**Recommendation:** Use Web UI for most tasks, use CLI for automation/scripting.

---

## Advanced Usage

### Programmatic Batch Processing

If you want to use the batch processing function in Python:

```python
import streamlit as st
from pathlib import Path
from travel_log import FaceLabeler

# Load your database
labeler = FaceLabeler(database_path='./face_database', model_name='Facenet512')

# Process batch (simulating the UI function)
from app import process_batch_faces

results = process_batch_faces(
    faces_dir=Path('./extracted_faces'),
    labeler=labeler,
    confidence_threshold=0.6
)

# Access results
print(f"Total: {results['summary']['total']}")
print(f"Identified: {results['summary']['identified']}")
for result in results['results']:
    print(f"{result['face_file']}: {result['match']}")
```

### Integrate with Your Workflow

```python
# Extract faces from photos
from travel_log import FaceDetector

detector = FaceDetector()
for photo in photos:
    detector.save_extracted_faces(photo, './extracted_faces')

# Then open Streamlit app and use Batch Processing tab
# or use the programmatic approach above
```

---

## Performance Notes

### Processing Speed
- **Per face**: 0.1-0.5 seconds (depending on model)
- **100 faces**: 10-50 seconds
- **1000 faces**: 100-500 seconds

### Memory Usage
- **ArcFace/Facenet**: 500MB
- **Facenet512**: 800MB
- **VGG-Face**: 1GB+

### GPU vs CPU
- **GPU**: 3-10x faster (if available)
- **CPU**: Slower but works fine for small batches
- **Automatic**: App detects and uses available GPU

---

## Conclusion

Batch processing in the Streamlit app makes face identification easy and accessible without command-line knowledge. Perfect for:
- Event photos organization
- Family photo collections
- Security applications
- Photo library management

For detailed documentation on all features, see:
- **FACE_IDENTIFICATION_GUIDE.md** - Complete reference
- **QUICK_START.md** - Quick tutorial

Happy batch processing! 🚀

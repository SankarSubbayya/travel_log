#!/usr/bin/env python3
"""
Travel Log Face Recognition - Streamlit Web App

A beautiful, interactive web interface for face detection and recognition.

Usage:
    streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import io
import tempfile
import shutil
import traceback
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from travel_log import (
    FaceDetector,
    FaceLabeler,
    FaceEmbeddings,
    config,
    ensure_compatible_image,
    HEIC_SUPPORTED,
    get_complete_metadata,
    format_gps_for_maps
)

# Page configuration
st.set_page_config(
    page_title="Travel Log Face Recognition",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1E88E5;
        padding-bottom: 1rem;
    }
    .face-card {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'detected_faces' not in st.session_state:
    st.session_state.detected_faces = []
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'image_metadata' not in st.session_state:
    st.session_state.image_metadata = None
if 'labeler' not in st.session_state:
    st.session_state.labeler = None
if 'face_identifications' not in st.session_state:
    st.session_state.face_identifications = {}
if 'current_db_path' not in st.session_state:
    st.session_state.current_db_path = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'batch_progress' not in st.session_state:
    st.session_state.batch_progress = 0

def initialize_detector(backend):
    """Initialize the face detector with specified backend."""
    try:
        with st.spinner(f"Initializing {backend} detector..."):
            detector = FaceDetector(detector_backend=backend)
            st.session_state.detector = detector
            return True
    except Exception as e:
        st.error(f"Error initializing detector: {str(e)}")
        return False

def initialize_labeler(database_path, model_name='Facenet512', distance_metric='cosine'):
    """Initialize the face labeler with a database."""
    try:
        with st.spinner(f"Initializing face recognition with {model_name}..."):
            labeler = FaceLabeler(
                database_path=database_path,
                model_name=model_name,
                distance_metric=distance_metric
            )
            st.session_state.labeler = labeler
            st.session_state.current_db_path = database_path
            return True
    except Exception as e:
        st.error(f"Error initializing face recognition: {str(e)}")
        return False

def process_batch_faces(faces_dir: Path, labeler, confidence_threshold: float = 0.6) -> Dict:
    """
    Process a batch of extracted face images.

    Args:
        faces_dir: Directory containing extracted face images
        labeler: FaceLabeler instance
        confidence_threshold: Minimum confidence for matches

    Returns:
        Dictionary with batch processing results
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    face_files = []

    for ext in supported_extensions:
        face_files.extend(faces_dir.glob(f'*{ext}'))
        face_files.extend(faces_dir.glob(f'*{ext.upper()}'))

    face_files = sorted(face_files)

    if not face_files:
        return {
            'status': 'no_files',
            'message': 'No face images found',
            'results': []
        }

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, face_file in enumerate(face_files):
        try:
            status_text.text(f"Processing {idx + 1}/{len(face_files)}: {face_file.name}")

            # Find matches
            match_results = labeler.find_face(str(face_file))

            if match_results and len(match_results) > 0 and not match_results[0].empty:
                best_match = match_results[0].iloc[0]
                distance = float(best_match['distance'])
                confidence = max(0, 1 - (distance / 2.0))

                # Extract person name from identity path
                person_name = Path(best_match['identity']).parent.name
                
                if confidence >= confidence_threshold:
                    status = 'identified'
                else:
                    status = 'low_confidence'

                results.append({
                    'face_file': face_file.name,
                    'status': status,
                    'match': person_name,
                    'confidence': round(confidence, 4),
                    'distance': round(distance, 4)
                })
            else:
                results.append({
                    'face_file': face_file.name,
                    'status': 'no_match',
                    'match': 'Unknown',
                    'confidence': 0.0,
                    'distance': 2.0
                })

        except Exception as e:
            results.append({
                'face_file': face_file.name,
                'status': 'error',
                'match': 'Error',
                'confidence': 0.0,
                'error': str(e)
            })

        progress_bar.progress((idx + 1) / len(face_files))

    progress_bar.empty()
    status_text.empty()

    # Calculate summary
    summary = {
        'total': len(face_files),
        'identified': len([r for r in results if r['status'] == 'identified']),
        'low_confidence': len([r for r in results if r['status'] == 'low_confidence']),
        'no_match': len([r for r in results if r['status'] == 'no_match']),
        'errors': len([r for r in results if r['status'] == 'error'])
    }

    return {
        'status': 'success',
        'results': results,
        'summary': summary
    }

def identify_faces(face_images: List[Dict], confidence_threshold: float = 0.6) -> Dict:
    """
    Identify detected faces against the database.

    Args:
        face_images: List of face image data
        confidence_threshold: Minimum confidence for matches

    Returns:
        Dictionary with identification results
    """
    if st.session_state.labeler is None:
        return {}

    results = {}

    for idx, face_data in enumerate(face_images):
        try:
            # Save face to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                face_data['image'].save(tmp.name, format='JPEG')
                tmp_path = tmp.name

            # Identify face
            match_results = st.session_state.labeler.find_face(tmp_path)

            # Clean up
            Path(tmp_path).unlink()

            if match_results and len(match_results) > 0 and not match_results[0].empty:
                best_match = match_results[0].iloc[0]
                distance = float(best_match['distance'])

                # Convert distance to confidence
                confidence = max(0, 1 - (distance / 2.0))

                if confidence >= confidence_threshold:
                    person_name = Path(best_match['identity']).parent.name
                    results[idx] = {
                        'match': person_name,
                        'confidence': confidence,
                        'distance': distance,
                        'matched_image': best_match['identity']
                    }
                else:
                    results[idx] = {
                        'match': 'Unknown',
                        'confidence': confidence,
                        'distance': distance,
                        'status': 'low_confidence'
                    }
            else:
                results[idx] = {
                    'match': 'Unknown',
                    'confidence': 0.0,
                    'status': 'no_match'
                }

        except Exception as e:
            results[idx] = {
                'match': 'Error',
                'error': str(e)
            }

    return results

def process_image(image_file, detector, min_confidence=0.9):
    """Process uploaded image and detect faces."""
    try:
        # Determine file extension
        file_ext = Path(image_file.name).suffix.lower()
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(image_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract EXIF metadata first (before conversion)
        with st.spinner("Reading photo metadata..."):
            metadata = get_complete_metadata(tmp_path)
        
        # Convert HEIC to JPEG if necessary
        if file_ext in ['.heic', '.heif']:
            with st.spinner("Converting HEIC image..."):
                tmp_path = ensure_compatible_image(tmp_path)
        
        # Extract faces using the correct method
        with st.spinner("Detecting faces..."):
            faces = detector.extract_faces(tmp_path)
            
            # Filter by confidence
            filtered_faces = [f for f in faces if f.get('confidence', 1.0) >= min_confidence]
            
            # Get original image
            img = Image.open(tmp_path)
            
            # Get face images from the extracted faces
            face_images = []
            for i, face in enumerate(filtered_faces):
                # Get facial area coordinates
                facial_area = face.get('facial_area', {})
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 0)
                h = facial_area.get('h', 0)
                
                # Crop face from original image
                face_img = img.crop((x, y, x + w, y + h))
                face_images.append({
                    'image': face_img,
                    'location': (x, y, w, h),
                    'confidence': face.get('confidence', 1.0)
                })
            
            # Clean up
            Path(tmp_path).unlink()
            
            return {
                'original_image': img,
                'faces': face_images,
                'num_faces': len(filtered_faces),
                'metadata': metadata
            }
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.error(f"Details: {traceback.format_exc()}")
        return None

def main():
    # Header
    st.title("📸 Travel Log Face Recognition")
    st.markdown("**Upload photos, detect faces, and identify them**")

    # Create tabs for different features
    detection_tab, identification_tab, database_tab = st.tabs([
        "🔍 Face Detection",
        "🎯 Face Identification",
        "💾 Database Management"
    ])

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Detection backend selection
        backend_options = ['opencv', 'ssd', 'mtcnn', 'retinaface', 'dlib']
        default_backend = config.get('face_detection', {}).get('default_backend', 'mtcnn')
        
        if default_backend not in backend_options:
            default_backend = 'mtcnn'
        
        backend = st.selectbox(
            "Detection Backend",
            backend_options,
            index=backend_options.index(default_backend),
            help="Choose the face detection algorithm"
        )
        
        st.markdown("""
        **Backend Guide:**
        - **opencv**: Fast, good for testing
        - **ssd**: Fast, balanced
        - **mtcnn**: Recommended, accurate
        - **retinaface**: Most accurate, slower
        - **dlib**: Alternative, accurate
        """)
        
        # Confidence threshold
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Filter faces below this confidence threshold"
        )
        
        st.divider()
        
        # Initialize button
        if st.button("🔄 Initialize Detector", type="primary"):
            if initialize_detector(backend):
                st.success(f"✅ {backend} detector initialized!")
        
        st.divider()
        
        # Info
        st.info("💡 **Tip:** Initialize the detector before uploading images.")
        
        # Statistics
        if st.session_state.detected_faces:
            st.divider()
            st.header("📊 Statistics")
            st.metric("Faces Detected", len(st.session_state.detected_faces))
    
    # ============================================================================
    # TAB 1: FACE DETECTION
    # ============================================================================
    with detection_tab:
        # Main content
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("📤 Upload Image")

            # Determine supported file types
            supported_types = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
            if HEIC_SUPPORTED:
                supported_types.extend(['heic', 'heif'])

            help_text = "Upload a photo to detect faces"
            if HEIC_SUPPORTED:
                help_text += " (supports JPEG, PNG, HEIC, and more)"

            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=supported_types,
                help=help_text
            )

            if uploaded_file is not None:
                # Display uploaded image
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

                # Process button
                if st.button("🔍 Detect Faces", type="primary"):
                    if st.session_state.detector is None:
                        st.warning("⚠️ Please initialize the detector first!")
                    else:
                        result = process_image(
                            uploaded_file,
                            st.session_state.detector,
                            min_confidence
                        )

                        if result:
                            st.session_state.detected_faces = result['faces']
                            st.session_state.processed_image = result['original_image']
                            st.session_state.image_metadata = result['metadata']
                            st.success(f"✅ Detected {result['num_faces']} face(s)!")
                            st.rerun()

        with col2:
            st.header("👤 Detected Faces")

            # Display EXIF metadata if available
            if st.session_state.image_metadata:
                with st.expander("📸 Photo Information", expanded=False):
                    metadata = st.session_state.image_metadata

                    col_meta1, col_meta2 = st.columns(2)

                    with col_meta1:
                        # Date and Time
                        if 'datetime_str' in metadata:
                            st.markdown("**📅 Date & Time**")
                            st.write(f"📆 {metadata['datetime_str']}")
                            if 'datetime' in metadata:
                                dt = metadata['datetime']
                                st.write(f"📍 {dt.strftime('%A, %B %d, %Y')}")
                                st.write(f"🕐 {dt.strftime('%I:%M:%S %p')}")

                        # Dimensions
                        if 'width' in metadata and 'height' in metadata:
                            st.markdown("**📐 Dimensions**")
                            st.write(f"{metadata['width']} × {metadata['height']} pixels")
                            st.write(f"Orientation: {metadata.get('orientation', 'N/A')}")

                    with col_meta2:
                        # GPS Location
                        if 'latitude' in metadata and 'longitude' in metadata:
                            st.markdown("**🗺️ Location**")
                            st.write(f"📍 {metadata['latitude']:.6f}°, {metadata['longitude']:.6f}°")
                            if 'altitude' in metadata:
                                st.write(f"⛰️ Altitude: {metadata['altitude']:.1f}m")

                            # Map links
                            maps = format_gps_for_maps(metadata['latitude'], metadata['longitude'])
                            st.markdown(f"[🗺️ Open in Google Maps]({maps['google_maps']})")

                        # Camera Info
                        if 'camera' in metadata and metadata['camera']:
                            st.markdown("**📷 Camera**")
                            camera = metadata['camera']
                            if 'camera_make' in camera and 'camera_model' in camera:
                                st.write(f"{camera['camera_make']} {camera['camera_model']}")
                            if 'iso' in camera:
                                st.write(f"ISO: {camera['iso']}")
                            if 'aperture' in camera:
                                st.write(f"Aperture: {camera['aperture']}")

            if st.session_state.detected_faces:
                st.success(f"Found {len(st.session_state.detected_faces)} face(s)")

                # Display faces in a grid
                num_cols = min(3, len(st.session_state.detected_faces))

                for idx in range(0, len(st.session_state.detected_faces), num_cols):
                    cols = st.columns(num_cols)

                    for i, col in enumerate(cols):
                        face_idx = idx + i
                        if face_idx < len(st.session_state.detected_faces):
                            face_data = st.session_state.detected_faces[face_idx]

                            with col:
                                st.image(
                                    face_data['image'],
                                    caption=f"Face {face_idx + 1}",
                                    use_container_width=True
                                )

                                # Display confidence
                                confidence = face_data['confidence']
                                st.progress(confidence, text=f"Confidence: {confidence:.1%}")

                                # Download button for individual face
                                buf = io.BytesIO()
                                face_data['image'].save(buf, format='JPEG')
                                buf.seek(0)

                                st.download_button(
                                    label=f"⬇️ Download",
                                    data=buf,
                                    file_name=f"face_{face_idx + 1}.jpg",
                                    mime="image/jpeg",
                                    key=f"download_{face_idx}"
                                )

                # Bulk download option
                st.divider()

                if st.button("📦 Download All Faces"):
                    # Create a zip file with all faces
                    import zipfile

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for idx, face_data in enumerate(st.session_state.detected_faces):
                            img_buffer = io.BytesIO()
                            face_data['image'].save(img_buffer, format='JPEG')
                            img_buffer.seek(0)
                            zip_file.writestr(f"face_{idx + 1}.jpg", img_buffer.getvalue())

                    zip_buffer.seek(0)

                    st.download_button(
                        label="⬇️ Download ZIP",
                        data=zip_buffer,
                        file_name="extracted_faces.zip",
                        mime="application/zip"
                    )
            else:
                st.info("👈 Upload an image and click 'Detect Faces' to see results here.")

        # Additional features section
        st.divider()

        with st.expander("ℹ️ About Face Detection"):
            st.markdown("""
            ### How it works

            This application uses state-of-the-art deep learning models to detect and extract faces from photos:

            1. **Upload** your photo (JPEG or PNG)
            2. **Select** a detection backend that suits your needs
            3. **Detect** faces automatically
            4. **Download** extracted faces individually or as a ZIP file

            ### Detection Backends

            - **OpenCV Haar Cascade**: Fast, classical method, good for frontal faces
            - **SSD (Single Shot Detector)**: Fast, deep learning-based, balanced performance
            - **MTCNN**: Multi-task CNN, highly accurate, recommended for most use cases
            - **RetinaFace**: State-of-the-art, most accurate, slower processing
            - **Dlib HOG**: Histogram of Oriented Gradients, good alternative

            ### Features

            - ✅ Multiple detection backends
            - ✅ Confidence filtering
            - ✅ Real-time preview
            - ✅ Individual face downloads
            - ✅ Bulk ZIP download
            - ✅ GPU acceleration (if available)

            ### Tips

            - Use **MTCNN** for best balance of speed and accuracy
            - Use **RetinaFace** for maximum accuracy
            - Use **OpenCV** for fastest processing
            - Adjust confidence threshold to filter low-quality detections
            """)

        with st.expander("🛠️ Technical Details"):
            st.markdown("""
            ### System Information

            **Framework**: DeepFace + TensorFlow
            **Backend**: Configurable (OpenCV, MTCNN, RetinaFace, etc.)
            **GPU Support**: Automatic detection
            **Models**: Automatically downloaded on first use

            ### Configuration

            The app uses settings from `config.yaml`:
            - Default detection backend
            - Default recognition model
            - Output directories

            ### Performance

            - **With GPU**: ~0.1-0.5s per face
            - **CPU only**: ~1-3s per face
            - **First run**: Models download automatically (~100-500MB)
            """)

            # Show current config
            st.code(f"""
Current Configuration:
- Detection Backend: {backend}
- Min Confidence: {min_confidence}
- GPU Available: {st.session_state.get('gpu_available', 'Checking...')}
            """)

    # ============================================================================
    # TAB 2: FACE IDENTIFICATION
    # ============================================================================
    with identification_tab:
        st.header("🎯 Face Identification")
        st.markdown("Identify detected faces against a known face database")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Database Setup")

            database_path = st.text_input(
                "Face Database Path",
                value="./face_database",
                help="Path to directory containing labeled face images"
            )

            model_options = ['Facenet512', 'VGG-Face', 'Facenet', 'ArcFace', 'DeepFace',
                           'OpenFace', 'DeepID', 'Dlib', 'SFace']

            recognition_model = st.selectbox(
                "Recognition Model",
                model_options,
                index=0,
                help="Face recognition model to use"
            )

            if st.button("⚙️ Initialize Identification", type="primary"):
                if initialize_labeler(database_path, recognition_model, 'cosine'):
                    st.success(f"✅ Initialized with {recognition_model} model")
                    st.rerun()

            st.markdown("---")

            st.markdown("**Model Performance Guide:**")
            st.markdown("""
            - **Facenet512**: Balanced, recommended (512D embeddings)
            - **Facenet**: Fast, accurate (128D embeddings)
            - **ArcFace**: Very fast, accurate
            - **VGG-Face**: Accurate, requires more computation
            - **DeepFace**: Balanced
            """)

        with col2:
            st.subheader("Identification Settings")

            confidence_threshold = st.slider(
                "Match Confidence Threshold",
                min_value=0.3,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Minimum confidence to accept a match"
            )

            show_top_matches = st.checkbox(
                "Show top 3 matches",
                value=False,
                help="Show all top candidates, not just the best match"
            )

            st.divider()

            # Database status
            if st.session_state.labeler:
                st.success("✅ Recognition model loaded")
                st.info(f"Database: {st.session_state.current_db_path}")

                # Check database content
                db_path = Path(st.session_state.current_db_path)
                if db_path.exists():
                    people = [d.name for d in db_path.iterdir() if d.is_dir()]
                    if people:
                        st.metric("People in Database", len(people))
                        with st.expander("View people in database"):
                            for person in sorted(people):
                                person_dir = db_path / person
                                img_count = len([f for f in person_dir.glob('*') if f.is_file()])
                                st.write(f"• {person} ({img_count} images)")
                    else:
                        st.warning("⚠️ Database is empty. Add people first.")
            else:
                st.warning("⚠️ Initialize identification model first")

        st.divider()

        # Tab sections
        id_method_tab1, id_method_tab2 = st.tabs(["📸 Single Photo", "📁 Batch Processing"])

        with id_method_tab1:
            # Identification section
            if st.session_state.detected_faces and st.session_state.labeler:
                st.subheader("Identify Detected Faces")

                if st.button("🎯 Identify All Faces", type="primary"):
                    with st.spinner("Identifying faces..."):
                        results = identify_faces(st.session_state.detected_faces, confidence_threshold)
                        st.session_state.face_identifications = results

                if st.session_state.face_identifications:
                    st.success(f"Identified {len(st.session_state.face_identifications)} face(s)")

                    # Display results
                    num_cols = min(3, len(st.session_state.detected_faces))

                    for idx in range(0, len(st.session_state.detected_faces), num_cols):
                        cols = st.columns(num_cols)

                        for i, col in enumerate(cols):
                            face_idx = idx + i
                            if face_idx < len(st.session_state.detected_faces):
                                face_data = st.session_state.detected_faces[face_idx]
                                identification = st.session_state.face_identifications.get(face_idx, {})

                                with col:
                                    st.image(
                                        face_data['image'],
                                        caption=f"Face {face_idx + 1}",
                                        use_container_width=True
                                    )

                                    match = identification.get('match', 'Unknown')
                                    confidence = identification.get('confidence', 0)

                                    if match == 'Unknown':
                                        st.warning(f"No match found")
                                    elif match == 'Error':
                                        st.error(f"Error: {identification.get('error', 'Unknown')}")
                                    else:
                                        st.success(f"👤 {match}")
                                        st.progress(
                                            confidence,
                                            text=f"Confidence: {confidence:.1%}"
                                        )

            else:
                if not st.session_state.detected_faces:
                    st.info("👈 Detect faces first in the Face Detection tab")
                else:
                    st.info("👈 Initialize identification model first")

        with id_method_tab2:
            # Batch processing section
            st.subheader("🔄 Batch Process Extracted Faces")
            st.markdown("Identify all faces in a directory at once")

            batch_faces_dir = st.text_input(
                "Extracted Faces Directory",
                value="./extracted_faces",
                help="Directory containing extracted face images"
            )

            if st.button("🚀 Start Batch Processing", type="primary"):
                if not st.session_state.labeler:
                    st.error("❌ Initialize identification model first!")
                else:
                    batch_path = Path(batch_faces_dir)
                    if not batch_path.exists():
                        st.error(f"❌ Directory not found: {batch_faces_dir}")
                    else:
                        with st.spinner("Processing batch..."):
                            batch_result = process_batch_faces(
                                batch_path,
                                st.session_state.labeler,
                                confidence_threshold
                            )
                            st.session_state.batch_results = batch_result

            # Display batch results
            if st.session_state.batch_results:
                result = st.session_state.batch_results

                if result['status'] == 'no_files':
                    st.warning("⚠️ No face images found in directory")
                else:
                    summary = result['summary']

                    # Summary metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Processed", summary['total'])
                    with col2:
                        st.metric("✅ Identified", summary['identified'])
                    with col3:
                        st.metric("⚠️ Low Confidence", summary['low_confidence'])
                    with col4:
                        st.metric("❌ No Match", summary['no_match'])
                    with col5:
                        st.metric("⚠️ Errors", summary['errors'])

                    st.divider()

                    # Results table
                    st.subheader("📊 Detailed Results")

                    # Create dataframe for display
                    import pandas as pd
                    df = pd.DataFrame(result['results'])

                    # Filter options
                    col_f1, col_f2, col_f3 = st.columns(3)
                    with col_f1:
                        filter_status = st.multiselect(
                            "Filter by Status",
                            options=['identified', 'low_confidence', 'no_match', 'error'],
                            default=['identified', 'low_confidence', 'no_match']
                        )
                    with col_f2:
                        show_confidence_min = st.slider(
                            "Min Confidence to Show",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.1
                        )
                    with col_f3:
                        if st.button("📥 Download Results"):
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv_data,
                                file_name="batch_identification_results.csv",
                                mime="text/csv"
                            )

                    # Filter and display
                    filtered_df = df[df['status'].isin(filter_status)]
                    filtered_df = filtered_df[filtered_df['confidence'] >= show_confidence_min]

                    st.dataframe(
                        filtered_df,
                        use_container_width=True,
                        height=400,
                        hide_index=True
                    )

                    # Identified faces section
                    if summary['identified'] > 0:
                        st.divider()
                        st.subheader(f"✅ Identified Faces ({summary['identified']})")

                        identified_results = [r for r in result['results'] if r['status'] == 'identified']

                        # Group by person
                        from collections import defaultdict
                        by_person = defaultdict(list)
                        for r in identified_results:
                            by_person[r['match']].append(r)

                        for person_name in sorted(by_person.keys()):
                            with st.expander(f"👤 {person_name} ({len(by_person[person_name])} faces)"):
                                person_faces = by_person[person_name]
                                cols = st.columns(min(3, len(person_faces)))

                                for idx, face_result in enumerate(person_faces):
                                    with cols[idx % 3]:
                                        st.write(f"**{face_result['face_file']}**")
                                        st.metric(
                                            "Confidence",
                                            f"{face_result['confidence']:.1%}"
                                        )

            else:
                st.info("Upload a directory with extracted faces and click 'Start Batch Processing'")

    # ============================================================================
    # TAB 3: DATABASE MANAGEMENT
    # ============================================================================
    with database_tab:
        st.header("💾 Face Database Management")
        st.markdown("Manage the face database for identification")

        db_path = st.text_input(
            "Database Directory",
            value="./face_database",
            help="Path to face database"
        )

        db_path = Path(db_path)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.subheader("📊 Database Stats")
            if db_path.exists():
                people = [d.name for d in db_path.iterdir() if d.is_dir()]
                total_images = sum(
                    len([f for f in (db_path / p).glob('*') if f.is_file()])
                    for p in people
                )

                st.metric("Total People", len(people))
                st.metric("Total Images", total_images)

                if total_images > 0:
                    st.metric("Avg per Person", round(total_images / len(people), 1) if people else 0)

                # Calculate database size
                import os
                def get_size(path):
                    total = 0
                    for dirpath, dirnames, filenames in os.walk(path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            total += os.path.getsize(filepath)
                    return total

                size_mb = get_size(db_path) / (1024 * 1024)
                st.metric("Database Size", f"{size_mb:.2f} MB")

        with col2:
            st.subheader("➕ Add Person")

            person_name = st.text_input("Person Name")

            uploaded_images = st.file_uploader(
                "Upload face images",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True
            )

            if st.button("Add to Database"):
                if person_name and uploaded_images:
                    person_dir = db_path / person_name
                    person_dir.mkdir(parents=True, exist_ok=True)

                    count = 0
                    for img_file in uploaded_images:
                        img_path = person_dir / img_file.name
                        with open(img_path, 'wb') as f:
                            f.write(img_file.getvalue())
                        count += 1

                    st.success(f"✅ Added {count} images for {person_name}")
                    st.rerun()
                else:
                    st.warning("⚠️ Enter name and select images")

        with col3:
            st.subheader("👥 People List")

            if db_path.exists():
                people = sorted([d.name for d in db_path.iterdir() if d.is_dir()])

                if people:
                    for person in people:
                        person_dir = db_path / person
                        img_count = len([f for f in person_dir.glob('*') if f.is_file()])

                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"👤 **{person}** ({img_count} images)")
                        with col_b:
                            if st.button("🗑️", key=f"delete_{person}"):
                                st.warning(f"Delete {person}?")
                                col_c, col_d = st.columns(2)
                                with col_c:
                                    if st.button("Confirm Delete", key=f"confirm_{person}"):
                                        shutil.rmtree(person_dir)
                                        st.success(f"Deleted {person}")
                                        st.rerun()

                else:
                    st.info("No people in database yet")

        st.divider()
        st.subheader("📁 Database Info")

        st.markdown("""
        **Database Structure:**
        ```
        face_database/
        ├── person1/
        │   ├── photo1.jpg
        │   ├── photo2.jpg
        ├── person2/
        │   ├── photo1.jpg
        └── unknown/
            ├── unknown_face1.jpg
        ```

        **Tips:**
        - Organize faces by person directory
        - Each person should have 3+ sample images
        - Use clear, frontal face photos
        - Ensure good lighting and image quality
        """)

if __name__ == "__main__":
    main()


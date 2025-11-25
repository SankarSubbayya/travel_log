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
import numpy as np
import hashlib
from datetime import datetime

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
from travel_log.caption_generator import CaptionGenerator

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
    # Auto-initialize with VGG-Face model (recommended for GPU)
    try:
        from travel_log import FaceLabeler
        st.session_state.labeler = FaceLabeler(
            database_path="face_database",
            model_name="VGG-Face",
            detector_backend="retinaface",  # Match detector used for face detection
            distance_metric="cosine"
        )
        st.session_state.current_db_path = "face_database"
    except Exception as e:
        st.session_state.labeler = None
if 'face_identifications' not in st.session_state:
    st.session_state.face_identifications = {}
if 'current_db_path' not in st.session_state:
    st.session_state.current_db_path = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'batch_progress' not in st.session_state:
    st.session_state.batch_progress = 0
if 'caption_generator' not in st.session_state:
    st.session_state.caption_generator = None
if 'image_captions' not in st.session_state:
    st.session_state.image_captions = None
if 'dspy_generator' not in st.session_state:
    st.session_state.dspy_generator = None
if 'use_dspy' not in st.session_state:
    st.session_state.use_dspy = False
if 'qdrant_store' not in st.session_state:
    st.session_state.qdrant_store = None
if 'current_photo_path' not in st.session_state:
    st.session_state.current_photo_path = None
if 'current_photo_id' not in st.session_state:
    st.session_state.current_photo_id = None
if 'journey_mapper' not in st.session_state:
    st.session_state.journey_mapper = None
if 'location_contextualizer' not in st.session_state:
    st.session_state.location_contextualizer = None
if 'semantic_search' not in st.session_state:
    st.session_state.semantic_search = None

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

def initialize_caption_generator(model_name="llava:7b"):
    """Initialize the caption generator with Ollama vision model."""
    try:
        with st.spinner(f"Connecting to Ollama and loading {model_name} model..."):
            generator = CaptionGenerator(model_name=model_name)
            st.session_state.caption_generator = generator
            st.success(f"✅ Caption generator ready! Connected to Ollama with {model_name} model.")
            return True
    except ConnectionError as e:
        st.error(f"❌ Cannot connect to Ollama: {str(e)}")
        st.markdown(f"""
        #### To use caption generation, you need to:
        1. **Install Ollama** from https://ollama.ai
        2. **Pull a vision model**:
           ```bash
           ollama pull {model_name}
           # Or alternative models:
           ollama pull llava:7b
           ollama pull qwen2.5vl:7b
           ```
        3. **Start Ollama** (in a new terminal):
           ```bash
           ollama serve
           ```
        4. Then come back here and click the button again.
        """)
        return False
    except Exception as e:
        st.error(f"❌ Error initializing caption generator: {str(e)}")
        st.error("**Details:**\n" + traceback.format_exc())
        return False

def initialize_dspy_generator():
    """Initialize the DSPy-enhanced caption generator."""
    try:
        with st.spinner("Initializing DSPy + LLaVA integration..."):
            from travel_log.dspy_llava_integration import DSPyLLaVACaptionGenerator
            generator = DSPyLLaVACaptionGenerator()
            st.session_state.dspy_generator = generator
            st.success("✅ DSPy caption generator ready! Enhanced reasoning enabled.")
            return True
    except ImportError:
        st.error("❌ DSPy not installed!")
        st.markdown("""
        #### To use DSPy-enhanced captions:
        1. **Install DSPy**:
           ```bash
           uv add dspy-ai
           ```
        2. **Ensure Llama3 model** (for reasoning):
           ```bash
           ollama pull llama3
           ```
        3. Then come back and click the button again.
        """)
        return False
    except ConnectionError as e:
        st.error(f"❌ Cannot connect to Ollama: {str(e)}")
        st.info("Make sure Ollama is running with both llava:7b and llama3 models")
        return False
    except Exception as e:
        st.error(f"❌ Error initializing DSPy generator: {str(e)}")
        st.error("**Details:**\n" + traceback.format_exc())
        return False

def save_uploaded_photo(uploaded_file, metadata=None):
    """
    Save uploaded photo to permanent storage directory.

    Args:
        uploaded_file: Streamlit UploadedFile object
        metadata: Optional metadata dict with datetime info

    Returns:
        Path to saved file
    """
    # Create storage directory
    storage_dir = Path("uploaded_photos")
    storage_dir.mkdir(exist_ok=True)

    # Generate unique filename using hash + original name
    file_content = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_content).hexdigest()[:8]

    # Use datetime from metadata or current time
    if metadata and 'datetime' in metadata:
        timestamp = metadata['datetime'].strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    original_name = Path(uploaded_file.name).stem
    extension = Path(uploaded_file.name).suffix
    filename = f"{timestamp}_{original_name}_{file_hash}{extension}"

    # Save file
    save_path = storage_dir / filename
    save_path.write_bytes(file_content)

    return str(save_path)

def initialize_qdrant_store():
    """Initialize Qdrant vector database connection."""
    try:
        with st.spinner("Connecting to Qdrant..."):
            from travel_log.qdrant_store import create_qdrant_store
            # Connect to Qdrant on sapphire server
            store = create_qdrant_store(url="http://sapphire:6333")
            stats = store.get_statistics()
            st.session_state.qdrant_store = store
            st.success(f"✅ Qdrant connected! {stats['total_photos']} photos in database.")
            return True
    except ImportError:
        st.error("❌ Qdrant client not installed!")
        st.markdown("""
        #### To use Qdrant storage:
        1. **Install Qdrant client**:
           ```bash
           uv add qdrant-client
           ```
        2. **Start Qdrant** (Docker):
           ```bash
           docker start <qdrant_container_name>
           ```
        3. Then come back and click the button again.
        """)
        return False
    except Exception as e:
        st.error(f"❌ Cannot connect to Qdrant: {str(e)}")
        st.markdown("""
        #### Troubleshooting:
        1. Check Qdrant is running:
           ```bash
           docker ps | grep qdrant
           ```
        2. Start Qdrant if needed:
           ```bash
           docker start <container_name>
           ```
        3. Verify connection:
           ```bash
           curl http://localhost:6333
           ```
        """)
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
    Identify detected faces using Qdrant reference_faces collection.

    Args:
        face_images: List of face image data
        confidence_threshold: Minimum confidence for matches (converted from distance threshold)

    Returns:
        Dictionary with identification results
    """
    if st.session_state.qdrant_store is None:
        st.error("❌ Qdrant not connected! Click 'Connect to Qdrant' in the sidebar first.")
        return {}

    from deepface import DeepFace
    import numpy as np

    results = {}

    # Log for debugging
    st.info(f"🔍 Identifying {len(face_images)} faces using Qdrant reference_faces collection...")

    # Distance threshold for VGG-Face cosine similarity
    # Lower distance = better match (0.0 = perfect match, 1.0 = no similarity)
    # Set to 0.75 to handle masked faces, different angles, and lighting conditions
    distance_threshold = 0.75  # Very lenient threshold for real-world photos (masks, angles, lighting)

    for idx, face_data in enumerate(face_images):
        try:
            # Convert PIL image to numpy array for DeepFace
            face_array = np.array(face_data['image'])

            # Generate embedding using VGG-Face
            embedding_result = DeepFace.represent(
                img_path=face_array,
                model_name="VGG-Face",
                enforce_detection=False
            )

            if not embedding_result:
                results[idx] = {
                    'match': 'Unknown',
                    'confidence': 0.0,
                    'status': 'no_embedding'
                }
                continue

            embedding = embedding_result[0]['embedding']

            # Search in Qdrant reference_faces collection
            if st.session_state.qdrant_store:
                query_response = st.session_state.qdrant_store.client.query_points(
                    collection_name="reference_faces",
                    query=embedding,
                    limit=3
                )

                if query_response.points:
                    best_match = query_response.points[0]
                    person_name = best_match.payload['person_name']
                    score = best_match.score  # Cosine similarity score (0-1, higher is better)
                    distance = 1 - score  # Convert to distance

                    # Convert distance to confidence percentage
                    confidence = max(0, 1 - distance)

                    # Debug info
                    st.info(f"Face {idx+1}: Best match = {person_name}, Distance = {distance:.4f}, Threshold = {distance_threshold}")

                    if distance <= distance_threshold:
                        results[idx] = {
                            'match': person_name,
                            'confidence': confidence,
                            'distance': distance,
                            'status': 'matched'
                        }
                    else:
                        results[idx] = {
                            'match': 'Unknown',
                            'confidence': confidence,
                            'distance': distance,
                            'status': 'low_confidence',
                            'debug_best_match': person_name,  # Show who was closest
                            'debug_distance': distance
                        }
                else:
                    results[idx] = {
                        'match': 'Unknown',
                        'confidence': 0.0,
                        'status': 'no_match'
                    }
            else:
                results[idx] = {
                    'match': 'Error',
                    'error': 'Qdrant not connected'
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

        # Save uploaded file to temporary location for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(image_file.getvalue())
            tmp_path = tmp_file.name

        # Extract EXIF metadata first (before conversion)
        with st.spinner("Reading photo metadata..."):
            metadata = get_complete_metadata(tmp_path)

        # Save to permanent storage
        permanent_path = save_uploaded_photo(image_file, metadata)
        
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
                # Get the face numpy array or crop from coordinates
                facial_area = face.get('facial_area', {})
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 0)
                h = facial_area.get('h', 0)

                # Crop from original image with padding for better face extraction
                # Add 30% padding around detected face for better identification
                img_width, img_height = img.size
                padding = 0.3

                pad_w = int(w * padding)
                pad_h = int(h * padding)

                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(img_width, x + w + pad_w)
                y2 = min(img_height, y + h + pad_h)

                face_img = img.crop((x1, y1, x2, y2))

                face_images.append({
                    'image': face_img,
                    'location': (x, y, w, h),
                    'confidence': face.get('confidence', 1.0)
                })
            
            # Clean up temp file - we have permanent storage
            try:
                Path(tmp_path).unlink()
            except:
                pass

            return {
                'original_image': img,
                'faces': face_images,
                'num_faces': len(filtered_faces),
                'metadata': metadata,
                'temp_path': permanent_path  # Use permanent path for Qdrant storage
            }
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.error(f"Details: {traceback.format_exc()}")
        return None

def main():
    # Header
    st.title("📸 Travel Log")
    st.markdown("**Upload photos, detect faces, and identify them**")

    # Create tabs for different features
    detection_tab, identification_tab, caption_tab, qdrant_tab, database_tab, journey_tab, search_tab = st.tabs([
        "🔍 Face Detection",
        "🎯 Face Identification",
        "📔 Travel Log",
        "🗄️ Qdrant Storage",
        "💾 Face Database",
        "🗺️ Journey Map",
        "🔎 Search"
    ])

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Detection backend selection
        backend_options = ['retinaface', 'mtcnn']

        backend = st.selectbox(
            "Detection Backend",
            backend_options,
            index=0,  # Default to retinaface
            help="Choose the face detection algorithm"
        )

        st.markdown("""
        **Backend Guide:**
        - **retinaface**: Most accurate (recommended)
        - **mtcnn**: Fast and reliable
        """)
        
        # Confidence threshold
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.3,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Lower = detect more faces (even with masks/angles). Higher = only clear faces"
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
                            st.session_state.current_photo_path = result.get('temp_path')  # Save for Qdrant
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
            st.subheader("Face Recognition")

            st.info("🤖 Using DeepFace with VGG-Face model")
            st.markdown("""
            **Recognition Method:**
            - Model: VGG-Face (4096D embeddings)
            - Search: Qdrant vector similarity
            - Distance: Cosine similarity
            - Threshold: 0.75
            """)

            # Show reference faces count
            if st.session_state.qdrant_store:
                try:
                    collection_info = st.session_state.qdrant_store.client.get_collection("reference_faces")
                    num_faces = collection_info.points_count
                    st.success(f"✅ {num_faces} reference faces in database")
                except:
                    st.warning("⚠️ Reference faces collection not found")
            else:
                st.warning("⚠️ Connect to Qdrant first")

        with col2:
            st.subheader("Quick Actions")

            # Identify faces button
            if st.button("🔍 Identify Faces", type="primary", use_container_width=True):
                if not st.session_state.detected_faces:
                    st.warning("⚠️ No faces detected. Upload a photo in Face Detection tab first.")
                elif not st.session_state.qdrant_store:
                    st.warning("⚠️ Connect to Qdrant first (in sidebar)")
                else:
                    with st.spinner("Identifying faces using DeepFace + Qdrant..."):
                        identifications = identify_faces(st.session_state.detected_faces)
                        st.session_state.face_identifications = identifications
                        st.rerun()

            st.divider()

            # Show identified faces summary
            if st.session_state.face_identifications:
                matched = sum(1 for i in st.session_state.face_identifications.values()
                            if i.get('match') not in ['Unknown', 'Error'])
                total = len(st.session_state.face_identifications)
                st.metric("Identified", f"{matched}/{total}")

                # List unique people found
                people = set(i.get('match') for i in st.session_state.face_identifications.values()
                           if i.get('match') not in ['Unknown', 'Error'])
                if people:
                    st.success(f"Found: {', '.join(sorted(people))}")

        st.divider()

        # Display identified faces
        if st.session_state.detected_faces:
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
                                        # Show debug info if available
                                        if 'debug_best_match' in identification:
                                            st.caption(f"Closest: {identification['debug_best_match']} (distance: {identification['debug_distance']:.4f})")
                                    elif match == 'Error':
                                        st.error(f"Error: {identification.get('error', 'Unknown')}")
                                    else:
                                        st.success(f"👤 {match}")
                                        st.progress(
                                            confidence,
                                            text=f"Confidence: {confidence:.1%}"
                                        )
        else:
            st.info("👈 Upload and detect faces in the Face Detection tab, then click 'Identify Faces'")

    # ============================================================================
    # TAB 3: TRAVEL LOG
    # ============================================================================
    with caption_tab:
        st.header("📔 Travel Log")
        st.markdown("View all your travel photos with captions, locations, and identified people")

        # Check if Qdrant is connected
        if st.session_state.qdrant_store is None:
            st.warning("⚠️ Please connect to Qdrant first")
            st.info("👈 Click 'Connect to Qdrant' in the sidebar to view your travel log")
        else:
            # Get all photos from Qdrant
            try:
                # Fetch all photos
                all_points = st.session_state.qdrant_store.client.scroll(
                    collection_name="travel_photos",
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )

                photos = all_points[0] if all_points else []

                if not photos:
                    st.info("📷 No photos in your travel log yet!")
                    st.markdown("""
                    **To add photos to your travel log:**
                    1. Upload a photo in the **Face Detection** tab
                    2. Detect faces (optional)
                    3. Identify people (optional) in **Face Identification** tab
                    4. Save to Qdrant in **Qdrant Storage** tab
                    """)
                else:
                    st.success(f"📸 Found {len(photos)} photos in your travel log")

                    # Sort by datetime if available
                    photos_sorted = sorted(
                        photos,
                        key=lambda p: p.payload.get('datetime', ''),
                        reverse=True
                    )

                    # Display each photo as a travel log entry
                    for idx, point in enumerate(photos_sorted):
                        payload = point.payload

                        with st.container():
                            st.divider()

                            # Create columns: image, details, delete button
                            col_img, col_details, col_delete = st.columns([1, 3, 0.3])

                            with col_img:
                                # Display photo if filepath exists and is readable
                                photo_path = payload.get('filepath')
                                image_displayed = False

                                if photo_path:
                                    try:
                                        photo_path_obj = Path(photo_path)
                                        if photo_path_obj.exists() and photo_path_obj.is_file():
                                            img = Image.open(photo_path)
                                            st.image(img, use_container_width=True)
                                            image_displayed = True
                                    except (PermissionError, FileNotFoundError, Exception):
                                        pass

                                if not image_displayed:
                                    st.info("📷 No preview")

                            with col_details:
                                # Title
                                captions = payload.get('captions', {})
                                if captions and 'title' in captions:
                                    st.markdown(f"**{captions['title']}**")
                                else:
                                    st.markdown(f"**{payload.get('filename', 'Untitled')}**")

                                # Caption
                                if captions and 'caption' in captions:
                                    st.caption(captions['caption'])

                                # Compact metadata
                                meta_parts = []

                                if 'datetime' in payload:
                                    meta_parts.append(f"📅 {payload['datetime']}")

                                people = payload.get('people_names', [])
                                if people:
                                    meta_parts.append(f"👥 {', '.join(people)}")

                                if 'latitude' in payload and 'longitude' in payload:
                                    lat, lon = payload['latitude'], payload['longitude']
                                    maps = format_gps_for_maps(lat, lon)
                                    meta_parts.append(f"[📍 Map]({maps['google_maps']})")

                                if meta_parts:
                                    st.markdown(" • ".join(meta_parts))

                            with col_delete:
                                # Delete button
                                if st.button("🗑️", key=f"delete_{point.id}", help="Delete this photo"):
                                    try:
                                        # Delete from Qdrant database
                                        st.session_state.qdrant_store.client.delete(
                                            collection_name="travel_photos",
                                            points_selector=[point.id]
                                        )

                                        # Try to delete file if it exists (ignore errors for temp files)
                                        if photo_path:
                                            try:
                                                photo_path_obj = Path(photo_path)
                                                if photo_path_obj.exists():
                                                    photo_path_obj.unlink()
                                            except (PermissionError, OSError, Exception):
                                                # Ignore file deletion errors (e.g., /tmp files, permission issues)
                                                pass

                                        st.success("✅ Photo deleted from database!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error deleting from database: {str(e)}")

            except Exception as e:
                st.error(f"❌ Error loading travel log: {str(e)}")
                st.error(f"Details: {traceback.format_exc()}")

        # Add caption generation for current photo
        st.divider()
        st.subheader("✨ Generate Caption for Current Photo")

        # Check if photo is uploaded
        if st.session_state.processed_image is None:
            st.info("👈 Please upload a photo in the **Face Detection** tab first")
        else:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📷 Current Photo")
                st.image(st.session_state.processed_image, caption="Uploaded Photo", use_container_width=True)

                # Caption mode selection
                caption_mode = st.radio(
                    "Caption Generation Mode:",
                    ["🔍 Basic (LLaVA only)", "🧠 Enhanced (DSPy + LLaVA)"],
                    help="Basic: Simple visual description | Enhanced: Context-aware with face names, location, mood"
                )

                use_dspy = "Enhanced" in caption_mode
                st.session_state.use_dspy = use_dspy

                # Model selection
                vision_model = st.selectbox(
                    "Vision Model",
                    options=["llava:7b", "qwen2.5vl:7b"],
                    index=0,
                    help="Select the Ollama vision model to use for caption generation"
                )

                # Initialize appropriate generator
                if use_dspy:
                    if st.button("🚀 Load Enhanced Caption Generator", type="primary"):
                        if initialize_dspy_generator():
                            st.success("✅ DSPy + LLaVA generator loaded!")

                    if st.session_state.dspy_generator:
                        st.success("✅ Enhanced caption generator is ready")
                        st.info("💡 Will use face recognition, GPS, and timestamp data")
                    else:
                        st.info("💡 Click 'Load Enhanced Caption Generator' for smart captions")
                else:
                    if st.button("🚀 Load Caption Generator", type="primary"):
                        if initialize_caption_generator(model_name=vision_model):
                            st.success(f"✅ Caption generator loaded with {vision_model}!")

                    if st.session_state.caption_generator:
                        current_model = st.session_state.caption_generator.model_name
                        st.success(f"✅ Caption generator is ready (using {current_model})")
                    else:
                        st.info("💡 Click 'Load Caption Generator' to enable caption generation")

                st.divider()

                # Caption generation options
                caption_type = st.radio(
                    "Type of caption to generate:",
                    ["All (caption, title, travel)", "Title only", "Detailed caption", "Travel caption only"],
                    help="Choose what type of captions you want"
                )

                if st.button("✨ Generate Captions", type="primary"):
                    # Check if appropriate generator is loaded
                    if use_dspy and st.session_state.dspy_generator is None:
                        st.warning("⚠️ Please load the Enhanced Caption Generator first!")
                    elif not use_dspy and st.session_state.caption_generator is None:
                        st.warning("⚠️ Please load the Caption Generator first!")
                    else:
                        try:
                            # Use the already processed image from Face Detection tab
                            image = st.session_state.processed_image

                            if use_dspy:
                                # Enhanced DSPy + LLaVA generation
                                with st.spinner("⏳ Analyzing with DSPy + LLaVA...\n\nCombining vision, face recognition, and context..."):
                                    # Get face names if available
                                    face_names = None
                                    if st.session_state.face_identifications:
                                        face_names = [
                                            ident.get('match', '')
                                            for ident in st.session_state.face_identifications.values()
                                            if ident.get('match') and ident.get('match') not in ['Unknown', 'Error']
                                        ]

                                    # Get metadata if available
                                    location = None
                                    timestamp = None
                                    if st.session_state.image_metadata:
                                        meta = st.session_state.image_metadata
                                        if 'latitude' in meta and 'longitude' in meta:
                                            location = f"{meta['latitude']:.4f}, {meta['longitude']:.4f}"
                                        timestamp = meta.get('datetime')

                                    # Generate enhanced captions
                                    result = st.session_state.dspy_generator.forward(
                                        image=image,
                                        face_names=face_names,
                                        location=location,
                                        timestamp=timestamp
                                    )

                                    # Format for display
                                    st.session_state.image_captions = {
                                        'title': result['title'],
                                        'caption': result['caption'],
                                        'scene_type': result['scene_type'],
                                        'mood': result['mood'],
                                        'hashtags': result['hashtags'],
                                        'raw_visual_analysis': result.get('raw_visual_analysis', '')
                                    }
                            else:
                                # Basic LLaVA generation
                                with st.spinner("⏳ Analyzing image with LLaVA...\n\nThis may take 30-60 seconds"):
                                    captions = st.session_state.caption_generator.generate_all(image)
                                    st.session_state.image_captions = captions

                            st.success("✅ Captions generated!")
                            st.rerun()

                        except TimeoutError:
                            st.error("⏱️ Caption generation timed out. The model is taking too long to load.")
                            st.info("💡 Try reducing the image size or restarting the app.")
                        except Exception as e:
                            st.error(f"Error generating captions: {str(e)}")
                            if "cuda" in str(e).lower() or "gpu" in str(e).lower():
                                st.error("GPU Error - Check GPU memory availability")
                            st.error(f"Details: {traceback.format_exc()}")

        with col2:
            st.subheader("📝 Generated Captions")

            if st.session_state.image_captions:
                captions = st.session_state.image_captions

                # Show DSPy-specific fields if available
                if st.session_state.use_dspy and 'scene_type' in captions:
                    col_meta1, col_meta2 = st.columns(2)
                    with col_meta1:
                        st.metric("Scene Type", captions.get('scene_type', 'N/A'))
                    with col_meta2:
                        st.metric("Mood", captions.get('mood', 'N/A'))
                    st.divider()

                if 'title' in captions:
                    st.markdown("### 🎯 Title")
                    st.markdown(f"> {captions['title']}")

                if 'caption' in captions:
                    st.markdown("### 📖 Detailed Caption")
                    st.markdown(f"> {captions['caption']}")

                if 'travel_caption' in captions:
                    st.markdown("### ✈️ Travel Caption")
                    st.markdown(f"> {captions['travel_caption']}")

                # Show DSPy hashtags
                if st.session_state.use_dspy and 'hashtags' in captions:
                    st.markdown("### #️⃣ Hashtags")
                    st.markdown(f"> {captions['hashtags']}")

                # Show raw visual analysis in expandable section
                if st.session_state.use_dspy and 'raw_visual_analysis' in captions:
                    with st.expander("🔍 Raw Visual Analysis (LLaVA)"):
                        st.write(captions['raw_visual_analysis'])

                # Copy to clipboard
                st.divider()
                st.subheader("📋 Copy Captions")

                all_captions = "\n\n".join(
                    f"**{key.replace('_', ' ').title()}:**\n{value}"
                    for key, value in captions.items()
                    if value
                )

                st.text_area(
                    "Copy all captions:",
                    value=all_captions,
                    height=200,
                    disabled=True
                )

                # Download as JSON
                import json
                json_captions = json.dumps(captions, indent=2)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_captions,
                    file_name="image_captions.json",
                    mime="application/json"
                )

                # Save captions to Qdrant
                st.divider()
                if st.button("💾 Save Captions to Qdrant", type="primary"):
                    if st.session_state.qdrant_store is None:
                        st.warning("⚠️ Please connect to Qdrant first (in the sidebar)")
                    elif st.session_state.current_photo_id is None:
                        st.warning("⚠️ Please save the photo to Qdrant first (in Qdrant Storage tab)")
                    else:
                        try:
                            # Update the photo record in Qdrant with captions
                            from qdrant_client.models import SetPayload

                            st.session_state.qdrant_store.client.set_payload(
                                collection_name="travel_photos",
                                payload={"captions": captions},
                                points=[st.session_state.current_photo_id]
                            )

                            st.success(f"✅ Captions saved to Qdrant for photo ID: {st.session_state.current_photo_id}")
                        except Exception as e:
                            st.error(f"❌ Error saving captions: {str(e)}")

            else:
                st.info("👈 Upload an image and click 'Generate Captions' to see results here.")

        st.divider()
        with st.expander("ℹ️ About Image Captions"):
            st.markdown("""
            ### How it works

            This feature uses **LLaVA** (Large Language and Vision Assistant), a powerful open-source vision-language model, to automatically generate captions.

            #### 🔍 Basic Mode (LLaVA only)
            - Pure visual description of the image
            - Fast and simple
            - Generates titles, captions, and travel-specific descriptions

            #### 🧠 Enhanced Mode (DSPy + LLaVA)
            - **Vision (LLaVA)**: Analyzes what's in the image
            - **Reasoning (DSPy)**: Adds context and intelligence
            - **Integration**: Combines face recognition, GPS, timestamps
            - **Output**: Scene type, mood, personalized captions with names, hashtags

            ### Capabilities

            **Basic Mode analyzes:**
            - Landscapes and scenery
            - People and activities
            - Buildings and landmarks
            - Weather and lighting
            - Visual elements

            **Enhanced Mode adds:**
            - ✅ Mentions people by name (from face recognition)
            - ✅ Location context (from GPS/EXIF)
            - ✅ Time awareness (morning/afternoon/evening)
            - ✅ Scene categorization (landscape/portrait/group)
            - ✅ Emotional tone (joyful/peaceful/adventurous)
            - ✅ Social media hashtags
            - ✅ Multi-format outputs

            ### Example Comparison

            **Basic Mode:**
            > "A beach scene with people at sunset"

            **Enhanced Mode (with face names + GPS):**
            > Title: "Golden Hour Memories"
            > Caption: "Sarah and John enjoying sunset at Malibu Beach on a perfect California evening"
            > Mood: Peaceful and romantic
            > Hashtags: #MalibuSunset #CaliforniaLove #BeachLife

            ### Tips

            - Use high-quality images for better results
            - For Enhanced mode: Detect and identify faces first in other tabs
            - Photos with GPS data get location-aware captions
            - Generated captions can be edited for perfect wording

            ### Requirements

            **Basic Mode:**
            - Ollama running with `llava:7b`

            **Enhanced Mode:**
            - Ollama with `llava:7b` and `llama3`
            - DSPy installed (`uv add dspy-ai`)
            - Optionally: Face recognition for name integration

            ### Processing Time

            - Basic: ~30 seconds per image
            - Enhanced: ~40 seconds per image (+10s for reasoning)
            - Runs locally (no cloud upload)
            - GPU acceleration recommended
            """)

    # ============================================================================
    # TAB 4: QDRANT STORAGE
    # ============================================================================
    with qdrant_tab:
        st.header("🗄️ Qdrant Vector Database")
        st.markdown("Store and search photos with embeddings, EXIF, faces, and captions")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("⚙️ Database Connection")

            # Initialize Qdrant
            if st.button("🔌 Connect to Qdrant", type="primary"):
                if initialize_qdrant_store():
                    st.rerun()

            if st.session_state.qdrant_store:
                stats = st.session_state.qdrant_store.get_statistics()
                st.success("✅ Qdrant connected")

                # Display stats
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Total Photos", stats.get('total_photos', 0))
                with col_stat2:
                    st.metric("Total Faces", stats.get('total_faces', 0))
                with col_stat3:
                    st.metric("Embedding Dim", stats.get('embedding_dimension', 512))

                st.info(f"Collections: {stats.get('collection_name')}, {stats.get('faces_collection_name', 'detected_faces')}")
            else:
                st.warning("⚠️ Not connected to Qdrant")

            st.divider()

            # Save current photo
            st.subheader("💾 Save Current Photo")

            if st.session_state.processed_image is not None:
                st.info(f"📸 Current photo ready to save")

                # Check if location exists
                has_location = (st.session_state.image_metadata and
                              'latitude' in st.session_state.image_metadata and
                              'longitude' in st.session_state.image_metadata)

                # Manual location input if no GPS data
                if not has_location:
                    st.warning("⚠️ No GPS location in photo metadata")

                    with st.expander("📍 Add Location Manually", expanded=True):
                        st.markdown("Enter GPS coordinates for this photo:")

                        col_lat, col_lon = st.columns(2)
                        with col_lat:
                            manual_lat = st.number_input(
                                "Latitude",
                                min_value=-90.0,
                                max_value=90.0,
                                value=0.0,
                                step=0.0001,
                                format="%.6f",
                                help="Example: 37.7749 (San Francisco)"
                            )
                        with col_lon:
                            manual_lon = st.number_input(
                                "Longitude",
                                min_value=-180.0,
                                max_value=180.0,
                                value=0.0,
                                step=0.0001,
                                format="%.6f",
                                help="Example: -122.4194 (San Francisco)"
                            )

                        if st.button("✅ Add Location to Photo"):
                            if manual_lat != 0.0 or manual_lon != 0.0:
                                if st.session_state.image_metadata is None:
                                    st.session_state.image_metadata = {}
                                st.session_state.image_metadata['latitude'] = manual_lat
                                st.session_state.image_metadata['longitude'] = manual_lon
                                st.session_state.image_metadata['manual_location'] = True
                                st.success(f"✅ Location added: {manual_lat:.6f}, {manual_lon:.6f}")
                                st.rerun()
                            else:
                                st.warning("Please enter non-zero coordinates")

                        st.caption("💡 Tip: You can find coordinates on Google Maps by right-clicking a location")
                else:
                    lat = st.session_state.image_metadata['latitude']
                    lon = st.session_state.image_metadata['longitude']
                    st.success(f"✅ Location: {lat:.6f}°, {lon:.6f}°")

                # Show what will be saved
                with st.expander("📋 Data to be saved"):
                    data_summary = []
                    if st.session_state.image_metadata:
                        data_summary.append("✅ EXIF metadata")
                        if 'latitude' in st.session_state.image_metadata:
                            if st.session_state.image_metadata.get('manual_location'):
                                data_summary.append("✅ GPS location (manually added)")
                            else:
                                data_summary.append("✅ GPS location (from EXIF)")
                    if st.session_state.detected_faces:
                        data_summary.append(f"✅ {len(st.session_state.detected_faces)} detected faces")
                    if st.session_state.face_identifications:
                        data_summary.append(f"✅ {len(st.session_state.face_identifications)} face IDs")
                    if st.session_state.image_captions:
                        data_summary.append("✅ Generated captions")

                    for item in data_summary:
                        st.write(item)

                if st.button("💾 Save to Qdrant", type="primary"):
                    if not st.session_state.qdrant_store:
                        st.error("❌ Connect to Qdrant first!")
                    elif not st.session_state.current_photo_path:
                        st.error("❌ No photo path available")
                    else:
                        try:
                            with st.spinner("Saving photo to Qdrant..."):
                                # Save photo
                                point_id = st.session_state.qdrant_store.store_photo(
                                    photo_path=st.session_state.current_photo_path,
                                    face_embedding=None,  # Could add face embeddings here
                                    exif_metadata=st.session_state.image_metadata,
                                    detected_faces=st.session_state.detected_faces,
                                    face_identifications=st.session_state.face_identifications,
                                    captions=st.session_state.image_captions
                                )
                                # Store the photo ID so captions can be saved later
                                st.session_state.current_photo_id = point_id
                                st.success(f"✅ Photo saved! ID: {point_id[:8]}...")

                                # Save individual faces if available
                                if st.session_state.detected_faces:
                                    with st.spinner("Saving individual faces..."):
                                        face_ids = st.session_state.qdrant_store.store_individual_faces(
                                            photo_id=point_id,
                                            photo_path=st.session_state.current_photo_path,
                                            detected_faces=st.session_state.detected_faces,
                                            face_identifications=st.session_state.face_identifications,
                                            exif_metadata=st.session_state.image_metadata
                                        )
                                        st.success(f"✅ Saved {len(face_ids)} individual faces!")

                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error saving: {str(e)}")
                            with st.expander("🔍 Error details"):
                                st.code(str(e))
            else:
                st.info("👈 Detect faces in a photo first, then save here")

        with col2:
            st.subheader("🔍 Search & Browse")

            if st.session_state.qdrant_store:
                search_type = st.radio(
                    "Search Type:",
                    ["📸 All Photos", "👤 By Person", "📍 By Location", "👥 All Faces", "🔎 Faces by Person"]
                )

                if search_type == "📸 All Photos":
                    if st.button("🔍 Get All Photos"):
                        photos = st.session_state.qdrant_store.get_all_photos(limit=50)
                        if photos:
                            st.success(f"Found {len(photos)} photos")
                            for photo in photos[:10]:
                                with st.expander(f"📷 {photo['filename']}"):
                                    st.write(f"**People**: {', '.join(photo.get('people', [])) or 'None'}")
                                    st.write(f"**Faces**: {photo.get('num_faces', 0)}")
                                    if photo.get('datetime'):
                                        st.write(f"**Date**: {photo['datetime']}")
                        else:
                            st.info("No photos in database yet")

                elif search_type == "👤 By Person":
                    person_name = st.text_input("Person name:")
                    if st.button("🔍 Search") and person_name:
                        photos = st.session_state.qdrant_store.search_by_person(person_name, limit=20)
                        if photos:
                            st.success(f"Found {len(photos)} photos with {person_name}")
                            for photo in photos:
                                with st.expander(f"📷 {photo['filename']}"):
                                    st.write(f"**People**: {', '.join(photo.get('people', []))}")
                                    if photo.get('datetime'):
                                        st.write(f"**Date**: {photo['datetime']}")
                        else:
                            st.info(f"No photos found with {person_name}")

                elif search_type == "📍 By Location":
                    col_lat, col_lon = st.columns(2)
                    with col_lat:
                        lat = st.number_input("Latitude:", value=37.7749)
                    with col_lon:
                        lon = st.number_input("Longitude:", value=-122.4194)
                    radius = st.slider("Radius (km):", 1, 100, 10)

                    if st.button("🔍 Search"):
                        photos = st.session_state.qdrant_store.search_by_location(lat, lon, radius)
                        if photos:
                            st.success(f"Found {len(photos)} photos within {radius}km")
                            for photo in photos:
                                with st.expander(f"📷 {photo['filename']} ({photo['distance_km']}km away)"):
                                    st.write(f"**Location**: {photo['location']['lat']:.4f}, {photo['location']['lon']:.4f}")
                                    st.write(f"**People**: {', '.join(photo.get('metadata', {}).get('people_names', []))}")
                        else:
                            st.info(f"No photos found within {radius}km")

                elif search_type == "👥 All Faces":
                    if st.button("🔍 Get All Faces"):
                        faces = st.session_state.qdrant_store.get_all_faces(limit=50)
                        if faces:
                            st.success(f"Found {len(faces)} faces")

                            # Group by person
                            from collections import defaultdict
                            faces_by_person = defaultdict(list)
                            for face in faces:
                                person = face.get('person_name', 'Unknown')
                                faces_by_person[person].append(face)

                            # Display grouped by person
                            for person, person_faces in sorted(faces_by_person.items()):
                                with st.expander(f"👤 {person} ({len(person_faces)} faces)"):
                                    for face in person_faces[:5]:  # Show first 5
                                        st.write(f"📷 {face['filename']} - Face #{face['face_index']}")
                                        if face.get('datetime'):
                                            st.write(f"📅 {face['datetime']}")
                                        if face.get('confidence'):
                                            st.write(f"✓ Confidence: {face['confidence']:.2%}")
                                        st.divider()
                        else:
                            st.info("No faces in database yet")

                elif search_type == "🔎 Faces by Person":
                    person_name = st.text_input("Person name:", key="face_search_person")
                    if st.button("🔍 Search Faces") and person_name:
                        faces = st.session_state.qdrant_store.search_faces_by_person(person_name, limit=50)
                        if faces:
                            st.success(f"Found {len(faces)} faces of {person_name}")

                            for face in faces[:10]:  # Show first 10
                                with st.expander(f"📷 {face['filename']} - Face #{face['face_index']}"):
                                    st.write(f"**Person**: {face['person_name']}")
                                    if face.get('confidence'):
                                        st.write(f"**Confidence**: {face['confidence']:.2%}")
                                    if face.get('datetime'):
                                        st.write(f"**Date**: {face['datetime']}")
                                    if face.get('bbox'):
                                        bbox = face['bbox']
                                        st.write(f"**Location in photo**: ({bbox['x']}, {bbox['y']}) - {bbox['w']}×{bbox['h']}px")
                        else:
                            st.info(f"No faces found for {person_name}")
            else:
                st.warning("⚠️ Connect to Qdrant to search photos")

        st.divider()

        with st.expander("ℹ️ About Qdrant Storage"):
            st.markdown("""
            ### What is Qdrant?

            Qdrant is a vector database that stores your photos with:
            - **Face embeddings** (512D vectors for similarity search)
            - **EXIF metadata** (GPS, timestamp, camera info)
            - **Detected faces** (bounding boxes, confidence)
            - **Face identifications** (names, confidence scores)
            - **Generated captions** (LLaVA, DSPy)

            ### Features

            - ✅ **Semantic search** by face similarity
            - ✅ **Search by person** name
            - ✅ **Geospatial search** by location
            - ✅ **Browse all photos** in database
            - ✅ **Persistent storage** (survives app restart)

            ### Workflow

            1. **Detect faces** in Face Detection tab
            2. **Identify people** in Face Identification tab
            3. **Generate captions** in Image Captions tab
            4. **Save to Qdrant** in this tab
            5. **Search and browse** your photo collection

            ### Requirements

            - Qdrant running (Docker):
              ```bash
              docker start <qdrant_container>
              ```
            - Verify at: http://localhost:6333
            """)

    # ============================================================================
    # TAB 5: FACE DATABASE MANAGEMENT
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

            # Option to replace existing files
            replace_existing = st.checkbox(
                "Replace existing files",
                value=False,
                help="If checked, duplicate filenames will be replaced. Otherwise, files will be auto-renamed (e.g., photo_1.jpg, photo_2.jpg)."
            )

            if st.button("Add to Database"):
                if person_name and uploaded_images:
                    person_dir = db_path / person_name
                    person_dir.mkdir(parents=True, exist_ok=True)

                    count = 0
                    skipped = 0
                    replaced = 0
                    skipped_files = []

                    for img_file in uploaded_images:
                        img_path = person_dir / img_file.name

                        # Check if file already exists
                        is_duplicate = img_path.exists()

                        if is_duplicate:
                            if replace_existing:
                                # User wants to replace existing file
                                replaced += 1
                            else:
                                # Auto-rename by adding a number suffix
                                base_name = img_path.stem
                                extension = img_path.suffix
                                counter = 1
                                while img_path.exists():
                                    img_path = person_dir / f"{base_name}_{counter}{extension}"
                                    counter += 1
                                # Now img_path is unique, so this is a new addition
                                is_duplicate = False

                        try:
                            with open(img_path, 'wb') as f:
                                f.write(img_file.getvalue())

                            if not is_duplicate:
                                count += 1
                        except Exception as e:
                            st.error(f"❌ Error saving {img_file.name}: {str(e)}")

                    # Show results
                    if count > 0:
                        st.success(f"✅ Added {count} new image(s) for {person_name}")
                    if replaced > 0:
                        st.success(f"✅ Replaced {replaced} existing image(s) for {person_name}")
                    if skipped > 0:
                        st.warning(f"⚠️ Skipped {skipped} duplicate file(s): {', '.join(skipped_files[:3])}{'...' if len(skipped_files) > 3 else ''}")

                    if count > 0 or replaced > 0 or skipped > 0:
                        st.info("💡 Click '🔄 Update Reference Faces' below to sync with Qdrant")
                        st.rerun()
                else:
                    st.warning("⚠️ Enter name and select images")

            st.divider()

            # Update reference database button
            st.markdown("**🔄 Update Qdrant Database**")
            st.caption("Sync face_database to Qdrant vector store")

            if st.button("🔄 Update Reference Faces", type="primary", use_container_width=True):
                if not st.session_state.qdrant_store:
                    st.warning("⚠️ Connect to Qdrant first (in sidebar)")
                else:
                    with st.spinner("Updating reference faces in Qdrant..."):
                        import subprocess
                        result = subprocess.run(
                            ["uv", "run", "python", "store_reference_faces.py"],
                            capture_output=True,
                            text=True
                        )
                        if result.returncode == 0:
                            # Extract the count from output
                            output = result.stdout
                            if "Successfully stored" in output:
                                import re
                                match = re.search(r'Successfully stored (\d+) reference faces', output)
                                if match:
                                    count = match.group(1)
                                    st.success(f"✅ Successfully updated {count} reference faces in Qdrant!")
                                else:
                                    st.success("✅ Reference faces updated successfully!")
                            else:
                                st.success("✅ Reference faces updated successfully!")

                            # Show summary in expander
                            with st.expander("📋 View Update Details"):
                                st.code(output, language="text")
                        else:
                            st.error(f"❌ Error updating reference faces:\n{result.stderr}")

            st.caption("💡 Run this after adding/removing people from face_database/")

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

    # ============================================================================
    # TAB 6: JOURNEY MAPPING
    # ============================================================================
    with journey_tab:
        st.header("🗺️ Journey Map")
        st.markdown("Visualize your travel path on a map with chronological locations")

        # Initialize journey mapper if not already done
        if st.session_state.journey_mapper is None and st.session_state.qdrant_store:
            from travel_log import create_journey_mapper
            st.session_state.journey_mapper = create_journey_mapper(st.session_state.qdrant_store)

        if not st.session_state.qdrant_store:
            st.warning("⚠️ Please connect to Qdrant first (in sidebar)")
        elif not st.session_state.journey_mapper:
            st.error("❌ Journey mapper not initialized")
        else:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("⚙️ Journey Settings")

                # Date range filter
                use_date_filter = st.checkbox("Filter by date range", value=False)

                start_date = None
                end_date = None

                if use_date_filter:
                    col_start, col_end = st.columns(2)
                    with col_start:
                        start_date = st.date_input("Start date")
                    with col_end:
                        end_date = st.date_input("End date")

                    if start_date:
                        start_date = datetime.combine(start_date, datetime.min.time())
                    if end_date:
                        end_date = datetime.combine(end_date, datetime.max.time())

                # Generate journey button
                if st.button("🗺️ Generate Journey Map", type="primary"):
                    try:
                        with st.spinner("Generating journey map..."):
                            journey_summary = st.session_state.journey_mapper.generate_journey_summary(
                                start_date=start_date,
                                end_date=end_date
                            )

                            if journey_summary['total_points'] == 0:
                                st.info("📷 No photos with GPS data found")
                            else:
                                st.session_state.journey_summary = journey_summary
                                st.success(f"✅ Found {journey_summary['total_points']} locations across {journey_summary['total_days']} days")
                                st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error generating journey: {str(e)}")

            with col2:
                st.subheader("📊 Journey Statistics")

                if hasattr(st.session_state, 'journey_summary'):
                    summary = st.session_state.journey_summary

                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Total Locations", summary['total_points'])
                    with col_stat2:
                        st.metric("Days", summary['total_days'])
                    with col_stat3:
                        if summary['total_points'] > 0:
                            st.metric("Start Date", summary['start_date'])

        # Display journey map
        st.divider()

        if hasattr(st.session_state, 'journey_summary'):
            summary = st.session_state.journey_summary

            if summary['total_points'] > 0:
                st.subheader("🌍 Your Journey")

                # Overall map link
                st.markdown(f"**[🗺️ Open Full Journey in Google Maps]({summary['map_url']})**")

                st.divider()

                # Daily breakdown
                st.subheader("📅 Daily Journeys")

                for date, day_info in sorted(summary['daily_maps'].items(), reverse=True):
                    with st.expander(f"📅 {date} ({day_info['num_photos']} photos)"):
                        st.markdown(f"**[🗺️ Open Day Map]({day_info['map_url']})**")

                        if day_info['people']:
                            st.write(f"**People**: {', '.join(day_info['people'])}")

                        # Show photo locations
                        for point in day_info['points']:
                            time_str = point['datetime'].strftime('%H:%M')
                            st.write(f"📍 {time_str} - {point.get('caption', point['filename'])}")

                # Google Maps Embed
                st.divider()
                st.subheader("🗺️ Journey Map")

                # Create Google Maps embed URL
                # Use the first and last points for directions
                if len(summary['points']) >= 2:
                    origin = summary['points'][0]
                    destination = summary['points'][-1]
                    waypoints = summary['points'][1:-1]

                    # Build embed URL for directions
                    origin_str = f"{origin['lat']},{origin['lon']}"
                    dest_str = f"{destination['lat']},{destination['lon']}"

                    if waypoints:
                        waypoint_str = "|".join(f"{p['lat']},{p['lon']}" for p in waypoints[:23])
                        embed_url = f"https://www.google.com/maps/embed/v1/directions?key=AIzaSyAOVYRIgupAurZup5y1PRh8Ismb1A3lLao&origin={origin_str}&destination={dest_str}&waypoints={waypoint_str}&mode=driving"
                    else:
                        embed_url = f"https://www.google.com/maps/embed/v1/directions?key=AIzaSyAOVYRIgupAurZup5y1PRh8Ismb1A3lLao&origin={origin_str}&destination={dest_str}&mode=driving"

                    # Display embedded map
                    st.markdown(f'<iframe width="100%" height="600" frameborder="0" style="border:0" src="{embed_url}" allowfullscreen></iframe>', unsafe_allow_html=True)
                elif len(summary['points']) == 1:
                    # Single point - show location
                    point = summary['points'][0]
                    embed_url = f"https://www.google.com/maps/embed/v1/place?key=AIzaSyAOVYRIgupAurZup5y1PRh8Ismb1A3lLao&q={point['lat']},{point['lon']}&zoom=14"
                    st.markdown(f'<iframe width="100%" height="600" frameborder="0" style="border:0" src="{embed_url}" allowfullscreen></iframe>', unsafe_allow_html=True)

        # GPS Coordinate Editor
        st.divider()
        st.subheader("📍 Edit GPS Coordinates")

        with st.expander("✏️ Update Photo Locations"):
            if not st.session_state.qdrant_store:
                st.warning("⚠️ Please connect to Qdrant first")
            else:
                # Get all photos
                all_photos = st.session_state.qdrant_store.get_all_photos(limit=100)

                if all_photos:
                    st.markdown("**Select a photo to update its GPS coordinates:**")

                    # Create a dropdown with photo names
                    photo_options = {f"{p.get('filename', 'Unknown')} - {p.get('datetime', 'No date')}": p for p in all_photos}
                    selected_photo_name = st.selectbox("Photo:", list(photo_options.keys()))

                    if selected_photo_name:
                        selected_photo = photo_options[selected_photo_name]

                        col1, col2 = st.columns(2)

                        with col1:
                            current_lat = selected_photo.get('latitude')
                            current_lon = selected_photo.get('longitude')

                            st.write(f"**Current coordinates:**")
                            if current_lat and current_lon:
                                st.write(f"Latitude: {current_lat}")
                                st.write(f"Longitude: {current_lon}")
                                st.markdown(f"[📍 View on Google Maps](https://www.google.com/maps/search/?api=1&query={current_lat},{current_lon})")
                            else:
                                st.write("No GPS coordinates set")

                        with col2:
                            st.write(f"**New coordinates:**")
                            new_lat = st.number_input("Latitude:", value=float(current_lat) if current_lat else 0.0, format="%.6f", key="edit_lat")
                            new_lon = st.number_input("Longitude:", value=float(current_lon) if current_lon else 0.0, format="%.6f", key="edit_lon")

                        st.markdown("**Quick location references:**")
                        location_col1, location_col2 = st.columns(2)
                        with location_col1:
                            st.write("🇮🇳 **India:**")
                            st.write("Chennai: 13.0827, 80.2707")
                            st.write("Mumbai: 19.0760, 72.8777")
                            st.write("Delhi: 28.6139, 77.2090")
                            st.write("Bangalore: 12.9716, 77.5946")
                            st.write("Kochi: 9.9312, 76.2673")
                        with location_col2:
                            st.write("🇺🇸 **United States:**")
                            st.write("New York: 40.7128, -74.0060")
                            st.write("San Francisco: 37.7749, -122.4194")
                            st.write("Los Angeles: 34.0522, -118.2437")
                            st.write("Chicago: 41.8781, -87.6298")
                            st.write("Seattle: 47.6062, -122.3321")

                        if st.button("💾 Update GPS Coordinates", type="primary"):
                            try:
                                from qdrant_client.models import SetPayload

                                st.session_state.qdrant_store.client.set_payload(
                                    collection_name="travel_photos",
                                    payload={
                                        "latitude": new_lat,
                                        "longitude": new_lon
                                    },
                                    points=[selected_photo['id']]
                                )
                                st.success(f"✅ Updated GPS coordinates to {new_lat}, {new_lon}")
                                st.info("🔄 Regenerate the journey map to see the updated location")
                            except Exception as e:
                                st.error(f"❌ Error updating coordinates: {e}")
                else:
                    st.info("No photos found in database")

        with st.expander("ℹ️ About Journey Mapping"):
            st.markdown("""
            ### How it works

            Journey Mapping creates visual routes from your travel photos by:
            1. Extracting GPS coordinates from photos
            2. Sorting by timestamp
            3. Generating Google Maps routes
            4. Creating interactive visualizations

            ### Features

            - ✅ Google Maps integration with multi-point routes
            - ✅ Embedded Google Maps view
            - ✅ Daily journey breakdowns
            - ✅ Interactive Leaflet.js maps
            - ✅ Chronological photo ordering
            - ✅ People and caption integration
            - ✅ GPS coordinate editor

            ### Tips

            - Ensure your photos have GPS metadata (EXIF)
            - Use date filters for specific trips
            - Click map markers to see photo details
            - Export HTML maps for offline viewing
            - Update GPS coordinates using the editor above
            """)

    # ============================================================================
    # TAB 7: SEMANTIC SEARCH
    # ============================================================================
    with search_tab:
        st.header("🔎 Semantic Search")
        st.markdown("Search your travel photos using natural language queries")

        # Initialize semantic search if not already done
        if st.session_state.semantic_search is None and st.session_state.qdrant_store:
            from travel_log import create_semantic_search
            st.session_state.semantic_search = create_semantic_search(st.session_state.qdrant_store)

        if not st.session_state.qdrant_store:
            st.warning("⚠️ Please connect to Qdrant first (in sidebar)")
        elif not st.session_state.semantic_search:
            st.error("❌ Semantic search not initialized")
        else:
            # Search input
            st.subheader("💬 Ask a Question")

            # Example queries
            example_queries = [
                "When, where and with whom did I see the turtles on the beach?",
                "Show me photos from my beach vacation",
                "Find photos with Sarah in Paris",
                "Beach photos from last summer",
                "Pictures of mountains and hiking"
            ]

            selected_example = st.selectbox(
                "Or try an example:",
                [""] + example_queries,
                index=0
            )

            search_query = st.text_input(
                "Search query:",
                value=selected_example if selected_example else "",
                placeholder="e.g., When did I visit the beach with John?"
            )

            col_search, col_limit = st.columns([3, 1])
            with col_search:
                search_button = st.button("🔍 Search", type="primary", use_container_width=True)
            with col_limit:
                result_limit = st.number_input("Max results", min_value=1, max_value=100, value=10)

            if search_button and search_query:
                try:
                    with st.spinner("Searching photos..."):
                        results = st.session_state.semantic_search.search(
                            query=search_query,
                            limit=result_limit
                        )

                        st.session_state.search_results = results

                        if results:
                            st.success(f"✅ Found {len(results)} matching photos")
                        else:
                            st.info("📷 No matching photos found")

                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Search error: {str(e)}")
                    st.error(f"Details: {traceback.format_exc()}")

            # Display search results
            st.divider()

            if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
                results = st.session_state.search_results

                st.subheader(f"📸 Search Results ({len(results)} photos)")

                for idx, result in enumerate(results):
                    with st.container():
                        st.divider()

                        col_img, col_details = st.columns([1, 2])

                        with col_img:
                            # Display photo if available
                            photo_path = result.get('filepath')
                            if photo_path:
                                try:
                                    photo_path_obj = Path(photo_path)
                                    if photo_path_obj.exists():
                                        img = Image.open(photo_path)
                                        st.image(img, use_container_width=True)
                                except:
                                    st.info("📷 No preview")
                            else:
                                st.info("📷 No preview")

                        with col_details:
                            # Title
                            captions = result.get('captions', {})
                            if captions and 'title' in captions:
                                st.markdown(f"### {captions['title']}")
                            else:
                                st.markdown(f"### {result.get('filename', 'Untitled')}")

                            # Relevance score
                            score = result.get('relevance_score', 0)
                            st.progress(score, text=f"Relevance: {score:.1%}")

                            # Match reasons
                            match_reasons = result.get('match_reasons', [])
                            if match_reasons:
                                st.markdown("**Why this matched:**")
                                for reason in match_reasons:
                                    st.write(f"- {reason}")

                            # Metadata
                            meta_parts = []

                            if 'datetime' in result:
                                meta_parts.append(f"📅 {result['datetime']}")

                            people = result.get('people', [])
                            if people:
                                meta_parts.append(f"👥 {', '.join(people)}")

                            if 'latitude' in result and 'longitude' in result:
                                lat, lon = result['latitude'], result['longitude']
                                maps = format_gps_for_maps(lat, lon)
                                meta_parts.append(f"[📍 Map]({maps['google_maps']})")

                            if meta_parts:
                                st.markdown(" • ".join(meta_parts))

                            # Caption
                            if captions and 'caption' in captions:
                                with st.expander("📝 Caption"):
                                    st.write(captions['caption'])

            # Context enrichment section
            st.divider()
            st.subheader("🌍 Enrich with Wikipedia Context")

            st.markdown("""
            Add location context from Wikipedia to all your photos with GPS data.
            This enables better semantic search based on place descriptions.
            """)

            # Initialize location contextualizer if needed
            if st.session_state.location_contextualizer is None and st.session_state.qdrant_store:
                from travel_log import create_location_contextualizer
                st.session_state.location_contextualizer = create_location_contextualizer(
                    st.session_state.qdrant_store
                )

            if st.button("🌍 Add Wikipedia Context to All Photos", type="secondary"):
                if not st.session_state.location_contextualizer:
                    st.error("❌ Location contextualizer not initialized")
                else:
                    try:
                        with st.spinner("Adding Wikipedia context to photos..."):
                            stats = st.session_state.location_contextualizer.bulk_contextualize_photos(limit=100)

                            st.success(f"✅ Contextualization complete!")
                            st.write(f"- Total photos: {stats['total_photos']}")
                            st.write(f"- Contextualized: {stats['contextualized']}")
                            st.write(f"- Skipped (no GPS): {stats['skipped_no_gps']}")
                            st.write(f"- Errors: {stats['errors']}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        with st.expander("ℹ️ About Semantic Search"):
            st.markdown("""
            ### How it works

            Semantic Search uses natural language processing to understand your queries and find relevant photos:

            1. **Query Parsing**: Extracts people, places, dates, and keywords
            2. **Multi-modal Filtering**: Searches across metadata, captions, locations, faces
            3. **Relevance Scoring**: Ranks results by how well they match
            4. **Context Integration**: Uses Wikipedia descriptions for richer search

            ### Supported Query Types

            **People queries:**
            - "Photos with Sarah"
            - "Show me pictures with John and Mary"

            **Location queries:**
            - "Photos from Paris"
            - "Beach photos"
            - "Pictures near the Eiffel Tower"

            **Time queries:**
            - "Photos from last summer"
            - "Pictures from 2024"
            - "January vacation photos"

            **Content queries:**
            - "Photos with turtles"
            - "Mountain hiking pictures"
            - "Sunset beach scenes"

            **Combined queries:**
            - "When, where and with whom did I see the turtles on the beach?"
            - "Show me Sarah's photos from Paris in 2024"

            ### Tips

            - Be specific with names and places
            - Use natural language - ask questions naturally
            - Add Wikipedia context for better location-based search
            - Identify faces before searching for people
            """)

if __name__ == "__main__":
    main()


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

    # Convert confidence threshold to distance threshold
    # For VGG-Face cosine distance: confidence = 1 - distance
    # So: distance_threshold = 1 - confidence_threshold
    # But we use a more lenient threshold for better matches
    distance_threshold = 0.25  # Works well for VGG-Face

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
                search_results = st.session_state.qdrant_store.client.search(
                    collection_name="reference_faces",
                    query_vector=embedding,
                    limit=3
                )

                if search_results:
                    best_match = search_results[0]
                    person_name = best_match.payload['person_name']
                    score = best_match.score  # Cosine similarity score (0-1, higher is better)
                    distance = 1 - score  # Convert to distance

                    # Convert distance to confidence percentage
                    confidence = max(0, 1 - distance)

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
                            'status': 'low_confidence'
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
                # Get the face numpy array or crop from coordinates
                facial_area = face.get('facial_area', {})
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 0)
                h = facial_area.get('h', 0)

                # Use face numpy array from DeepFace extraction if available
                if 'face' in face:
                    # Convert numpy array to PIL Image
                    face_array = face['face']
                    # DeepFace returns normalized 0-1 or 0-255 arrays
                    if face_array.max() <= 1.0:
                        face_array = (face_array * 255).astype(np.uint8)
                    else:
                        face_array = face_array.astype(np.uint8)
                    face_img = Image.fromarray(face_array)
                else:
                    # Fallback: crop from original image
                    face_img = img.crop((x, y, x + w, y + h))

                face_images.append({
                    'image': face_img,
                    'location': (x, y, w, h),
                    'confidence': face.get('confidence', 1.0)
                })
            
            # Don't clean up temp file yet - we might need it for Qdrant
            # Path(tmp_path).unlink()

            return {
                'original_image': img,
                'faces': face_images,
                'num_faces': len(filtered_faces),
                'metadata': metadata,
                'temp_path': tmp_path  # Keep path for Qdrant storage
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
    detection_tab, identification_tab, caption_tab, qdrant_tab, database_tab = st.tabs([
        "🔍 Face Detection",
        "🎯 Face Identification",
        "✍️ Image Captions",
        "🗄️ Qdrant Storage",
        "💾 Face Database"
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
            st.subheader("Database Setup")

            database_path = st.text_input(
                "Face Database Path",
                value="./face_database",
                help="Path to directory containing labeled face images"
            )

            model_options = ['VGG-Face', 'Facenet512', 'Facenet', 'ArcFace', 'DeepFace',
                           'OpenFace', 'DeepID', 'Dlib', 'SFace']

            recognition_model = st.selectbox(
                "Recognition Model",
                model_options,
                index=0,
                help="Face recognition model to use (VGG-Face recommended - works best with GPU)"
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
    # TAB 3: IMAGE CAPTIONS
    # ============================================================================
    with caption_tab:
        st.header("✍️ Image Captions & Titles")
        st.markdown("Generate AI-powered captions and titles for your travel photos using LLaVA")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📤 Upload Image for Captions")

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

            # Determine supported file types
            supported_types = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
            if HEIC_SUPPORTED:
                supported_types.extend(['heic', 'heif'])

            uploaded_file = st.file_uploader(
                "Choose an image file for captions",
                type=supported_types,
                help="Upload a photo to generate captions"
            )

            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

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
                            # Load image
                            image = Image.open(uploaded_file)

                            # Convert HEIC to JPEG if necessary (for file processing)
                            file_ext = Path(uploaded_file.name).suffix.lower()
                            if file_ext in ['.heic', '.heif']:
                                # Convert HEIC file to PIL Image via temp conversion
                                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    tmp_path = tmp_file.name
                                with st.spinner("Converting HEIC image..."):
                                    tmp_path = ensure_compatible_image(tmp_path)
                                image = Image.open(tmp_path)
                                Path(tmp_path).unlink()

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

                # Show what will be saved
                with st.expander("📋 Data to be saved"):
                    data_summary = []
                    if st.session_state.image_metadata:
                        data_summary.append("✅ EXIF metadata")
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


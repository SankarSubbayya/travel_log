"""
Face Detection Module for Travel Log

This module handles face detection and extraction from images, including group photos.
It uses DeepFace library with multiple backend options for robust face detection.
"""

# Suppress TensorFlow warnings before imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detects and extracts faces from images using DeepFace.
    
    Supports multiple detection backends for flexibility and robustness:
    - opencv: Fast, good for simple cases
    - ssd: Faster than MTCNN, decent accuracy
    - mtcnn: Robust, handles various poses and lighting
    - retinaface: High accuracy, slower
    - dlib: Classic approach, reliable
    """
    
    SUPPORTED_BACKENDS = ['opencv', 'ssd', 'mtcnn', 'retinaface', 'dlib']
    
    def __init__(
        self, 
        detector_backend: str = 'mtcnn',
        align: bool = True,
        expand_percentage: int = 0
    ):
        """
        Initialize the FaceDetector.
        
        Args:
            detector_backend: Face detection backend to use
            align: Whether to align faces before extraction
            expand_percentage: Expand face region by this percentage
        """
        if detector_backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Backend must be one of {self.SUPPORTED_BACKENDS}, "
                f"got {detector_backend}"
            )
        
        self.detector_backend = detector_backend
        self.align = align
        self.expand_percentage = expand_percentage
        logger.info(f"Initialized FaceDetector with backend: {detector_backend}")
    
    def extract_faces(
        self, 
        image_path: Union[str, Path],
        grayscale: bool = False
    ) -> List[Dict]:
        """
        Extract all faces from an image.
        
        Args:
            image_path: Path to the image file
            grayscale: Convert faces to grayscale
            
        Returns:
            List of dictionaries containing:
                - 'face': numpy array of face image
                - 'facial_area': dict with x, y, w, h coordinates
                - 'confidence': detection confidence score
        """
        image_path = str(image_path)
        logger.info(f"Extracting faces from: {image_path}")
        
        try:
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                align=self.align,
                expand_percentage=self.expand_percentage,
                grayscale=grayscale
            )
            
            logger.info(f"Detected {len(faces)} face(s) in {image_path}")
            return faces
            
        except Exception as e:
            logger.error(f"Error extracting faces from {image_path}: {str(e)}")
            return []
    
    def save_extracted_faces(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        prefix: str = "face"
    ) -> List[Path]:
        """
        Extract faces from an image and save them to disk.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save extracted faces
            prefix: Prefix for saved face files
            
        Returns:
            List of paths to saved face images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        faces = self.extract_faces(image_path)
        saved_paths = []
        
        image_name = Path(image_path).stem
        
        for i, face_data in enumerate(faces):
            face_img = face_data['face']
            
            # DeepFace returns images in RGB with values 0-1, convert to 0-255
            if face_img.max() <= 1.0:
                face_img = (face_img * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            
            # Save the face
            output_path = output_dir / f"{prefix}_{image_name}_{i:03d}.jpg"
            cv2.imwrite(str(output_path), face_img_bgr)
            saved_paths.append(output_path)
            
            logger.info(f"Saved face to: {output_path}")
        
        return saved_paths
    
    def get_face_locations(
        self, 
        image_path: Union[str, Path]
    ) -> List[Dict[str, int]]:
        """
        Get bounding box coordinates for all faces in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries with 'x', 'y', 'w', 'h' coordinates
        """
        faces = self.extract_faces(image_path)
        return [face['facial_area'] for face in faces]
    
    def annotate_image(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            image_path: Path to the input image
            output_path: Optional path to save annotated image
            color: BGR color for bounding boxes
            thickness: Thickness of bounding box lines
            
        Returns:
            Annotated image as numpy array
        """
        # Read the original image
        img = cv2.imread(str(image_path))
        
        # Get face locations
        face_locations = self.get_face_locations(image_path)
        
        # Draw rectangles
        for face_area in face_locations:
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(str(output_path), img)
            logger.info(f"Saved annotated image to: {output_path}")
        
        return img
    
    def batch_extract_faces(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path]
    ) -> Dict[str, List[Path]]:
        """
        Extract faces from multiple images.
        
        Args:
            image_paths: List of image paths to process
            output_dir: Directory to save extracted faces
            
        Returns:
            Dictionary mapping input image path to list of extracted face paths
        """
        results = {}
        
        for image_path in image_paths:
            try:
                saved_paths = self.save_extracted_faces(
                    image_path, 
                    output_dir,
                    prefix=f"face_{Path(image_path).stem}"
                )
                results[str(image_path)] = saved_paths
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                results[str(image_path)] = []
        
        return results


def extract_faces_from_photo(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    detector_backend: str = 'mtcnn'
) -> List[Path]:
    """
    Convenience function to extract faces from a single photo.
    
    Args:
        image_path: Path to the image
        output_dir: Directory to save extracted faces
        detector_backend: Detection backend to use
        
    Returns:
        List of paths to saved face images
    """
    detector = FaceDetector(detector_backend=detector_backend)
    return detector.save_extracted_faces(image_path, output_dir)


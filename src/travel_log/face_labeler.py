"""
Face Labeling Module for Travel Log

This module handles face identification and labeling by comparing against a database
of known faces. It supports both automatic recognition and manual labeling workflows.
"""

# Suppress TensorFlow warnings before imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from deepface import DeepFace
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceLabeler:
    """
    Labels and identifies faces by comparing against a database of known individuals.
    
    The database is expected to be a directory with subdirectories named after individuals,
    containing sample images of each person:
    
    database/
        ├── person1/
        │   ├── photo1.jpg
        │   ├── photo2.jpg
        ├── person2/
        │   ├── photo1.jpg
        └── unknown/
    """
    
    DISTANCE_METRICS = ['cosine', 'euclidean', 'euclidean_l2']
    
    def __init__(
        self,
        database_path: Union[str, Path],
        model_name: str = 'Facenet512',
        detector_backend: str = 'mtcnn',
        distance_metric: str = 'cosine',
        enforce_detection: bool = True
    ):
        """
        Initialize the FaceLabeler.
        
        Args:
            database_path: Path to directory containing labeled face images
            model_name: Face recognition model to use
            detector_backend: Face detection backend
            distance_metric: Distance metric for comparison
            enforce_detection: Whether to enforce face detection
        """
        self.database_path = Path(database_path)
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.enforce_detection = enforce_detection
        
        if not self.database_path.exists():
            logger.warning(f"Database path does not exist: {self.database_path}")
            logger.warning("Creating database directory...")
            self.database_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FaceLabeler with database: {database_path}")
    
    def find_face(
        self,
        image_path: Union[str, Path],
        threshold: Optional[float] = None
    ) -> List[pd.DataFrame]:
        """
        Find matching faces in the database for a given image.
        
        Args:
            image_path: Path to query face image
            threshold: Distance threshold for matching (None = use model default)
            
        Returns:
            List of DataFrames with matches for each detected face
        """
        image_path = str(image_path)
        logger.info(f"Searching for face in: {image_path}")
        
        try:
            results = DeepFace.find(
                img_path=image_path,
                db_path=str(self.database_path),
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=self.enforce_detection,
                silent=False
            )
            
            # Apply threshold if provided
            if threshold is not None and results:
                filtered_results = []
                for df in results:
                    if not df.empty:
                        # Filter by threshold
                        mask = df[self.distance_metric] <= threshold
                        filtered_results.append(df[mask])
                    else:
                        filtered_results.append(df)
                results = filtered_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding face in {image_path}: {str(e)}")
            return []
    
    def identify_face(
        self,
        image_path: Union[str, Path],
        threshold: Optional[float] = None,
        return_confidence: bool = True
    ) -> Optional[Dict]:
        """
        Identify a face and return the best match.
        
        Args:
            image_path: Path to query face image
            threshold: Distance threshold for matching
            return_confidence: Include confidence score in result
            
        Returns:
            Dictionary with identification results or None if no match
        """
        results = self.find_face(image_path, threshold)
        
        if not results or len(results) == 0:
            logger.info(f"No matches found for {image_path}")
            return None
        
        # Get the first result (first detected face)
        df = results[0]
        
        if df.empty:
            logger.info(f"No matches above threshold for {image_path}")
            return None
        
        # Get best match (lowest distance)
        best_match = df.iloc[0]
        identity_path = Path(best_match['identity'])
        
        # Extract person name from path (parent directory name)
        person_name = identity_path.parent.name
        
        result = {
            'name': person_name,
            'match_image': str(identity_path),
            'query_image': str(image_path)
        }
        
        if return_confidence:
            distance = best_match[self.distance_metric]
            # Convert distance to confidence (0-1, higher is better)
            if self.distance_metric == 'cosine':
                confidence = 1 - distance
            else:
                # For euclidean, we'll use inverse (approximate)
                confidence = 1 / (1 + distance)
            
            result['confidence'] = float(confidence)
            result['distance'] = float(distance)
        
        logger.info(f"Identified face as: {person_name}")
        return result
    
    def identify_faces_batch(
        self,
        image_paths: List[Union[str, Path]],
        threshold: Optional[float] = None
    ) -> List[Optional[Dict]]:
        """
        Identify faces in multiple images.
        
        Args:
            image_paths: List of query image paths
            threshold: Distance threshold for matching
            
        Returns:
            List of identification results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.identify_face(image_path, threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error identifying {image_path}: {str(e)}")
                results.append(None)
        
        return results
    
    def add_person(
        self,
        person_name: str,
        image_paths: List[Union[str, Path]]
    ) -> Path:
        """
        Add a new person to the database.
        
        Args:
            person_name: Name of the person
            image_paths: List of sample images of the person
            
        Returns:
            Path to the person's directory in the database
        """
        person_dir = self.database_path / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        for i, image_path in enumerate(image_paths):
            image_path = Path(image_path)
            destination = person_dir / f"{person_name}_{i}{image_path.suffix}"
            shutil.copy2(image_path, destination)
            logger.info(f"Added image to database: {destination}")
        
        logger.info(f"Added person '{person_name}' with {len(image_paths)} images")
        return person_dir
    
    def list_known_people(self) -> List[str]:
        """
        List all people in the database.
        
        Returns:
            List of person names
        """
        if not self.database_path.exists():
            return []
        
        people = [
            d.name for d in self.database_path.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        
        return sorted(people)
    
    def get_person_images(self, person_name: str) -> List[Path]:
        """
        Get all images for a specific person.
        
        Args:
            person_name: Name of the person
            
        Returns:
            List of image paths
        """
        person_dir = self.database_path / person_name
        if not person_dir.exists():
            return []
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images.extend(person_dir.glob(ext))
        
        return sorted(images)
    
    def verify_faces(
        self,
        image1_path: Union[str, Path],
        image2_path: Union[str, Path]
    ) -> Dict:
        """
        Verify if two images contain the same person.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Dictionary with verification results
        """
        try:
            result = DeepFace.verify(
                img1_path=str(image1_path),
                img2_path=str(image2_path),
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=self.enforce_detection
            )
            
            return {
                'verified': result['verified'],
                'distance': result['distance'],
                'threshold': result['threshold'],
                'model': self.model_name,
                'metric': self.distance_metric
            }
            
        except Exception as e:
            logger.error(f"Error verifying faces: {str(e)}")
            return {
                'verified': False,
                'error': str(e)
            }
    
    def create_label_mapping(
        self,
        face_images: List[Union[str, Path]],
        output_path: Union[str, Path],
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Create a mapping of face images to identified labels.
        
        Args:
            face_images: List of face image paths to label
            output_path: Path to save the mapping CSV
            threshold: Distance threshold for matching
            
        Returns:
            DataFrame with face-to-label mappings
        """
        mappings = []
        
        for image_path in face_images:
            result = self.identify_face(image_path, threshold)
            
            mappings.append({
                'face_image': str(image_path),
                'identified_as': result['name'] if result else 'unknown',
                'confidence': result.get('confidence', 0.0) if result else 0.0,
                'distance': result.get('distance', float('inf')) if result else float('inf'),
                'timestamp': datetime.now().isoformat()
            })
        
        df = pd.DataFrame(mappings)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved label mapping to {output_path}")
        
        return df
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the face database.
        
        Returns:
            Dictionary with database statistics
        """
        people = self.list_known_people()
        stats = {
            'total_people': len(people),
            'people': {}
        }
        
        for person in people:
            images = self.get_person_images(person)
            stats['people'][person] = len(images)
        
        stats['total_images'] = sum(stats['people'].values())
        
        return stats


def identify_faces_in_photos(
    face_images_dir: Union[str, Path],
    database_path: Union[str, Path],
    output_csv: Union[str, Path],
    model_name: str = 'Facenet512'
) -> pd.DataFrame:
    """
    Convenience function to identify all faces in a directory.
    
    Args:
        face_images_dir: Directory containing face images to identify
        database_path: Path to the face database
        output_csv: Path to save results
        model_name: Model to use for recognition
        
    Returns:
        DataFrame with identification results
    """
    labeler = FaceLabeler(database_path, model_name=model_name)
    
    face_images_dir = Path(face_images_dir)
    face_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        face_images.extend(face_images_dir.glob(ext))
    
    return labeler.create_label_mapping(face_images, output_csv)


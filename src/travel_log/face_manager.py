"""
Face Manager - Orchestrator Module for Travel Log

This module provides a high-level interface that orchestrates all face detection,
recognition, labeling, and embedding functionality. It's the main entry point for
working with faces in travel photos.
"""

from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from .face_detector import FaceDetector
from .face_embeddings import FaceEmbeddings
from .face_labeler import FaceLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TravelLogFaceManager:
    """
    High-level manager for all face-related operations in travel photos.
    
    This class orchestrates:
    1. Face detection and extraction from group photos
    2. Face recognition and labeling
    3. Face embedding generation
    4. Face clustering and organization
    """
    
    def __init__(
        self,
        workspace_dir: Union[str, Path],
        database_dir: Optional[Union[str, Path]] = None,
        detector_backend: str = 'mtcnn',
        recognition_model: str = 'Facenet512'
    ):
        """
        Initialize the TravelLogFaceManager.
        
        Args:
            workspace_dir: Working directory for all face operations
            database_dir: Directory for known faces database (default: workspace_dir/face_database)
            detector_backend: Face detection backend to use
            recognition_model: Face recognition model to use
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up directory structure
        self.faces_dir = self.workspace_dir / "extracted_faces"
        self.embeddings_dir = self.workspace_dir / "embeddings"
        self.results_dir = self.workspace_dir / "results"
        
        for dir_path in [self.faces_dir, self.embeddings_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up database directory
        if database_dir is None:
            self.database_dir = self.workspace_dir / "face_database"
        else:
            self.database_dir = Path(database_dir)
        self.database_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.detector = FaceDetector(detector_backend=detector_backend)
        self.embeddings = FaceEmbeddings(
            model_name=recognition_model,
            detector_backend=detector_backend
        )
        self.labeler = FaceLabeler(
            database_path=self.database_dir,
            model_name=recognition_model,
            detector_backend=detector_backend
        )
        
        logger.info(f"Initialized TravelLogFaceManager at {workspace_dir}")
    
    def process_photo(
        self,
        image_path: Union[str, Path],
        extract_faces: bool = True,
        identify_faces: bool = True,
        generate_embeddings: bool = True
    ) -> Dict:
        """
        Process a single photo - extract, identify, and embed faces.
        
        Args:
            image_path: Path to the photo
            extract_faces: Whether to extract faces
            identify_faces: Whether to identify faces
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            Dictionary with processing results
        """
        image_path = Path(image_path)
        logger.info(f"Processing photo: {image_path.name}")
        
        result = {
            'source_image': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'faces': []
        }
        
        # Step 1: Extract faces
        if extract_faces:
            logger.info("Extracting faces...")
            face_paths = self.detector.save_extracted_faces(
                image_path,
                self.faces_dir,
                prefix=image_path.stem
            )
            result['extracted_faces'] = [str(p) for p in face_paths]
            result['num_faces'] = len(face_paths)
        else:
            face_paths = []
            result['extracted_faces'] = []
            result['num_faces'] = 0
        
        # Step 2 & 3: Identify and generate embeddings for each face
        for i, face_path in enumerate(face_paths):
            face_data = {
                'face_index': i,
                'face_path': str(face_path)
            }
            
            # Identify face
            if identify_faces:
                try:
                    identification = self.labeler.identify_face(face_path)
                    if identification:
                        face_data['identified_as'] = identification['name']
                        face_data['confidence'] = identification['confidence']
                        face_data['distance'] = identification['distance']
                    else:
                        face_data['identified_as'] = 'unknown'
                        face_data['confidence'] = 0.0
                except Exception as e:
                    logger.error(f"Error identifying face: {str(e)}")
                    face_data['identified_as'] = 'error'
                    face_data['error'] = str(e)
            
            # Generate embedding
            if generate_embeddings:
                try:
                    embedding_data = self.embeddings.generate_embedding(face_path)
                    if embedding_data:
                        face_data['embedding_dimension'] = embedding_data['dimension']
                        face_data['has_embedding'] = True
                        
                        # Save embedding separately
                        embedding_file = self.embeddings_dir / f"{face_path.stem}.npy"
                        np.save(embedding_file, embedding_data['embedding'])
                        face_data['embedding_file'] = str(embedding_file)
                    else:
                        face_data['has_embedding'] = False
                except Exception as e:
                    logger.error(f"Error generating embedding: {str(e)}")
                    face_data['has_embedding'] = False
                    face_data['embedding_error'] = str(e)
            
            result['faces'].append(face_data)
        
        logger.info(f"Processed {len(face_paths)} faces from {image_path.name}")
        return result
    
    def process_photos_batch(
        self,
        image_paths: List[Union[str, Path]],
        save_report: bool = True
    ) -> List[Dict]:
        """
        Process multiple photos in batch.
        
        Args:
            image_paths: List of image paths to process
            save_report: Whether to save a summary report
            
        Returns:
            List of processing results for each photo
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.process_photo(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'source_image': str(image_path),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save report
        if save_report:
            report_path = self.results_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            import json
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved batch report to {report_path}")
        
        return results
    
    def process_directory(
        self,
        photos_dir: Union[str, Path],
        pattern: str = "*.jpg"
    ) -> List[Dict]:
        """
        Process all photos in a directory.
        
        Args:
            photos_dir: Directory containing photos
            pattern: File pattern to match
            
        Returns:
            List of processing results
        """
        photos_dir = Path(photos_dir)
        image_paths = list(photos_dir.glob(pattern))
        
        logger.info(f"Found {len(image_paths)} images matching '{pattern}' in {photos_dir}")
        return self.process_photos_batch(image_paths)
    
    def add_person_to_database(
        self,
        person_name: str,
        sample_images: List[Union[str, Path]]
    ) -> Dict:
        """
        Add a new person to the recognition database.
        
        Args:
            person_name: Name of the person
            sample_images: List of sample face images of the person
            
        Returns:
            Dictionary with addition results
        """
        person_dir = self.labeler.add_person(person_name, sample_images)
        
        # Generate embeddings for the person's images
        person_images = self.labeler.get_person_images(person_name)
        embeddings = self.embeddings.generate_embeddings_batch(person_images)
        
        # Save person's embeddings
        if embeddings:
            embedding_file = self.embeddings_dir / f"{person_name}_embeddings.pkl"
            self.embeddings.save_embeddings(embeddings, embedding_file)
        
        return {
            'person_name': person_name,
            'person_dir': str(person_dir),
            'num_samples': len(sample_images),
            'embeddings_generated': len(embeddings)
        }
    
    def get_face_clusters(
        self,
        threshold: float = 0.6,
        metric: str = 'cosine'
    ) -> Dict[str, List[str]]:
        """
        Cluster extracted faces by similarity.
        
        Args:
            threshold: Similarity threshold for clustering
            metric: Distance metric to use
            
        Returns:
            Dictionary mapping cluster_id to list of face image paths
        """
        # Load all embeddings
        embedding_files = list(self.embeddings_dir.glob("*.npy"))
        
        if not embedding_files:
            logger.warning("No embeddings found. Process photos first.")
            return {}
        
        embeddings = []
        paths = []
        
        for emb_file in embedding_files:
            try:
                embedding = np.load(emb_file)
                embeddings.append(embedding)
                paths.append(str(emb_file.stem))
            except Exception as e:
                logger.error(f"Error loading {emb_file}: {str(e)}")
        
        if not embeddings:
            return {}
        
        # Simple clustering using similarity threshold
        embeddings_array = np.array(embeddings)
        clusters = {}
        cluster_id = 0
        assigned = set()
        
        for i, emb1 in enumerate(embeddings_array):
            if i in assigned:
                continue
            
            # Start new cluster
            cluster_members = [paths[i]]
            assigned.add(i)
            
            # Find similar faces
            for j, emb2 in enumerate(embeddings_array):
                if j in assigned or i == j:
                    continue
                
                similarity = self.embeddings.compute_similarity(
                    emb1, emb2, metric=metric
                )
                
                # Check if similar enough
                if metric == 'cosine' and similarity >= threshold:
                    cluster_members.append(paths[j])
                    assigned.add(j)
                elif metric != 'cosine' and similarity <= threshold:
                    cluster_members.append(paths[j])
                    assigned.add(j)
            
            clusters[f"cluster_{cluster_id}"] = cluster_members
            cluster_id += 1
        
        logger.info(f"Created {len(clusters)} face clusters")
        return clusters
    
    def generate_summary_report(self) -> Dict:
        """
        Generate a summary report of all processed faces.
        
        Returns:
            Dictionary with summary statistics
        """
        # Count extracted faces
        extracted_faces = list(self.faces_dir.glob("*.jpg"))
        
        # Count embeddings
        embeddings = list(self.embeddings_dir.glob("*.npy"))
        
        # Get database stats
        db_stats = self.labeler.get_database_stats()
        
        # Analyze identification results
        identified_count = 0
        unknown_count = 0
        
        for face_path in extracted_faces:
            try:
                result = self.labeler.identify_face(face_path)
                if result:
                    identified_count += 1
                else:
                    unknown_count += 1
            except:
                unknown_count += 1
        
        summary = {
            'workspace': str(self.workspace_dir),
            'timestamp': datetime.now().isoformat(),
            'extracted_faces': len(extracted_faces),
            'generated_embeddings': len(embeddings),
            'identified_faces': identified_count,
            'unknown_faces': unknown_count,
            'database': db_stats
        }
        
        # Save summary
        summary_path = self.results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary report to {summary_path}")
        
        return summary
    
    def export_labeled_dataset(
        self,
        output_dir: Union[str, Path]
    ) -> Dict:
        """
        Export all extracted and labeled faces as a organized dataset.
        
        Args:
            output_dir: Directory to export the dataset
            
        Returns:
            Dictionary with export statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_faces = list(self.faces_dir.glob("*.jpg"))
        export_stats = {'total': len(extracted_faces), 'by_label': {}}
        
        import shutil
        
        for face_path in extracted_faces:
            try:
                result = self.labeler.identify_face(face_path)
                label = result['name'] if result else 'unknown'
            except:
                label = 'unknown'
            
            # Create label directory
            label_dir = output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy face to labeled directory
            dest_path = label_dir / face_path.name
            shutil.copy2(face_path, dest_path)
            
            # Update stats
            export_stats['by_label'][label] = export_stats['by_label'].get(label, 0) + 1
        
        logger.info(f"Exported {export_stats['total']} faces to {output_dir}")
        return export_stats


def create_face_manager(
    workspace_dir: Union[str, Path],
    detector_backend: str = 'mtcnn',
    recognition_model: str = 'Facenet512'
) -> TravelLogFaceManager:
    """
    Convenience function to create a TravelLogFaceManager instance.
    
    Args:
        workspace_dir: Working directory for face operations
        detector_backend: Face detection backend
        recognition_model: Face recognition model
        
    Returns:
        Configured TravelLogFaceManager instance
    """
    return TravelLogFaceManager(
        workspace_dir=workspace_dir,
        detector_backend=detector_backend,
        recognition_model=recognition_model
    )


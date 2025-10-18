"""
Face Embeddings Module for Travel Log

This module handles generation of face embeddings (signature vectors) for faces.
These embeddings can be used for face recognition, clustering, and similarity search.
"""

# Suppress TensorFlow warnings before imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from typing import List, Dict, Union, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from deepface import DeepFace
import pickle
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceEmbeddings:
    """
    Generates and manages face embeddings using DeepFace.
    
    Supports multiple recognition models:
    - VGG-Face: Classic, reliable
    - Facenet: Google's model, good balance
    - Facenet512: Higher dimensional, more accurate
    - OpenFace: Lightweight
    - DeepFace: Facebook's model
    - DeepID: High accuracy
    - ArcFace: State-of-the-art accuracy
    - Dlib: Classic approach
    - SFace: Optimized for speed
    """
    
    SUPPORTED_MODELS = [
        'VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 
        'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace'
    ]
    
    def __init__(
        self,
        model_name: str = 'Facenet512',
        detector_backend: str = 'mtcnn',
        normalization: str = 'base'
    ):
        """
        Initialize the FaceEmbeddings generator.
        
        Args:
            model_name: Face recognition model to use
            detector_backend: Face detection backend
            normalization: Normalization method ('base', 'raw', 'Facenet', 'Facenet2018')
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model must be one of {self.SUPPORTED_MODELS}, "
                f"got {model_name}"
            )
        
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.normalization = normalization
        logger.info(f"Initialized FaceEmbeddings with model: {model_name}")
    
    def generate_embedding(
        self,
        image_path: Union[str, Path],
        enforce_detection: bool = True
    ) -> Optional[Dict]:
        """
        Generate embedding for a single face image.
        
        Args:
            image_path: Path to face image
            enforce_detection: Raise error if no face detected
            
        Returns:
            Dictionary containing:
                - 'embedding': numpy array of face embedding
                - 'model': name of the model used
                - 'dimension': dimension of the embedding
        """
        image_path = str(image_path)
        logger.info(f"Generating embedding for: {image_path}")
        
        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=enforce_detection,
                normalization=self.normalization
            )
            
            # DeepFace.represent returns a list of embeddings
            if result and len(result) > 0:
                embedding_data = result[0]
                embedding_vector = np.array(embedding_data['embedding'])
                
                return {
                    'embedding': embedding_vector,
                    'model': self.model_name,
                    'dimension': len(embedding_vector),
                    'image_path': image_path
                }
            else:
                logger.warning(f"No embedding generated for {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embedding for {image_path}: {str(e)}")
            return None
    
    def generate_embeddings_batch(
        self,
        image_paths: List[Union[str, Path]],
        skip_errors: bool = True
    ) -> List[Dict]:
        """
        Generate embeddings for multiple face images.
        
        Args:
            image_paths: List of image paths
            skip_errors: Continue on errors vs raise exception
            
        Returns:
            List of embedding dictionaries
        """
        embeddings = []
        
        for image_path in image_paths:
            try:
                embedding = self.generate_embedding(
                    image_path,
                    enforce_detection=not skip_errors
                )
                if embedding:
                    embeddings.append(embedding)
            except Exception as e:
                if not skip_errors:
                    raise
                logger.error(f"Skipping {image_path} due to error: {str(e)}")
        
        logger.info(f"Generated {len(embeddings)} embeddings from {len(image_paths)} images")
        return embeddings
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine', 'euclidean', 'euclidean_l2')
            
        Returns:
            Similarity score (lower is more similar for distance metrics)
        """
        if metric == 'cosine':
            # Cosine similarity: 1 - cosine distance
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            cosine_sim = dot_product / (norm1 * norm2)
            return float(cosine_sim)
        
        elif metric == 'euclidean':
            return float(np.linalg.norm(embedding1 - embedding2))
        
        elif metric == 'euclidean_l2':
            # Normalized euclidean
            return float(np.sqrt(np.sum((embedding1 - embedding2) ** 2)))
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[Dict],
        top_k: int = 5,
        metric: str = 'cosine'
    ) -> List[Dict]:
        """
        Find most similar faces to a query embedding.
        
        Args:
            query_embedding: Query face embedding
            candidate_embeddings: List of candidate embedding dictionaries
            top_k: Number of top matches to return
            metric: Similarity metric to use
            
        Returns:
            List of top-k most similar embeddings with similarity scores
        """
        similarities = []
        
        for candidate in candidate_embeddings:
            similarity = self.compute_similarity(
                query_embedding,
                candidate['embedding'],
                metric=metric
            )
            
            similarities.append({
                **candidate,
                'similarity': similarity
            })
        
        # Sort by similarity (descending for cosine, ascending for distance)
        reverse = (metric == 'cosine')
        similarities.sort(key=lambda x: x['similarity'], reverse=reverse)
        
        return similarities[:top_k]
    
    def save_embeddings(
        self,
        embeddings: List[Dict],
        output_path: Union[str, Path],
        format: str = 'pickle'
    ) -> None:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: List of embedding dictionaries
            output_path: Path to save embeddings
            format: Format to save ('pickle' or 'npz')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings to {output_path} (pickle format)")
        
        elif format == 'npz':
            # Save as numpy compressed format
            embeddings_array = np.array([e['embedding'] for e in embeddings])
            metadata = [
                {k: v for k, v in e.items() if k != 'embedding'}
                for e in embeddings
            ]
            
            np.savez_compressed(
                output_path,
                embeddings=embeddings_array,
                metadata=json.dumps(metadata)
            )
            logger.info(f"Saved embeddings to {output_path} (npz format)")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_embeddings(
        self,
        input_path: Union[str, Path],
        format: str = 'pickle'
    ) -> List[Dict]:
        """
        Load embeddings from disk.
        
        Args:
            input_path: Path to embeddings file
            format: Format of the file ('pickle' or 'npz')
            
        Returns:
            List of embedding dictionaries
        """
        input_path = Path(input_path)
        
        if format == 'pickle':
            with open(input_path, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"Loaded {len(embeddings)} embeddings from {input_path}")
            return embeddings
        
        elif format == 'npz':
            data = np.load(input_path, allow_pickle=True)
            embeddings_array = data['embeddings']
            metadata = json.loads(str(data['metadata']))
            
            embeddings = []
            for emb, meta in zip(embeddings_array, metadata):
                embeddings.append({
                    'embedding': emb,
                    **meta
                })
            
            logger.info(f"Loaded {len(embeddings)} embeddings from {input_path}")
            return embeddings
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def create_embedding_database(
        self,
        face_images_dir: Union[str, Path],
        output_path: Union[str, Path],
        pattern: str = "*.jpg"
    ) -> List[Dict]:
        """
        Create an embedding database from a directory of face images.
        
        Args:
            face_images_dir: Directory containing face images
            output_path: Path to save embedding database
            pattern: File pattern to match (e.g., "*.jpg", "*.png")
            
        Returns:
            List of generated embeddings
        """
        face_images_dir = Path(face_images_dir)
        image_paths = list(face_images_dir.glob(pattern))
        
        logger.info(f"Creating embedding database from {len(image_paths)} images")
        
        embeddings = self.generate_embeddings_batch(image_paths)
        self.save_embeddings(embeddings, output_path)
        
        return embeddings


def generate_face_embedding(
    image_path: Union[str, Path],
    model_name: str = 'Facenet512'
) -> Optional[np.ndarray]:
    """
    Convenience function to generate a single face embedding.
    
    Args:
        image_path: Path to face image
        model_name: Model to use for embedding
        
    Returns:
        Embedding vector as numpy array or None if failed
    """
    embedder = FaceEmbeddings(model_name=model_name)
    result = embedder.generate_embedding(image_path)
    return result['embedding'] if result else None


"""
Qdrant Vector Database Integration for Travel Log

Store photos with:
- Face embeddings (Facenet512 - 512 dimensions)
- EXIF metadata (GPS, timestamp, camera info)
- Detected faces (bounding boxes, confidence)
- Face identifications (names, confidence)
- Generated captions (LLaVA, DSPy)
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import numpy as np
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

logger = logging.getLogger(__name__)


class TravelLogQdrantStore:
    """
    Qdrant storage for travel photos and metadata.

    Schema:
    - Vector: Average face embedding (512D for Facenet512)
    - Payload: All metadata (EXIF, faces, captions, etc.)
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "travel_photos",
        faces_collection_name: str = "detected_faces",
        embedding_dim: int = 512  # Facenet512 dimension
    ):
        """
        Initialize Qdrant client and collections.

        Args:
            qdrant_url: Qdrant server URL
            collection_name: Collection name for photos
            faces_collection_name: Collection name for individual faces
            embedding_dim: Face embedding dimension (512 for Facenet512)
        """
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.faces_collection_name = faces_collection_name
        self.embedding_dim = embedding_dim

        # Initialize collections
        self._setup_collection()
        self._setup_faces_collection()

    def _setup_collection(self):
        """Create photos collection if it doesn't exist."""
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✓ Created collection: {self.collection_name}")
            else:
                logger.info(f"✓ Using existing collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise

    def _setup_faces_collection(self):
        """Create faces collection if it doesn't exist."""
        try:
            if not self.client.collection_exists(self.faces_collection_name):
                self.client.create_collection(
                    collection_name=self.faces_collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✓ Created faces collection: {self.faces_collection_name}")
            else:
                logger.info(f"✓ Using existing faces collection: {self.faces_collection_name}")
        except Exception as e:
            logger.error(f"Error setting up faces collection: {e}")
            raise

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def store_photo(
        self,
        photo_path: Union[str, Path],
        face_embedding: Optional[np.ndarray] = None,
        exif_metadata: Optional[Dict] = None,
        detected_faces: Optional[List[Dict]] = None,
        face_identifications: Optional[Dict] = None,
        captions: Optional[Dict] = None,
        custom_id: Optional[str] = None
    ) -> str:
        """
        Store a photo with all associated data.

        Args:
            photo_path: Path to photo file
            face_embedding: Average face embedding (512D numpy array)
            exif_metadata: EXIF data dictionary
            detected_faces: List of detected face data
            face_identifications: Dict of face index -> identification
            captions: Generated captions dictionary
            custom_id: Custom point ID (default: auto-generated)

        Returns:
            Point ID in Qdrant
        """
        photo_path = Path(photo_path)

        # Generate point ID (UUID format required by Qdrant)
        if custom_id is None:
            point_id = str(uuid.uuid4())
        else:
            # Try to use custom_id as UUID, generate new one if invalid
            try:
                point_id = str(uuid.UUID(custom_id))
            except:
                point_id = str(uuid.uuid4())

        # Prepare payload
        payload = {
            "filename": photo_path.name,
            "filepath": str(photo_path.absolute()),
            "upload_timestamp": datetime.now().isoformat(),
        }

        # Add EXIF metadata (convert numpy types)
        if exif_metadata:
            exif_data = {}

            if "datetime_str" in exif_metadata:
                exif_data["datetime"] = str(exif_metadata["datetime_str"])

            if "datetime" in exif_metadata:
                exif_data["datetime_iso"] = exif_metadata["datetime"].isoformat()

            if "camera" in exif_metadata:
                camera = exif_metadata["camera"]
                if camera:
                    exif_data["camera_make"] = str(camera.get("camera_make", ""))
                    exif_data["camera_model"] = str(camera.get("camera_model", ""))
                    if "iso" in camera and camera["iso"] is not None:
                        exif_data["iso"] = int(camera["iso"]) if camera["iso"] else None
                    if "aperture" in camera and camera["aperture"]:
                        exif_data["aperture"] = str(camera["aperture"])

            if "width" in exif_metadata:
                exif_data["width"] = int(exif_metadata["width"])
            if "height" in exif_metadata:
                exif_data["height"] = int(exif_metadata["height"])
            if "orientation" in exif_metadata:
                exif_data["orientation"] = str(exif_metadata["orientation"])

            payload["exif"] = exif_data

            # GPS data (for geospatial queries) - convert to float
            if "latitude" in exif_metadata and "longitude" in exif_metadata:
                payload["latitude"] = float(exif_metadata["latitude"])
                payload["longitude"] = float(exif_metadata["longitude"])

                if "altitude" in exif_metadata:
                    payload["altitude"] = float(exif_metadata["altitude"])

        # Add detected faces (convert numpy types)
        if detected_faces:
            faces_data = []
            for i, face in enumerate(detected_faces):
                face_info = {
                    "index": int(i),
                    "confidence": float(face.get("confidence", 1.0))
                }
                if "facial_area" in face:
                    bbox = face["facial_area"]
                    face_info["bbox"] = {
                        "x": int(bbox.get("x", 0)),
                        "y": int(bbox.get("y", 0)),
                        "w": int(bbox.get("w", 0)),
                        "h": int(bbox.get("h", 0))
                    }
                faces_data.append(face_info)

            payload["detected_faces"] = faces_data
            payload["num_faces"] = int(len(detected_faces))

        # Add face identifications (convert numpy types)
        if face_identifications:
            identified_people = []
            people_names = []

            for idx, ident in face_identifications.items():
                match_name = ident.get("match")
                if match_name and match_name not in ["Unknown", "Error"]:
                    identified_people.append({
                        "face_index": int(idx),
                        "name": str(match_name),
                        "confidence": float(ident.get("confidence", 0.0)),
                        "distance": float(ident.get("distance", 0.0))
                    })
                    people_names.append(str(match_name))

            if identified_people:
                payload["identified_people"] = identified_people
                payload["people_names"] = list(set(people_names))  # Unique names

        # Add captions
        if captions:
            payload["captions"] = captions

            # Create searchable caption text
            caption_texts = []
            for key in ["title", "caption", "travel_caption", "hashtags"]:
                if key in captions and captions[key]:
                    caption_texts.append(str(captions[key]))

            if caption_texts:
                payload["caption_text"] = " ".join(caption_texts)

            # Extract specific fields for easier querying
            if "scene_type" in captions:
                payload["scene_type"] = captions["scene_type"]
            if "mood" in captions:
                payload["mood"] = captions["mood"]

        # Prepare vector (face embedding or zero vector)
        if face_embedding is not None:
            if isinstance(face_embedding, np.ndarray):
                vector = face_embedding.flatten().tolist()
            else:
                vector = face_embedding

            # Ensure correct dimensions
            if len(vector) != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch: {len(vector)} vs {self.embedding_dim}")
                vector = [0.0] * self.embedding_dim
        else:
            # No face detected - use zero vector
            vector = [0.0] * self.embedding_dim

        # Store in Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            logger.info(f"✓ Stored photo: {point_id}")
            return point_id
        except Exception as e:
            logger.error(f"Error storing photo: {e}")
            raise

    def store_individual_faces(
        self,
        photo_id: str,
        photo_path: Union[str, Path],
        detected_faces: List[Dict],
        face_identifications: Optional[Dict] = None,
        exif_metadata: Optional[Dict] = None,
        save_face_images: bool = True,
        face_images_dir: Union[str, Path] = "extracted_faces"
    ) -> List[str]:
        """
        Store individual detected faces with their embeddings.

        Args:
            photo_id: ID of the parent photo in travel_photos collection
            photo_path: Path to the source photo
            detected_faces: List of detected faces with embeddings
            face_identifications: Dict of face index -> identification
            exif_metadata: EXIF data for context (datetime, location)
            save_face_images: Whether to save face images to disk
            face_images_dir: Directory to save face images

        Returns:
            List of face point IDs in Qdrant
        """
        photo_path = Path(photo_path)
        face_ids = []

        # Extract and save face images if requested
        saved_face_paths = {}
        if save_face_images:
            try:
                from travel_log.face_extractor import extract_and_save_faces
                saved_faces = extract_and_save_faces(
                    photo_path=photo_path,
                    detected_faces=detected_faces,
                    face_identifications=face_identifications,
                    exif_metadata=exif_metadata,
                    output_dir=face_images_dir,
                    organize_by_person=False,
                    photo_id=photo_id
                )
                # Map face index to saved path
                for face_info in saved_faces:
                    saved_face_paths[face_info['face_index']] = {
                        'saved_path': face_info['saved_path'],
                        'filename': face_info['filename']
                    }
                logger.info(f"✓ Saved {len(saved_faces)} face images to {face_images_dir}")
            except Exception as e:
                logger.warning(f"Could not save face images: {e}")

        for i, face in enumerate(detected_faces):
            # Generate unique face ID
            face_id = str(uuid.uuid4())

            # Get face embedding
            if "embedding" in face:
                embedding = face["embedding"]
                if isinstance(embedding, np.ndarray):
                    vector = embedding.flatten().tolist()
                else:
                    vector = embedding

                # Ensure correct dimensions
                if len(vector) != self.embedding_dim:
                    logger.warning(f"Face {i} embedding dimension mismatch: {len(vector)} vs {self.embedding_dim}")
                    continue
            else:
                logger.warning(f"Face {i} has no embedding, skipping")
                continue

            # Build face payload
            payload = {
                "photo_id": str(photo_id),  # Link to parent photo
                "face_index": int(i),
                "filename": photo_path.name,
                "filepath": str(photo_path.absolute()),
                "upload_timestamp": datetime.now().isoformat(),
            }

            # Add saved face image path if available
            if i in saved_face_paths:
                payload["face_image_path"] = saved_face_paths[i]['saved_path']
                payload["face_image_filename"] = saved_face_paths[i]['filename']

            # Add bounding box
            if "facial_area" in face:
                bbox = face["facial_area"]
                payload["bbox"] = {
                    "x": int(bbox.get("x", 0)),
                    "y": int(bbox.get("y", 0)),
                    "w": int(bbox.get("w", 0)),
                    "h": int(bbox.get("h", 0))
                }

            # Add detection confidence
            if "confidence" in face:
                payload["detection_confidence"] = float(face["confidence"])

            # Add face identification if available
            if face_identifications and i in face_identifications:
                ident = face_identifications[i]
                match_name = ident.get("match")

                if match_name and match_name not in ["Unknown", "Error"]:
                    payload["person_name"] = str(match_name)
                    payload["identification_confidence"] = float(ident.get("confidence", 0.0))
                    payload["identification_distance"] = float(ident.get("distance", 0.0))
                else:
                    payload["person_name"] = "Unknown"
            else:
                payload["person_name"] = "Unknown"

            # Add contextual metadata from photo
            if exif_metadata:
                # Add datetime
                if "datetime_str" in exif_metadata:
                    payload["datetime"] = str(exif_metadata["datetime_str"])

                # Add GPS location
                if "latitude" in exif_metadata and "longitude" in exif_metadata:
                    payload["latitude"] = float(exif_metadata["latitude"])
                    payload["longitude"] = float(exif_metadata["longitude"])
                    if "altitude" in exif_metadata:
                        payload["altitude"] = float(exif_metadata["altitude"])

            # Store face in Qdrant
            try:
                self.client.upsert(
                    collection_name=self.faces_collection_name,
                    points=[
                        PointStruct(
                            id=face_id,
                            vector=vector,
                            payload=payload
                        )
                    ]
                )
                face_ids.append(face_id)
                logger.info(f"✓ Stored face {i} ({payload.get('person_name', 'Unknown')}): {face_id}")
            except Exception as e:
                logger.error(f"Error storing face {i}: {e}")
                continue

        logger.info(f"✓ Stored {len(face_ids)} faces from photo {photo_id}")
        return face_ids

    def search_similar_faces(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Search for photos with similar faces.

        Args:
            query_embedding: Query face embedding
            limit: Number of results
            score_threshold: Minimum similarity score

        Returns:
            List of matching photos with scores
        """
        if isinstance(query_embedding, np.ndarray):
            query_vector = query_embedding.flatten().tolist()
        else:
            query_vector = query_embedding

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )

            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "filename": hit.payload.get("filename"),
                    "filepath": hit.payload.get("filepath"),
                    "people": hit.payload.get("people_names", []),
                    "metadata": hit.payload
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def search_by_person(
        self,
        person_name: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Find all photos containing a specific person.

        Args:
            person_name: Name to search for
            limit: Maximum results

        Returns:
            List of photos containing the person
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="people_names",
                            match=MatchValue(value=person_name)
                        )
                    ]
                ),
                limit=limit
            )

            return [
                {
                    "id": point.id,
                    "filename": point.payload.get("filename"),
                    "filepath": point.payload.get("filepath"),
                    "people": point.payload.get("people_names", []),
                    "datetime": point.payload.get("exif", {}).get("datetime"),
                    "metadata": point.payload
                }
                for point in results[0]
            ]
        except Exception as e:
            logger.error(f"Error searching by person: {e}")
            return []

    def search_by_location(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10.0,
        limit: int = 100
    ) -> List[Dict]:
        """
        Find photos near a location (within radius).

        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            limit: Maximum results

        Returns:
            List of nearby photos
        """
        # Simple distance-based search
        # For more advanced geo queries, we'd need geo-indexed fields
        try:
            all_photos = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit * 2  # Get more to filter
            )

            # Filter by distance
            from math import radians, sin, cos, sqrt, atan2

            def haversine_distance(lat1, lon1, lat2, lon2):
                """Calculate distance in km between two points."""
                R = 6371  # Earth radius in km
                dlat = radians(lat2 - lat1)
                dlon = radians(lon2 - lon1)
                a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                return R * c

            nearby = []
            for point in all_photos[0]:
                photo_lat = point.payload.get("latitude")
                photo_lon = point.payload.get("longitude")

                if photo_lat and photo_lon:
                    distance = haversine_distance(lat, lon, photo_lat, photo_lon)
                    if distance <= radius_km:
                        nearby.append({
                            "id": point.id,
                            "filename": point.payload.get("filename"),
                            "filepath": point.payload.get("filepath"),
                            "distance_km": round(distance, 2),
                            "location": {"lat": photo_lat, "lon": photo_lon},
                            "metadata": point.payload
                        })

            # Sort by distance and limit
            nearby.sort(key=lambda x: x["distance_km"])
            return nearby[:limit]

        except Exception as e:
            logger.error(f"Error searching by location: {e}")
            return []

    def get_photo(self, photo_id: str) -> Optional[Dict]:
        """Retrieve photo by ID."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[photo_id]
            )

            if result:
                return {
                    "id": result[0].id,
                    "metadata": result[0].payload
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving photo: {e}")
            return None

    def get_all_photos(self, limit: int = 1000) -> List[Dict]:
        """Get all photos."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit
            )

            return [
                {
                    "id": point.id,
                    "filename": point.payload.get("filename"),
                    "filepath": point.payload.get("filepath"),
                    "datetime": point.payload.get("exif", {}).get("datetime"),
                    "people": point.payload.get("people_names", []),
                    "num_faces": point.payload.get("num_faces", 0),
                    "metadata": point.payload
                }
                for point in results[0]
            ]
        except Exception as e:
            logger.error(f"Error getting all photos: {e}")
            return []

    def search_faces_by_person(
        self,
        person_name: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Find all face instances of a specific person.

        Args:
            person_name: Name to search for
            limit: Maximum results

        Returns:
            List of face instances
        """
        try:
            results = self.client.scroll(
                collection_name=self.faces_collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="person_name",
                            match=MatchValue(value=person_name)
                        )
                    ]
                ),
                limit=limit
            )

            return [
                {
                    "id": point.id,
                    "photo_id": point.payload.get("photo_id"),
                    "face_index": point.payload.get("face_index"),
                    "person_name": point.payload.get("person_name"),
                    "confidence": point.payload.get("identification_confidence"),
                    "filename": point.payload.get("filename"),
                    "datetime": point.payload.get("datetime"),
                    "bbox": point.payload.get("bbox"),
                    "metadata": point.payload
                }
                for point in results[0]
            ]
        except Exception as e:
            logger.error(f"Error searching faces by person: {e}")
            return []

    def search_faces_similar_to(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        score_threshold: float = 0.7,
        person_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for faces similar to a query embedding.

        Args:
            query_embedding: Query face embedding
            limit: Number of results
            score_threshold: Minimum similarity score
            person_name: Optional filter by person name

        Returns:
            List of similar faces with scores
        """
        if isinstance(query_embedding, np.ndarray):
            query_vector = query_embedding.flatten().tolist()
        else:
            query_vector = query_embedding

        try:
            # Build filter if person_name provided
            query_filter = None
            if person_name:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="person_name",
                            match=MatchValue(value=person_name)
                        )
                    ]
                )

            results = self.client.search(
                collection_name=self.faces_collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )

            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "photo_id": hit.payload.get("photo_id"),
                    "face_index": hit.payload.get("face_index"),
                    "person_name": hit.payload.get("person_name"),
                    "filename": hit.payload.get("filename"),
                    "bbox": hit.payload.get("bbox"),
                    "metadata": hit.payload
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            return []

    def get_all_faces(self, limit: int = 1000) -> List[Dict]:
        """Get all stored faces."""
        try:
            results = self.client.scroll(
                collection_name=self.faces_collection_name,
                limit=limit
            )

            return [
                {
                    "id": point.id,
                    "photo_id": point.payload.get("photo_id"),
                    "face_index": point.payload.get("face_index"),
                    "person_name": point.payload.get("person_name"),
                    "confidence": point.payload.get("identification_confidence"),
                    "filename": point.payload.get("filename"),
                    "datetime": point.payload.get("datetime"),
                    "bbox": point.payload.get("bbox")
                }
                for point in results[0]
            ]
        except Exception as e:
            logger.error(f"Error getting all faces: {e}")
            return []

    def get_statistics(self) -> Dict:
        """Get database statistics for both collections."""
        try:
            photos_info = self.client.get_collection(self.collection_name)
            faces_info = self.client.get_collection(self.faces_collection_name)

            return {
                "total_photos": photos_info.points_count,
                "total_faces": faces_info.points_count,
                "collection_name": self.collection_name,
                "faces_collection_name": self.faces_collection_name,
                "embedding_dimension": self.embedding_dim,
                "qdrant_url": self.client._client.rest_uri
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def delete_photo(self, photo_id: str) -> bool:
        """Delete a photo."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[photo_id]
            )
            logger.info(f"✓ Deleted photo: {photo_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting photo: {e}")
            return False


# Convenience function
def create_qdrant_store(
    url: str = "http://localhost:6333",
    collection: str = "travel_photos"
) -> TravelLogQdrantStore:
    """Create and return a Qdrant store instance."""
    return TravelLogQdrantStore(
        qdrant_url=url,
        collection_name=collection
    )
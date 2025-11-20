"""
Face Extractor - Save detected faces to disk with metadata

This module extracts and saves individual face images from photos,
storing them with comprehensive metadata.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime
import json
import shutil

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class FaceExtractor:
    """
    Extracts and saves individual face images to disk with metadata.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "extracted_faces",
        save_metadata: bool = True
    ):
        """
        Initialize FaceExtractor.

        Args:
            output_dir: Directory to save extracted faces
            save_metadata: Whether to save JSON metadata alongside images
        """
        self.output_dir = Path(output_dir)
        self.save_metadata = save_metadata

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FaceExtractor initialized: {self.output_dir}")

    def extract_and_save_faces(
        self,
        photo_path: Union[str, Path],
        detected_faces: List[Dict],
        face_identifications: Optional[Dict] = None,
        exif_metadata: Optional[Dict] = None,
        photo_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract face images from photo and save to disk.

        Args:
            photo_path: Path to source photo
            detected_faces: List of detected faces from DeepFace
            face_identifications: Dict of face_index -> identification
            exif_metadata: EXIF data from photo
            photo_id: Optional photo ID for organization

        Returns:
            List of dicts with saved face info
        """
        photo_path = Path(photo_path)
        saved_faces = []

        # Load original image
        try:
            original_image = cv2.imread(str(photo_path))
            if original_image is None:
                raise ValueError(f"Could not load image: {photo_path}")
        except Exception as e:
            logger.error(f"Error loading image {photo_path}: {e}")
            return []

        # Create subdirectory for this photo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_name = photo_path.stem
        photo_dir = self.output_dir / f"{photo_name}_{timestamp}"
        photo_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting {len(detected_faces)} faces from {photo_path.name}")

        for i, face in enumerate(detected_faces):
            try:
                # Get person name
                person_name = "Unknown"
                identification_data = {}
                if face_identifications and i in face_identifications:
                    ident = face_identifications[i]
                    match_name = ident.get("match")
                    if match_name and match_name not in ["Unknown", "Error"]:
                        person_name = match_name
                    identification_data = {
                        "person_name": person_name,
                        "confidence": float(ident.get("confidence", 0.0)),
                        "distance": float(ident.get("distance", 0.0))
                    }

                # Extract face region using bounding box
                if "facial_area" in face:
                    bbox = face["facial_area"]
                    x = int(bbox.get("x", 0))
                    y = int(bbox.get("y", 0))
                    w = int(bbox.get("w", 0))
                    h = int(bbox.get("h", 0))

                    # Crop face from original image
                    face_img = original_image[y:y+h, x:x+w]

                    # Generate filename
                    face_filename = f"face_{i:02d}_{person_name}.jpg"
                    face_path = photo_dir / face_filename

                    # Save face image
                    cv2.imwrite(str(face_path), face_img)

                    # Prepare metadata
                    face_metadata = {
                        "face_index": i,
                        "source_photo": photo_path.name,
                        "source_photo_path": str(photo_path.absolute()),
                        "photo_id": photo_id,
                        "saved_path": str(face_path.absolute()),
                        "filename": face_filename,
                        "extraction_timestamp": datetime.now().isoformat(),

                        # Bounding box
                        "bbox": {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h
                        },

                        # Detection info
                        "detection_confidence": float(face.get("confidence", 1.0)),

                        # Identification
                        **identification_data,

                        # Image dimensions
                        "face_width": w,
                        "face_height": h,
                    }

                    # Add EXIF context
                    if exif_metadata:
                        face_metadata["exif_context"] = {
                            "datetime": exif_metadata.get("datetime_str"),
                            "latitude": exif_metadata.get("latitude"),
                            "longitude": exif_metadata.get("longitude"),
                            "altitude": exif_metadata.get("altitude"),
                            "camera": exif_metadata.get("camera", {})
                        }

                    # Add embedding if available
                    if "embedding" in face:
                        embedding = face["embedding"]
                        if isinstance(embedding, np.ndarray):
                            face_metadata["embedding_dimension"] = len(embedding)
                            face_metadata["has_embedding"] = True
                            # Save embedding separately as numpy file
                            embedding_path = face_path.with_suffix(".npy")
                            np.save(embedding_path, embedding)
                            face_metadata["embedding_path"] = str(embedding_path)
                        else:
                            face_metadata["has_embedding"] = False

                    # Save metadata JSON
                    if self.save_metadata:
                        metadata_path = face_path.with_suffix(".json")
                        with open(metadata_path, 'w') as f:
                            json.dump(face_metadata, f, indent=2)
                        face_metadata["metadata_path"] = str(metadata_path)

                    saved_faces.append(face_metadata)
                    logger.info(f"✓ Saved face {i}: {person_name} → {face_path.name}")

            except Exception as e:
                logger.error(f"Error extracting face {i}: {e}")
                continue

        # Save photo-level metadata
        if self.save_metadata and saved_faces:
            photo_metadata = {
                "source_photo": photo_path.name,
                "source_photo_path": str(photo_path.absolute()),
                "photo_id": photo_id,
                "extraction_timestamp": datetime.now().isoformat(),
                "total_faces": len(saved_faces),
                "faces": saved_faces,
                "exif_metadata": exif_metadata
            }

            photo_metadata_path = photo_dir / "photo_metadata.json"
            with open(photo_metadata_path, 'w') as f:
                json.dump(photo_metadata, f, indent=2)

        logger.info(f"✓ Extracted and saved {len(saved_faces)} faces to {photo_dir}")
        return saved_faces

    def extract_and_save_by_person(
        self,
        photo_path: Union[str, Path],
        detected_faces: List[Dict],
        face_identifications: Optional[Dict] = None,
        exif_metadata: Optional[Dict] = None,
        photo_id: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Extract faces and organize by person name.

        Args:
            photo_path: Path to source photo
            detected_faces: List of detected faces
            face_identifications: Face identifications
            exif_metadata: EXIF metadata
            photo_id: Optional photo ID

        Returns:
            Dict mapping person_name -> list of saved face info
        """
        photo_path = Path(photo_path)
        faces_by_person = {}

        # Load original image
        try:
            original_image = cv2.imread(str(photo_path))
            if original_image is None:
                raise ValueError(f"Could not load image: {photo_path}")
        except Exception as e:
            logger.error(f"Error loading image {photo_path}: {e}")
            return {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, face in enumerate(detected_faces):
            try:
                # Get person name
                person_name = "Unknown"
                identification_data = {}
                if face_identifications and i in face_identifications:
                    ident = face_identifications[i]
                    match_name = ident.get("match")
                    if match_name and match_name not in ["Unknown", "Error"]:
                        person_name = match_name
                    identification_data = {
                        "confidence": float(ident.get("confidence", 0.0)),
                        "distance": float(ident.get("distance", 0.0))
                    }

                # Create person directory
                person_dir = self.output_dir / person_name
                person_dir.mkdir(parents=True, exist_ok=True)

                # Extract and save face
                if "facial_area" in face:
                    bbox = face["facial_area"]
                    x = int(bbox.get("x", 0))
                    y = int(bbox.get("y", 0))
                    w = int(bbox.get("w", 0))
                    h = int(bbox.get("h", 0))

                    face_img = original_image[y:y+h, x:x+w]

                    # Generate unique filename
                    photo_name = photo_path.stem
                    face_filename = f"{photo_name}_{timestamp}_face{i:02d}.jpg"
                    face_path = person_dir / face_filename

                    # Save face image
                    cv2.imwrite(str(face_path), face_img)

                    # Metadata
                    face_metadata = {
                        "face_index": i,
                        "person_name": person_name,
                        "source_photo": photo_path.name,
                        "saved_path": str(face_path.absolute()),
                        "filename": face_filename,
                        "extraction_timestamp": datetime.now().isoformat(),
                        "bbox": {"x": x, "y": y, "width": w, "height": h},
                        "detection_confidence": float(face.get("confidence", 1.0)),
                        **identification_data
                    }

                    # Save embedding
                    if "embedding" in face and isinstance(face["embedding"], np.ndarray):
                        embedding_path = face_path.with_suffix(".npy")
                        np.save(embedding_path, face["embedding"])
                        face_metadata["embedding_path"] = str(embedding_path)

                    # Save metadata
                    if self.save_metadata:
                        metadata_path = face_path.with_suffix(".json")
                        with open(metadata_path, 'w') as f:
                            json.dump(face_metadata, f, indent=2)

                    # Add to person's list
                    if person_name not in faces_by_person:
                        faces_by_person[person_name] = []
                    faces_by_person[person_name].append(face_metadata)

                    logger.info(f"✓ Saved {person_name} face to {person_dir}/{face_filename}")

            except Exception as e:
                logger.error(f"Error extracting face {i}: {e}")
                continue

        logger.info(f"✓ Extracted faces for {len(faces_by_person)} people")
        return faces_by_person

    def get_extraction_stats(self) -> Dict:
        """Get statistics about extracted faces."""
        if not self.output_dir.exists():
            return {"total_faces": 0, "total_people": 0}

        # Count by person (if organized by person)
        person_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]

        stats = {
            "output_directory": str(self.output_dir.absolute()),
            "total_directories": len(person_dirs),
            "faces_by_person": {}
        }

        total_faces = 0
        for person_dir in person_dirs:
            face_count = len(list(person_dir.glob("*.jpg")))
            if face_count > 0:
                stats["faces_by_person"][person_dir.name] = face_count
                total_faces += face_count

        stats["total_faces"] = total_faces
        stats["total_people"] = len(stats["faces_by_person"])

        return stats


# Convenience function
def extract_and_save_faces(
    photo_path: Union[str, Path],
    detected_faces: List[Dict],
    face_identifications: Optional[Dict] = None,
    exif_metadata: Optional[Dict] = None,
    output_dir: Union[str, Path] = "extracted_faces",
    organize_by_person: bool = False,
    photo_id: Optional[str] = None
) -> Union[List[Dict], Dict[str, List[Dict]]]:
    """
    Convenience function to extract and save faces.

    Args:
        photo_path: Path to source photo
        detected_faces: List of detected faces
        face_identifications: Face identifications
        exif_metadata: EXIF metadata
        output_dir: Output directory
        organize_by_person: If True, organize by person name
        photo_id: Optional photo ID

    Returns:
        List of saved face info or dict by person
    """
    extractor = FaceExtractor(output_dir=output_dir)

    if organize_by_person:
        return extractor.extract_and_save_by_person(
            photo_path, detected_faces, face_identifications,
            exif_metadata, photo_id
        )
    else:
        return extractor.extract_and_save_faces(
            photo_path, detected_faces, face_identifications,
            exif_metadata, photo_id
        )

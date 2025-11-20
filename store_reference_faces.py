#!/usr/bin/env python3
"""
Store reference face embeddings in Qdrant.

This script processes all reference faces in face_database/ and stores
their embeddings in a Qdrant collection for efficient similarity search.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from deepface import DeepFace
import uuid
from datetime import datetime
import json

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def store_reference_faces(
    face_db_path: str = "face_database",
    qdrant_url: str = "http://sapphire:6333",
    model_name: str = "VGG-Face",
    collection_name: str = "reference_faces"
):
    """
    Store reference face embeddings in Qdrant.

    Args:
        face_db_path: Path to face database directory
        qdrant_url: Qdrant server URL
        model_name: Face recognition model to use
        collection_name: Qdrant collection name for reference faces
    """

    print("=" * 70)
    print("Store Reference Faces in Qdrant")
    print("=" * 70)
    print(f"\nFace Database: {face_db_path}")
    print(f"Qdrant URL: {qdrant_url}")
    print(f"Model: {model_name}")
    print(f"Collection: {collection_name}\n")

    # Connect to Qdrant
    print("Connecting to Qdrant...")
    client = QdrantClient(url=qdrant_url)

    # Get embedding dimension
    print(f"Getting embedding dimension for {model_name}...")
    test_embedding = DeepFace.represent(
        img_path=str(Path(__file__).parent / "face_database" / "sankar" / "sankar.jpg"),
        model_name=model_name,
        enforce_detection=False
    )
    embedding_dim = len(test_embedding[0]['embedding'])
    print(f"✓ Embedding dimension: {embedding_dim}D\n")

    # Create or recreate collection
    try:
        client.get_collection(collection_name)
        print(f"⚠️  Collection '{collection_name}' already exists. Recreating...")
        client.delete_collection(collection_name)
    except:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )
    print(f"✓ Created collection '{collection_name}'\n")

    # Process each person in face database
    face_db = Path(face_db_path)

    if not face_db.exists():
        print(f"❌ Face database not found: {face_db}")
        return

    person_dirs = [d for d in face_db.iterdir() if d.is_dir()]

    if not person_dirs:
        print(f"❌ No person directories found in {face_db}")
        return

    print(f"Found {len(person_dirs)} people in database\n")
    print("Processing reference faces...")
    print("-" * 70)

    points = []
    total_faces = 0

    for person_dir in sorted(person_dirs):
        person_name = person_dir.name

        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(person_dir.glob(ext))

        if not image_files:
            print(f"⚠️  {person_name}: No images found")
            continue

        print(f"\n📁 {person_name}: {len(image_files)} image(s)")

        for img_path in image_files:
            try:
                # Generate embedding
                embeddings = DeepFace.represent(
                    img_path=str(img_path),
                    model_name=model_name,
                    enforce_detection=False
                )

                if not embeddings:
                    print(f"   ✗ {img_path.name}: No face detected")
                    continue

                # Use first face if multiple detected
                embedding = embeddings[0]['embedding']

                # Create point
                point_id = str(uuid.uuid4())

                payload = {
                    "person_name": person_name,
                    "image_filename": img_path.name,
                    "image_path": str(img_path.absolute()),
                    "is_reference": True,
                    "model_name": model_name,
                    "embedding_dimension": len(embedding),
                    "stored_timestamp": datetime.now().isoformat(),
                    "face_type": "reference",
                }

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )

                points.append(point)
                total_faces += 1

                print(f"   ✓ {img_path.name}: Embedded ({len(embedding)}D)")

            except Exception as e:
                print(f"   ✗ {img_path.name}: {e}")

    # Upload all points to Qdrant
    if points:
        print(f"\n{'='*70}")
        print(f"Uploading {len(points)} reference face embeddings to Qdrant...")

        client.upsert(
            collection_name=collection_name,
            points=points
        )

        print(f"✓ Successfully stored {len(points)} reference faces!")

        # Get collection info
        collection_info = client.get_collection(collection_name)
        print(f"\nCollection '{collection_name}':")
        print(f"  • Points: {collection_info.points_count}")
        print(f"  • Vector dimension: {collection_info.config.params.vectors.size}")
        print(f"  • Distance metric: {collection_info.config.params.vectors.distance}")
    else:
        print("\n❌ No reference faces were processed")

    print("\n" + "=" * 70)

    return len(points)


def verify_reference_faces(
    qdrant_url: str = "http://sapphire:6333",
    collection_name: str = "reference_faces"
):
    """Verify stored reference faces."""

    print("\n" + "=" * 70)
    print("Verify Reference Faces in Qdrant")
    print("=" * 70)

    client = QdrantClient(url=qdrant_url)

    try:
        collection_info = client.get_collection(collection_name)
        print(f"\n✓ Collection '{collection_name}' exists")
        print(f"  Points: {collection_info.points_count}")

        # Get sample points
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=100
        )

        points = scroll_result[0]

        if points:
            print(f"\nReference faces by person:")

            # Group by person
            people = {}
            for point in points:
                person = point.payload.get('person_name', 'Unknown')
                if person not in people:
                    people[person] = []
                people[person].append(point.payload.get('image_filename'))

            for person, images in sorted(people.items()):
                print(f"  • {person}: {len(images)} image(s)")
                for img in images[:3]:  # Show first 3
                    print(f"    - {img}")
                if len(images) > 3:
                    print(f"    ... and {len(images) - 3} more")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Store reference faces in Qdrant")
    parser.add_argument(
        "--face-db",
        default="face_database",
        help="Path to face database directory"
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://sapphire:6333",
        help="Qdrant server URL"
    )
    parser.add_argument(
        "--model",
        default="VGG-Face",
        choices=["VGG-Face", "Facenet512", "Facenet", "ArcFace"],
        help="Face recognition model"
    )
    parser.add_argument(
        "--collection",
        default="reference_faces",
        help="Qdrant collection name"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing reference faces"
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_reference_faces(args.qdrant_url, args.collection)
    else:
        count = store_reference_faces(
            args.face_db,
            args.qdrant_url,
            args.model,
            args.collection
        )

        if count > 0:
            verify_reference_faces(args.qdrant_url, args.collection)

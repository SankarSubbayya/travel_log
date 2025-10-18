#!/usr/bin/env python3
"""
Example: Face Embeddings and Similarity Search

This example demonstrates how to:
1. Generate face embeddings (signature vectors)
2. Compute similarity between faces
3. Find similar faces
4. Save and load embeddings
"""

from pathlib import Path
from travel_log import FaceEmbeddings
import numpy as np

def main():
    # Initialize the embeddings generator
    embedder = FaceEmbeddings(
        model_name='Facenet512',  # 512-dimensional embeddings
        detector_backend='mtcnn'
    )
    
    # Step 1: Generate embedding for a single face
    print("Step 1: Generating single embedding...")
    
    face_image = "extracted_faces/face_001.jpg"
    embedding = embedder.generate_embedding(face_image)
    
    if embedding:
        print(f"Generated embedding:")
        print(f"  Dimension: {embedding['dimension']}")
        print(f"  Model: {embedding['model']}")
        print(f"  Vector shape: {embedding['embedding'].shape}")
        print(f"  First 5 values: {embedding['embedding'][:5]}")
    
    # Step 2: Generate embeddings for multiple faces
    print("\nStep 2: Batch embedding generation...")
    
    face_images = [
        "extracted_faces/face_001.jpg",
        "extracted_faces/face_002.jpg",
        "extracted_faces/face_003.jpg",
        "extracted_faces/face_004.jpg"
    ]
    
    embeddings = embedder.generate_embeddings_batch(face_images)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Step 3: Compute similarity between faces
    print("\nStep 3: Computing face similarity...")
    
    if len(embeddings) >= 2:
        emb1 = embeddings[0]['embedding']
        emb2 = embeddings[1]['embedding']
        
        # Cosine similarity (higher = more similar)
        cosine_sim = embedder.compute_similarity(emb1, emb2, metric='cosine')
        print(f"Cosine similarity: {cosine_sim:.4f}")
        
        # Euclidean distance (lower = more similar)
        euclidean_dist = embedder.compute_similarity(emb1, emb2, metric='euclidean')
        print(f"Euclidean distance: {euclidean_dist:.4f}")
    
    # Step 4: Find most similar faces
    print("\nStep 4: Finding similar faces...")
    
    query_embedding = embeddings[0]['embedding']
    similar_faces = embedder.find_most_similar(
        query_embedding=query_embedding,
        candidate_embeddings=embeddings[1:],  # Compare with rest
        top_k=3,
        metric='cosine'
    )
    
    print(f"Top 3 similar faces to {embeddings[0]['image_path']}:")
    for i, match in enumerate(similar_faces, 1):
        print(f"  {i}. {Path(match['image_path']).name} "
              f"(similarity: {match['similarity']:.4f})")
    
    # Step 5: Save embeddings to disk
    print("\nStep 5: Saving embeddings...")
    
    # Save as pickle
    embedder.save_embeddings(
        embeddings,
        output_path="embeddings/face_embeddings.pkl",
        format='pickle'
    )
    print("Saved embeddings to: embeddings/face_embeddings.pkl")
    
    # Save as numpy compressed format
    embedder.save_embeddings(
        embeddings,
        output_path="embeddings/face_embeddings.npz",
        format='npz'
    )
    print("Saved embeddings to: embeddings/face_embeddings.npz")
    
    # Step 6: Load embeddings from disk
    print("\nStep 6: Loading embeddings...")
    
    loaded_embeddings = embedder.load_embeddings(
        "embeddings/face_embeddings.pkl",
        format='pickle'
    )
    print(f"Loaded {len(loaded_embeddings)} embeddings from disk")
    
    # Step 7: Create an embedding database
    print("\nStep 7: Creating embedding database...")
    
    db_embeddings = embedder.create_embedding_database(
        face_images_dir="extracted_faces",
        output_path="embeddings/database.pkl",
        pattern="*.jpg"
    )
    print(f"Created database with {len(db_embeddings)} face embeddings")
    
    # Demonstrate clustering potential
    print("\nBonus: Embedding statistics...")
    all_embeddings = np.array([e['embedding'] for e in embeddings])
    print(f"Embedding matrix shape: {all_embeddings.shape}")
    print(f"Mean value: {all_embeddings.mean():.4f}")
    print(f"Std deviation: {all_embeddings.std():.4f}")
    
    # Compute pairwise similarities
    print("\nPairwise similarity matrix:")
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = embedder.compute_similarity(
                embeddings[i]['embedding'],
                embeddings[j]['embedding'],
                metric='cosine'
            )
    
    print(sim_matrix)


if __name__ == "__main__":
    main()


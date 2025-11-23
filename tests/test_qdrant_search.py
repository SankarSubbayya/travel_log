#!/usr/bin/env python3
"""
Test Qdrant search functionality to diagnose the search error.
"""

from qdrant_client import QdrantClient
import numpy as np

def test_qdrant_search():
    """Test if Qdrant search method works."""

    print("🔍 Testing Qdrant search functionality...\n")

    # Connect to Qdrant
    qdrant_url = "http://sapphire:6333"
    print(f"Connecting to: {qdrant_url}")

    try:
        client = QdrantClient(url=qdrant_url)
        print("✅ Connected to Qdrant\n")

        # Check collections
        collections = client.get_collections()
        print(f"Available collections: {[c.name for c in collections.collections]}\n")

        # Check if reference_faces exists
        if not client.collection_exists("reference_faces"):
            print("❌ reference_faces collection does not exist!")
            return False

        # Get collection info
        collection_info = client.get_collection("reference_faces")
        print(f"reference_faces collection:")
        print(f"  - Points count: {collection_info.points_count}")
        print(f"  - Vector size: {collection_info.config.params.vectors.size}")
        print()

        # Create a random embedding for testing (VGG-Face = 4096D)
        test_embedding = np.random.rand(4096).tolist()

        print("Testing query_points method (modern API)...")
        query_response = client.query_points(
            collection_name="reference_faces",
            query=test_embedding,
            limit=3
        )

        print(f"✅ Query successful! Found {len(query_response.points)} results\n")

        for i, result in enumerate(query_response.points):
            print(f"Result {i+1}:")
            print(f"  - Score: {result.score:.4f}")
            print(f"  - Person: {result.payload.get('person_name', 'Unknown')}")
            print()

        return True

    except AttributeError as e:
        print(f"❌ AttributeError: {e}")
        print("\nThis suggests the QdrantClient object doesn't have the search method.")
        print("Checking available methods...")
        print(f"Available methods: {[m for m in dir(client) if not m.startswith('_')]}")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qdrant_search()

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")

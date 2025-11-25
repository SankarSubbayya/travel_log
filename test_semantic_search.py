#!/usr/bin/env python3
"""
Test semantic search functionality with example queries.

This script demonstrates how to use the semantic search feature
to answer natural language queries about travel photos.
"""

from travel_log.qdrant_store import create_qdrant_store
from travel_log import create_semantic_search

def main():
    print("=== Travel Log Semantic Search Test ===\n")

    # Connect to Qdrant
    print("1. Connecting to Qdrant...")
    try:
        qdrant_store = create_qdrant_store(url="http://sapphire:6333")
        stats = qdrant_store.get_statistics()
        print(f"   ✓ Connected! Found {stats['total_photos']} photos in database\n")
    except Exception as e:
        print(f"   ✗ Error connecting to Qdrant: {e}")
        print("   Make sure Qdrant is running on sapphire:6333")
        return

    # Create semantic search
    print("2. Initializing semantic search...")
    search = create_semantic_search(qdrant_store)
    print("   ✓ Ready!\n")

    # Test queries
    test_queries = [
        "When, where and with whom did I see the turtles on the beach?",
        "Show me photos from my beach vacation",
        "Find photos with Sarah",
        "Pictures from last summer",
        "Mountain hiking photos"
    ]

    print("3. Testing example queries:\n")
    print("=" * 80)

    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("-" * 80)

        try:
            # Parse the query
            parsed = search.parse_query(query)
            print(f"\n   Parsed query:")
            if parsed['people']:
                print(f"   - People: {', '.join(parsed['people'])}")
            if parsed['locations']:
                print(f"   - Locations: {', '.join(parsed['locations'])}")
            if parsed['content_keywords']:
                print(f"   - Content: {', '.join(parsed['content_keywords'])}")
            if parsed['time_filters']:
                print(f"   - Time filters: {parsed['time_filters']}")

            # Search
            results = search.search(query, limit=5)

            if results:
                print(f"\n   ✓ Found {len(results)} matching photos:")
                for i, result in enumerate(results, 1):
                    score = result.get('relevance_score', 0)
                    filename = result.get('filename', 'Unknown')
                    people = result.get('people', [])
                    datetime = result.get('datetime', 'Unknown')

                    print(f"\n   {i}. {filename} (relevance: {score:.1%})")
                    print(f"      Date: {datetime}")
                    if people:
                        print(f"      People: {', '.join(people)}")

                    # Show match reasons
                    match_reasons = result.get('match_reasons', [])
                    if match_reasons:
                        print(f"      Why it matched:")
                        for reason in match_reasons:
                            print(f"        - {reason}")
            else:
                print("\n   ℹ No matching photos found")

        except Exception as e:
            print(f"\n   ✗ Error: {e}")

        print("\n" + "=" * 80)

    print("\n✓ Test complete!")
    print("\nTo use semantic search in the app:")
    print("1. Run: ./run_app.sh")
    print("2. Go to the 'Search' tab")
    print("3. Enter your query and click 'Search'")

if __name__ == "__main__":
    main()

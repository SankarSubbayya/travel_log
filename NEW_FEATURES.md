# New Features: Journey Mapping, Location Context, and Semantic Search

## Overview

Three powerful new features have been added to the Travel Log application to enhance your photo organization and search capabilities:

1. **Journey Mapping** - Visualize your travel paths on interactive maps
2. **Location Contextualization** - Enrich photos with Wikipedia descriptions of locations
3. **Semantic Search** - Natural language search across all your photos

## 1. Journey Mapping 🗺️

### What It Does

Creates visual journey maps by connecting your photos chronologically with an embedded Google Maps view.

### Features

- **Embedded Google Maps**: Full interactive Google Maps displayed directly in Streamlit
- **Automatic Route Generation**: Creates driving directions through all photo locations
- **Daily Breakdowns**: Group photos by day with separate map links
- **Time Filtering**: Filter journeys by date range
- **People Integration**: See who was at each location
- **GPS Coordinate Editor**: Update photo locations directly in the app

### How to Use

1. Open the Streamlit app: `./run_app.sh`
2. Go to the **🗺️ Journey Map** tab
3. Connect to Qdrant (if not already connected)
4. (Optional) Enable date filtering to focus on specific trips
5. Click **"🗺️ Generate Journey Map"**
6. View results:
   - **Embedded Google Maps** showing your full journey route
   - Daily breakdowns with clickable links
   - Statistics (locations, days, dates)

### GPS Coordinate Editor

Update photo locations easily:
1. Scroll to "📍 Edit GPS Coordinates" section
2. Click "✏️ Update Photo Locations" to expand
3. Select a photo from the dropdown
4. View current coordinates with Google Maps link
5. Enter new latitude/longitude (with quick reference for India & US cities)
6. Click "💾 Update GPS Coordinates"
7. Regenerate journey map to see changes

### Code Location

- Module: `src/travel_log/journey_mapper.py`
- UI: `app.py` (Journey Map tab)

### Example Usage

```python
from travel_log import create_journey_mapper
from travel_log.qdrant_store import create_qdrant_store

# Connect to database
store = create_qdrant_store(url="http://sapphire:6333")

# Create journey mapper
mapper = create_journey_mapper(store)

# Generate journey summary
summary = mapper.generate_journey_summary()

print(f"Journey: {summary['total_points']} locations across {summary['total_days']} days")
print(f"Google Maps: {summary['map_url']}")

# Create interactive HTML map
html_map = mapper.create_html_map(summary['points'])
```

## 2. Location Contextualization 🌍

### What It Does

Automatically enriches your photos with context about their locations:
- Reverse geocodes GPS coordinates to place names
- Fetches Wikipedia articles about locations
- Extracts summaries and descriptions
- Stores context in Qdrant for search

### Features

- **Reverse Geocoding**: GPS → City, State, Country names
- **Wikipedia Integration**: Automatic article retrieval
- **Smart Caching**: Avoids redundant API calls
- **Bulk Processing**: Contextualize all photos at once
- **No API Key Required**: Uses free OpenStreetMap Nominatim and Wikipedia APIs

### How to Use

#### Via Streamlit App

1. Go to the **🔎 Search** tab
2. Scroll to "🌍 Enrich with Wikipedia Context"
3. Click **"🌍 Add Wikipedia Context to All Photos"**
4. Wait for processing (shows progress)
5. Review statistics (contextualized, skipped, errors)

#### Via Python

```python
from travel_log import create_location_contextualizer
from travel_log.qdrant_store import create_qdrant_store

# Connect to database
store = create_qdrant_store(url="http://sapphire:6333")

# Create contextualizer
contextualizer = create_location_contextualizer(store)

# Reverse geocode a location
place_info = contextualizer.reverse_geocode(13.0827, 80.2707)
print(f"Location: {place_info['city']}, {place_info['country']}")

# Get Wikipedia article
wiki = contextualizer.search_wikipedia("Chennai")
print(f"Article: {wiki['title']}")
print(f"Summary: {wiki['summary']}")

# Contextualize all photos
stats = contextualizer.bulk_contextualize_photos(limit=100)
print(f"Contextualized {stats['contextualized']} photos")
```

### Code Location

- Module: `src/travel_log/location_context.py`
- UI: `app.py` (Search tab)

### Data Added to Photos

Each photo gets a `location_context` field with:
```json
{
  "place_name": "Chennai, Tamil Nadu, India",
  "city": "Chennai",
  "country": "India",
  "wikipedia_title": "Chennai",
  "wikipedia_summary": "Chennai is the capital of Tamil Nadu...",
  "wikipedia_url": "https://en.wikipedia.org/wiki/Chennai"
}
```

## 3. Semantic Search 🔎

### What It Does

Enables natural language search across all your travel photos, understanding queries like:
- "When, where and with whom did I see the turtles on the beach?"
- "Show me photos from my beach vacation"
- "Find photos with Sarah in Paris"

### Features

- **Natural Language Understanding**: Ask questions naturally
- **Multi-modal Search**: Searches across:
  - People (face identifications)
  - Locations (GPS + Wikipedia context)
  - Dates/times (EXIF timestamps)
  - Content (AI captions)
  - Metadata (all photo data)
- **Relevance Scoring**: Ranks results by match quality
- **Match Explanations**: Shows why each photo matched
- **Query Parsing**: Automatically extracts entities (people, places, dates, keywords)

### How to Use

#### Via Streamlit App

1. Go to the **🔎 Search** tab
2. Type your natural language query
   - Or select an example from the dropdown
3. Set max results (default: 10)
4. Click **"🔍 Search"**
5. Browse results with:
   - Relevance scores
   - Match reasons (why it matched)
   - Photo previews
   - Metadata (date, people, location)

#### Via Python

```python
from travel_log import create_semantic_search
from travel_log.qdrant_store import create_qdrant_store

# Connect and create search
store = create_qdrant_store(url="http://sapphire:6333")
search = create_semantic_search(store)

# Natural language search
results = search.search(
    query="When, where and with whom did I see the turtles on the beach?",
    limit=10
)

# Process results
for result in results:
    print(f"Photo: {result['filename']}")
    print(f"Relevance: {result['relevance_score']:.1%}")
    print(f"Date: {result['datetime']}")
    print(f"People: {', '.join(result.get('people', []))}")
    print(f"Match reasons: {result['match_reasons']}")
    print()
```

### Code Location

- Module: `src/travel_log/semantic_search.py`
- UI: `app.py` (Search tab)

### Supported Query Types

| Query Type | Examples |
|------------|----------|
| **People** | "Photos with Sarah", "Pictures of John and Mary" |
| **Location** | "Photos from Paris", "Beach pictures", "Near Eiffel Tower" |
| **Time** | "Photos from last summer", "January 2024", "Last week" |
| **Content** | "Photos with turtles", "Mountain hiking", "Sunset beach" |
| **Combined** | "Sarah in Paris last summer", "Beach photos with John" |

### Query Parsing

The system automatically extracts:
- **People names**: Capitalized words, "with [Name]" patterns
- **Locations**: After "in", "at", "near", "from"
- **Dates**: Relative ("last week"), specific ("2024-01-15"), month/year
- **Content keywords**: After "see", "show", "find"

## Installation & Dependencies

### New Dependencies Added

```bash
# Already installed automatically
uv add sentence-transformers  # For text embeddings (optional)
```

### Requirements

- Python ≥ 3.12
- Existing dependencies (DeepFace, Qdrant, Streamlit, etc.)
- Internet connection (for Wikipedia/geocoding APIs)
- Qdrant running on `sapphire:6333` (or configure URL)

## Testing

### Test Semantic Search

```bash
# Run test script
python test_semantic_search.py
```

This will:
1. Connect to Qdrant
2. Test example queries
3. Show parsed queries and results
4. Display match reasons

### Test Journey Mapping

```python
from travel_log import create_journey_mapper
from travel_log.qdrant_store import create_qdrant_store

store = create_qdrant_store(url="http://sapphire:6333")
mapper = create_journey_mapper(store)

# Get journey summary
summary = mapper.generate_journey_summary()
print(f"Total locations: {summary['total_points']}")
print(f"Map URL: {summary['map_url']}")

# Save HTML map
mapper.create_html_map(summary['points'], output_path="journey_map.html")
print("Saved to journey_map.html")
```

### Test Location Context

```python
from travel_log import create_location_contextualizer
from travel_log.qdrant_store import create_qdrant_store

store = create_qdrant_store(url="http://sapphire:6333")
contextualizer = create_location_contextualizer(store)

# Test reverse geocoding
place = contextualizer.reverse_geocode(37.7749, -122.4194)
print(f"Location: {place['display_name']}")

# Test Wikipedia search
wiki = contextualizer.search_wikipedia("San Francisco")
print(f"Wikipedia: {wiki['title']}")
print(f"Summary: {wiki['summary'][:200]}...")
```

## Architecture

### Module Structure

```
src/travel_log/
├── journey_mapper.py         # Journey mapping functionality
├── location_context.py        # Wikipedia & geocoding
├── semantic_search.py         # Natural language search
└── qdrant_store.py           # (Updated) Qdrant integration
```

### Data Flow

```
Photo Upload
    ↓
Face Detection → EXIF Extraction → GPS Coordinates
    ↓                                    ↓
Qdrant Storage  ←────────────  Location Context
    ↓                          (Wikipedia, geocoding)
Journey Mapping + Semantic Search
```

### Key Classes

1. **JourneyMapper**
   - `get_journey_points()`: Extract GPS points chronologically
   - `create_google_maps_url()`: Generate Maps routes
   - `create_html_map()`: Build interactive Leaflet maps
   - `generate_journey_summary()`: Complete journey data

2. **LocationContextualizer**
   - `reverse_geocode()`: GPS → place names
   - `search_wikipedia()`: Fetch Wikipedia articles
   - `get_location_context()`: Complete location data
   - `bulk_contextualize_photos()`: Process all photos

3. **SemanticPhotoSearch**
   - `parse_query()`: Extract entities from text
   - `search()`: Natural language photo search
   - `_calculate_relevance_score()`: Rank results
   - `_get_match_reasons()`: Explain matches

## UI Integration

### New Tabs in app.py

1. **🗺️ Journey Map** (Tab 6)
   - Journey settings (date filtering)
   - Statistics (locations, days)
   - Google Maps links (full + daily)
   - Interactive Leaflet map
   - Daily breakdowns

2. **🔎 Search** (Tab 7)
   - Search input with examples
   - Result limit control
   - Result display with:
     - Photo previews
     - Relevance scores
     - Match reasons
     - Metadata
   - Wikipedia context enrichment button

## Workflow Example

### Complete Photo Processing Workflow

1. **Upload Photo** (Face Detection tab)
   - Detects faces
   - Extracts EXIF (GPS, datetime)
   - Saves to permanent storage

2. **Identify People** (Face Identification tab)
   - Matches faces to known people
   - Stores identifications

3. **Generate Captions** (Travel Log tab)
   - Creates AI descriptions
   - Adds context

4. **Save to Qdrant** (Qdrant Storage tab)
   - Stores all data
   - Gets photo ID

5. **Add Wikipedia Context** (Search tab)
   - Enriches with location info
   - Adds place descriptions

6. **View Journey** (Journey Map tab)
   - See travel paths
   - Explore routes

7. **Search Photos** (Search tab)
   - Natural language queries
   - Find specific memories

## Performance Notes

- **Journey Mapping**: Fast (< 1 second for 100 photos)
- **Location Context**: ~2-3 seconds per unique location (cached)
- **Semantic Search**: Very fast (< 1 second for 100 photos)
- **Wikipedia API**: Free, no rate limits for reasonable use
- **Geocoding API**: Free OpenStreetMap Nominatim

## Limitations & Future Improvements

### Current Limitations

1. **Google Maps Route Limit**: Max 25 waypoints per route
   - Solution: Auto-samples points if > 25 locations

2. **Wikipedia Language**: Currently English only
   - Solution: Can be extended to other languages

3. **Semantic Search**: Pattern-based (not true NLP embeddings yet)
   - Solution: Can add sentence-transformers for deeper understanding

4. **Location Context**: Requires manual trigger
   - Solution: Could auto-contextualize on photo upload

### Future Enhancements

- [ ] Multi-language Wikipedia support
- [ ] Automatic location contextualization
- [ ] Export journey maps as PDF/KML
- [ ] Advanced NLP with sentence embeddings
- [ ] Voice search input
- [ ] Search filters (date range, people, location)
- [ ] Saved searches
- [ ] Search history

## Troubleshooting

### Journey Mapping Issues

**Problem**: "No photos with GPS data found"
- **Solution**: Ensure photos have GPS EXIF data or manually add coordinates

**Problem**: Too many waypoints warning
- **Solution**: Use date filtering to reduce route points

### Location Context Issues

**Problem**: "Cannot connect to Wikipedia/Nominatim"
- **Solution**: Check internet connection

**Problem**: Wrong location names
- **Solution**: GPS accuracy may be low; verify coordinates

### Semantic Search Issues

**Problem**: No results for valid query
- **Solution**:
  - Ensure photos have relevant metadata (captions, people, locations)
  - Try broader queries
  - Add Wikipedia context for better location search

**Problem**: Low relevance scores
- **Solution**:
  - Be more specific with names and places
  - Identify faces before searching for people
  - Generate captions for content-based search

## API Documentation

See docstrings in each module for detailed API documentation:

```bash
python -c "from travel_log import create_journey_mapper; help(create_journey_mapper)"
python -c "from travel_log import create_location_contextualizer; help(create_location_contextualizer)"
python -c "from travel_log import create_semantic_search; help(create_semantic_search)"
```

## Contributing

To extend these features:

1. **Add new search filters**: Edit `semantic_search.py` → `parse_query()`
2. **Add new map providers**: Edit `journey_mapper.py` → `create_*_url()`
3. **Add new context sources**: Edit `location_context.py` → Add new methods

## Questions?

For issues or questions:
1. Check logs in Streamlit app (errors are displayed)
2. Review module docstrings
3. Test with example scripts (`test_semantic_search.py`)
4. Check Qdrant connection: `curl http://sapphire:6333`

---

**Version**: 1.0
**Date**: 2025-11-25
**Author**: Travel Log Development Team

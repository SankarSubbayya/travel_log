"""
Location Contextualization - Wikipedia integration and embeddings

This module enriches travel photos with location context:
1. Reverse geocode GPS coordinates to place names
2. Fetch Wikipedia articles about locations
3. Generate vector embeddings from Wikipedia text
4. Store in Qdrant for semantic search
"""

from typing import List, Dict, Optional, Tuple
import logging
import requests
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class LocationContextualizer:
    """
    Add Wikipedia context to locations in travel photos.

    Features:
    - Reverse geocoding (GPS → place name)
    - Wikipedia article retrieval
    - Text embedding generation
    - Qdrant storage for semantic search
    """

    def __init__(self, qdrant_store=None, embedding_model=None):
        """
        Initialize location contextualizer.

        Args:
            qdrant_store: Optional TravelLogQdrantStore instance
            embedding_model: Optional embedding model (sentence-transformers)
        """
        self.qdrant_store = qdrant_store
        self.embedding_model = embedding_model
        self._cache = {}  # Cache for Wikipedia results

    def reverse_geocode(
        self,
        latitude: float,
        longitude: float
    ) -> Dict:
        """
        Convert GPS coordinates to place name using Nominatim.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            Dict with place information (city, country, etc.)
        """
        # Use OpenStreetMap Nominatim (free, no API key needed)
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': latitude,
            'lon': longitude,
            'format': 'json',
            'addressdetails': 1
        }
        headers = {
            'User-Agent': 'TravelLog/1.0'
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            address = data.get('address', {})
            return {
                'display_name': data.get('display_name', ''),
                'city': (address.get('city') or
                        address.get('town') or
                        address.get('village') or
                        address.get('municipality', '')),
                'state': address.get('state', ''),
                'country': address.get('country', ''),
                'country_code': address.get('country_code', '').upper(),
                'raw': data
            }
        except Exception as e:
            logger.error(f"Reverse geocoding error: {e}")
            return {'error': str(e)}

    def search_wikipedia(
        self,
        query: str,
        language: str = 'en'
    ) -> Optional[Dict]:
        """
        Search Wikipedia for a location/place.

        Args:
            query: Place name to search
            language: Wikipedia language code (default: 'en')

        Returns:
            Dict with Wikipedia page data (title, summary, full text, URL)
        """
        # Check cache
        cache_key = f"{language}:{query}"
        if cache_key in self._cache:
            logger.info(f"Using cached Wikipedia data for: {query}")
            return self._cache[cache_key]

        # Wikipedia API endpoint
        api_url = f"https://{language}.wikipedia.org/w/api.php"

        # First, search for the page
        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': 1
        }

        try:
            response = requests.get(api_url, params=search_params, timeout=10)
            response.raise_for_status()
            search_data = response.json()

            search_results = search_data.get('query', {}).get('search', [])
            if not search_results:
                logger.warning(f"No Wikipedia results for: {query}")
                return None

            # Get the top result's page title
            page_title = search_results[0]['title']
            page_id = search_results[0]['pageid']

            # Now fetch the full page content
            page_params = {
                'action': 'query',
                'prop': 'extracts|info',
                'pageids': page_id,
                'explaintext': True,  # Plain text, not HTML
                'exintro': False,  # Get full text, not just intro
                'inprop': 'url',
                'format': 'json'
            }

            response = requests.get(api_url, params=page_params, timeout=10)
            response.raise_for_status()
            page_data = response.json()

            page = page_data['query']['pages'][str(page_id)]

            result = {
                'title': page_title,
                'pageid': page_id,
                'extract': page.get('extract', ''),
                'url': page.get('fullurl', ''),
                'summary': self._extract_summary(page.get('extract', ''))
            }

            # Cache the result
            self._cache[cache_key] = result
            logger.info(f"Retrieved Wikipedia article: {page_title}")

            return result

        except Exception as e:
            logger.error(f"Wikipedia search error for '{query}': {e}")
            return None

    def _extract_summary(self, full_text: str, max_sentences: int = 3) -> str:
        """
        Extract a summary (first N sentences) from Wikipedia text.

        Args:
            full_text: Full Wikipedia article text
            max_sentences: Number of sentences to include

        Returns:
            Summary string
        """
        if not full_text:
            return ""

        # Split by periods followed by space and capital letter
        sentences = []
        current = ""
        for char in full_text:
            current += char
            if char == '.' and len(current) > 20:
                sentences.append(current.strip())
                current = ""
                if len(sentences) >= max_sentences:
                    break

        return ' '.join(sentences)

    def get_location_context(
        self,
        latitude: float,
        longitude: float
    ) -> Dict:
        """
        Get complete context for a GPS location.

        This combines:
        - Reverse geocoding (GPS → place name)
        - Wikipedia article retrieval
        - Summary extraction

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            Dict with place info and Wikipedia context
        """
        # Reverse geocode
        place_info = self.reverse_geocode(latitude, longitude)

        if 'error' in place_info:
            return place_info

        # Try to find Wikipedia article
        # Search in order: city, state, country
        wiki_data = None
        search_terms = [
            place_info.get('city'),
            place_info.get('state'),
            place_info.get('country')
        ]

        for term in search_terms:
            if term:
                wiki_data = self.search_wikipedia(term)
                if wiki_data:
                    break

        return {
            'place': place_info,
            'wikipedia': wiki_data,
            'latitude': latitude,
            'longitude': longitude
        }

    def generate_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate vector embedding from text using sentence-transformers.

        Args:
            text: Text to embed

        Returns:
            List of floats (embedding vector)
        """
        if not self.embedding_model:
            logger.warning("No embedding model available")
            return None

        try:
            # Use sentence-transformers
            from sentence_transformers import SentenceTransformer

            if isinstance(self.embedding_model, str):
                # Load model if string name provided
                self.embedding_model = SentenceTransformer(self.embedding_model)

            embedding = self.embedding_model.encode(text)
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return None

    def contextualize_photo(
        self,
        photo_id: str,
        latitude: float,
        longitude: float,
        update_qdrant: bool = True
    ) -> Dict:
        """
        Add location context to a photo and optionally update Qdrant.

        Args:
            photo_id: Photo ID in Qdrant
            latitude: Photo GPS latitude
            longitude: Photo GPS longitude
            update_qdrant: Whether to update Qdrant with context

        Returns:
            Dict with location context
        """
        # Get location context
        context = self.get_location_context(latitude, longitude)

        if update_qdrant and self.qdrant_store:
            # Update photo in Qdrant with location context
            try:
                from qdrant_client.models import SetPayload

                payload = {
                    'location_context': {
                        'place_name': context['place'].get('display_name', ''),
                        'city': context['place'].get('city', ''),
                        'country': context['place'].get('country', ''),
                        'wikipedia_title': context['wikipedia']['title'] if context.get('wikipedia') else None,
                        'wikipedia_summary': context['wikipedia']['summary'] if context.get('wikipedia') else None,
                        'wikipedia_url': context['wikipedia']['url'] if context.get('wikipedia') else None
                    }
                }

                self.qdrant_store.client.set_payload(
                    collection_name=self.qdrant_store.collection_name,
                    payload=payload,
                    points=[photo_id]
                )

                logger.info(f"Updated photo {photo_id} with location context")

            except Exception as e:
                logger.error(f"Error updating Qdrant: {e}")

        return context

    def bulk_contextualize_photos(
        self,
        limit: int = 100
    ) -> Dict:
        """
        Add location context to all photos with GPS data.

        Args:
            limit: Maximum number of photos to process

        Returns:
            Summary dict with processing stats
        """
        if not self.qdrant_store:
            raise ValueError("QdrantStore required for bulk contextualization")

        # Get all photos with GPS
        photos = self.qdrant_store.get_all_photos(limit=limit)

        stats = {
            'total_photos': len(photos),
            'contextualized': 0,
            'skipped_no_gps': 0,
            'errors': 0
        }

        for photo in photos:
            lat = photo.get('latitude')
            lon = photo.get('longitude')
            photo_id = photo.get('id')

            if not lat or not lon:
                stats['skipped_no_gps'] += 1
                continue

            try:
                self.contextualize_photo(photo_id, lat, lon, update_qdrant=True)
                stats['contextualized'] += 1
            except Exception as e:
                logger.error(f"Error contextualizing photo {photo_id}: {e}")
                stats['errors'] += 1

        logger.info(f"Bulk contextualization complete: {stats}")
        return stats


def create_location_contextualizer(
    qdrant_store=None,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> LocationContextualizer:
    """
    Factory function to create LocationContextualizer.

    Args:
        qdrant_store: Optional TravelLogQdrantStore instance
        embedding_model: Sentence-transformers model name

    Returns:
        LocationContextualizer instance
    """
    return LocationContextualizer(
        qdrant_store=qdrant_store,
        embedding_model=embedding_model
    )

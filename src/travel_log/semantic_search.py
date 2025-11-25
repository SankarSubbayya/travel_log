"""
Semantic Search - Natural language search for travel photos

This module enables natural language queries like:
- "When, where and with whom did I see the turtles on the beach?"
- "Show me photos from my beach vacation last summer"
- "Find photos with Sarah in Paris"

Uses:
1. Query parsing and entity extraction
2. Semantic search across photo metadata, captions, Wikipedia context
3. Multi-modal filtering (time, location, people, content)
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import re

logger = logging.getLogger(__name__)


class SemanticPhotoSearch:
    """
    Natural language search for travel photos.

    Features:
    - Parse natural language queries
    - Extract entities (people, places, dates, keywords)
    - Search across all photo metadata
    - Rank results by relevance
    """

    def __init__(self, qdrant_store=None, embedding_model=None):
        """
        Initialize semantic search.

        Args:
            qdrant_store: TravelLogQdrantStore instance
            embedding_model: Optional embedding model for semantic similarity
        """
        self.qdrant_store = qdrant_store
        self.embedding_model = embedding_model

    def parse_query(self, query: str) -> Dict:
        """
        Parse natural language query to extract search parameters.

        Args:
            query: Natural language search query

        Returns:
            Dict with extracted entities and filters
        """
        query_lower = query.lower()

        # Initialize parsed components
        parsed = {
            'original_query': query,
            'keywords': [],
            'people': [],
            'locations': [],
            'time_filters': {},
            'content_keywords': []
        }

        # Extract people names (capitalized words)
        # Look for patterns like "with [Name]" or proper nouns
        name_patterns = [
            r'with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'(?:^|[.,;]\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:and|in|at|near)'
        ]

        for pattern in name_patterns:
            matches = re.findall(pattern, query)
            parsed['people'].extend(matches)

        # Extract time references
        time_patterns = {
            'yesterday': lambda: datetime.now() - timedelta(days=1),
            'last week': lambda: datetime.now() - timedelta(weeks=1),
            'last month': lambda: datetime.now() - timedelta(days=30),
            'last year': lambda: datetime.now() - timedelta(days=365)
        }

        for time_phrase, time_func in time_patterns.items():
            if time_phrase in query_lower:
                parsed['time_filters']['after'] = time_func()
                break

        # Extract specific date patterns (YYYY-MM-DD, Month YYYY, etc.)
        date_patterns = [
            (r'(\d{4})-(\d{2})-(\d{2})', 'full_date'),
            (r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', 'month_year'),
            (r'in\s+(\d{4})', 'year')
        ]

        for pattern, date_type in date_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if date_type == 'full_date':
                    year, month, day = match.groups()
                    parsed['time_filters']['specific_date'] = f"{year}-{month}-{day}"
                elif date_type == 'month_year':
                    month_name, year = match.groups()
                    parsed['time_filters']['month'] = month_name
                    parsed['time_filters']['year'] = year
                elif date_type == 'year':
                    parsed['time_filters']['year'] = match.group(1)
                break

        # Extract location references
        location_keywords = ['in', 'at', 'near', 'from']
        for keyword in location_keywords:
            pattern = rf'\b{keyword}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
            matches = re.findall(pattern, query)
            parsed['locations'].extend(matches)

        # Extract content keywords (things the user wants to see)
        # Look for nouns after question words
        content_patterns = [
            r'see(?:ing)?\s+(?:the\s+)?(\w+)',
            r'show(?:ing)?\s+(?:me\s+)?(?:the\s+)?(\w+)',
            r'find\s+(?:the\s+)?(\w+)',
            r'photos?\s+(?:of|with)\s+(\w+)'
        ]

        for pattern in content_patterns:
            matches = re.findall(pattern, query_lower)
            parsed['content_keywords'].extend(matches)

        # Additional keywords (filter out common words)
        stopwords = {'the', 'and', 'or', 'in', 'at', 'on', 'with', 'when', 'where', 'who',
                    'what', 'how', 'i', 'me', 'my', 'did', 'do', 'see', 'show', 'find',
                    'photos', 'photo', 'pictures', 'picture', 'images', 'image'}

        words = query_lower.split()
        parsed['keywords'] = [w for w in words if w not in stopwords and len(w) > 2]

        logger.info(f"Parsed query: {parsed}")
        return parsed

    def search(
        self,
        query: str,
        limit: int = 50,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search photos using natural language query.

        Args:
            query: Natural language search query
            limit: Maximum number of results
            score_threshold: Minimum relevance score (0-1)

        Returns:
            List of matching photos with relevance scores
        """
        if not self.qdrant_store:
            raise ValueError("QdrantStore required for search")

        # Parse the query
        parsed = self.parse_query(query)

        # Get all photos (we'll filter and rank them)
        all_photos = self.qdrant_store.get_all_photos(limit=limit * 2)

        # Score and filter photos
        scored_results = []
        for photo in all_photos:
            score = self._calculate_relevance_score(photo, parsed)
            if score > score_threshold:
                result = photo.copy()
                result['relevance_score'] = score
                result['match_reasons'] = self._get_match_reasons(photo, parsed)
                scored_results.append(result)

        # Sort by relevance score
        scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Return top results
        results = scored_results[:limit]
        logger.info(f"Found {len(results)} matching photos for query: {query}")

        return results

    def _calculate_relevance_score(self, photo: Dict, parsed_query: Dict) -> float:
        """
        Calculate relevance score for a photo based on parsed query.

        Args:
            photo: Photo metadata dict
            parsed_query: Parsed query dict

        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        max_score = 0.0

        # People match (high weight)
        if parsed_query['people']:
            max_score += 0.3
            photo_people = photo.get('people', [])
            for person in parsed_query['people']:
                if any(person.lower() in p.lower() for p in photo_people):
                    score += 0.3
                    break

        # Location match (high weight)
        if parsed_query['locations']:
            max_score += 0.25
            location_text = (
                photo.get('location_context', {}).get('place_name', '') +
                photo.get('location_context', {}).get('city', '') +
                photo.get('location_context', {}).get('country', '')
            ).lower()

            for location in parsed_query['locations']:
                if location.lower() in location_text:
                    score += 0.25
                    break

        # Time filter match (medium weight)
        if parsed_query['time_filters']:
            max_score += 0.2
            photo_date = photo.get('datetime')
            if photo_date:
                if isinstance(photo_date, str):
                    try:
                        photo_date = datetime.fromisoformat(photo_date)
                    except:
                        photo_date = None

                if photo_date:
                    time_match = self._check_time_match(photo_date, parsed_query['time_filters'])
                    if time_match:
                        score += 0.2

        # Content keyword match (medium weight)
        if parsed_query['content_keywords']:
            max_score += 0.15
            content_text = (
                photo.get('captions', {}).get('caption', '') +
                photo.get('captions', {}).get('title', '') +
                photo.get('location_context', {}).get('wikipedia_summary', '')
            ).lower()

            for keyword in parsed_query['content_keywords']:
                if keyword in content_text:
                    score += 0.15 / len(parsed_query['content_keywords'])

        # General keyword match (low weight)
        if parsed_query['keywords']:
            max_score += 0.1
            all_text = (
                str(photo.get('captions', {})) +
                str(photo.get('location_context', {})) +
                str(photo.get('people', []))
            ).lower()

            matched_keywords = sum(1 for kw in parsed_query['keywords'] if kw in all_text)
            if matched_keywords > 0:
                score += 0.1 * (matched_keywords / len(parsed_query['keywords']))

        # Normalize score
        if max_score > 0:
            return min(score / max_score, 1.0)
        else:
            return 0.0

    def _check_time_match(self, photo_date: datetime, time_filters: Dict) -> bool:
        """
        Check if photo date matches time filters.

        Args:
            photo_date: Photo datetime
            time_filters: Time filter dict

        Returns:
            True if matches
        """
        # Check 'after' filter
        if 'after' in time_filters:
            if photo_date < time_filters['after']:
                return False

        # Check 'before' filter
        if 'before' in time_filters:
            if photo_date > time_filters['before']:
                return False

        # Check specific date
        if 'specific_date' in time_filters:
            date_str = photo_date.strftime('%Y-%m-%d')
            if date_str != time_filters['specific_date']:
                return False

        # Check year
        if 'year' in time_filters:
            if str(photo_date.year) != str(time_filters['year']):
                return False

        # Check month
        if 'month' in time_filters:
            month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                          'july', 'august', 'september', 'october', 'november', 'december']
            month_num = photo_date.month
            if month_names[month_num - 1] != time_filters['month'].lower():
                return False

        return True

    def _get_match_reasons(self, photo: Dict, parsed_query: Dict) -> List[str]:
        """
        Generate human-readable reasons why this photo matched.

        Args:
            photo: Photo metadata
            parsed_query: Parsed query

        Returns:
            List of match reason strings
        """
        reasons = []

        # People matches
        photo_people = photo.get('people', [])
        for person in parsed_query['people']:
            if any(person.lower() in p.lower() for p in photo_people):
                reasons.append(f"Contains person: {person}")

        # Location matches
        location_context = photo.get('location_context', {})
        for location in parsed_query['locations']:
            if location.lower() in str(location_context).lower():
                place = location_context.get('city') or location_context.get('country', location)
                reasons.append(f"Location: {place}")

        # Time matches
        if parsed_query['time_filters'] and photo.get('datetime'):
            date_str = photo['datetime']
            if isinstance(date_str, datetime):
                date_str = date_str.strftime('%Y-%m-%d')
            reasons.append(f"Date: {date_str}")

        # Content matches
        captions = photo.get('captions', {})
        for keyword in parsed_query['content_keywords']:
            caption_text = str(captions).lower()
            if keyword in caption_text:
                reasons.append(f"Content: {keyword}")

        return reasons

    def search_by_similarity(
        self,
        text_query: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search using semantic text similarity (requires embedding model).

        Args:
            text_query: Text to search for
            limit: Maximum results

        Returns:
            List of similar photos
        """
        if not self.embedding_model:
            logger.warning("Embedding model not available, falling back to keyword search")
            return self.search(text_query, limit=limit)

        try:
            from sentence_transformers import SentenceTransformer

            if isinstance(self.embedding_model, str):
                self.embedding_model = SentenceTransformer(self.embedding_model)

            # Generate query embedding
            query_embedding = self.embedding_model.encode(text_query)

            # Search Qdrant (this requires caption embeddings to be stored)
            # For now, fall back to parsed search
            logger.info("Semantic similarity search - using parsed query fallback")
            return self.search(text_query, limit=limit)

        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return self.search(text_query, limit=limit)


def create_semantic_search(
    qdrant_store=None,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> SemanticPhotoSearch:
    """
    Factory function to create SemanticPhotoSearch instance.

    Args:
        qdrant_store: TravelLogQdrantStore instance
        embedding_model: Sentence-transformers model name

    Returns:
        SemanticPhotoSearch instance
    """
    return SemanticPhotoSearch(
        qdrant_store=qdrant_store,
        embedding_model=embedding_model
    )

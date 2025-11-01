#!/usr/bin/env python3
"""
Process Extracted Faces Script

This script processes extracted face images and identifies matches against a face database.
It's designed to work with faces extracted from photos using the face detection module.

Usage:
    python process_extracted_faces.py --faces-dir ./extracted_faces --db-dir ./face_database
    python process_extracted_faces.py --faces-dir ./extracted_faces --db-dir ./face_database --model ArcFace
    python process_extracted_faces.py --faces-dir ./extracted_faces --db-dir ./face_database --confidence-threshold 0.5
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime
import csv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from travel_log import FaceEmbeddings, FaceLabeler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtractedFaceProcessor:
    """Process extracted face images and identify matches."""

    AVAILABLE_MODELS = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace',
                        'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']

    def __init__(
        self,
        faces_dir: Path,
        database_dir: Path,
        model_name: str = 'Facenet512',
        distance_metric: str = 'cosine',
        confidence_threshold: float = 0.6
    ):
        """
        Initialize the processor.

        Args:
            faces_dir: Directory containing extracted face images
            database_dir: Directory containing labeled face database
            model_name: Face recognition model to use
            distance_metric: Distance metric for comparison
            confidence_threshold: Minimum confidence for matches
        """
        self.faces_dir = Path(faces_dir)
        self.database_dir = Path(database_dir)
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.confidence_threshold = confidence_threshold

        # Validate directories
        if not self.faces_dir.exists():
            raise FileNotFoundError(f"Faces directory not found: {faces_dir}")

        if not self.database_dir.exists():
            logger.warning(f"Database directory does not exist: {database_dir}")
            self.database_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.embeddings = FaceEmbeddings(
            model_name=model_name,
            distance_metric=distance_metric
        )

        self.labeler = FaceLabeler(
            database_path=str(database_dir),
            model_name=model_name,
            distance_metric=distance_metric
        )

        logger.info(f"Initialized with model: {model_name}, metric: {distance_metric}")

    def get_face_files(self) -> List[Path]:
        """Get all supported image files from faces directory."""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        face_files = []

        for ext in supported_extensions:
            face_files.extend(self.faces_dir.glob(f'*{ext}'))
            face_files.extend(self.faces_dir.glob(f'*{ext.upper()}'))

        return sorted(face_files)

    def identify_face(self, face_path: Path) -> Dict:
        """
        Identify a single face image.

        Args:
            face_path: Path to face image file

        Returns:
            Dictionary with identification results
        """
        try:
            # Check if database has any faces
            people_dirs = [d for d in self.database_dir.iterdir() if d.is_dir()]

            if not people_dirs:
                logger.warning(f"Database is empty, skipping identification for {face_path.name}")
                return {
                    'face_file': face_path.name,
                    'status': 'skipped',
                    'reason': 'empty_database',
                    'match': None,
                    'confidence': None
                }

            # Find matches in database
            results = self.labeler.find_face(str(face_path))

            if not results or len(results) == 0 or results[0].empty:
                return {
                    'face_file': face_path.name,
                    'status': 'no_match',
                    'match': None,
                    'confidence': None
                }

            # Get best match
            matches_df = results[0]
            if matches_df.empty:
                return {
                    'face_file': face_path.name,
                    'status': 'no_match',
                    'match': None,
                    'confidence': None
                }

            best_match = matches_df.iloc[0]
            distance = best_match['distance']

            # Convert distance to confidence score (lower distance = higher confidence)
            # Using cosine distance (0-2 range, 0 = perfect match)
            confidence = max(0, 1 - (distance / 2.0))

            # Extract person name from identity path
            identity = best_match['identity']
            person_name = Path(identity).parent.name

            result = {
                'face_file': face_path.name,
                'status': 'identified' if confidence >= self.confidence_threshold else 'low_confidence',
                'match': person_name,
                'confidence': round(confidence, 4),
                'distance': round(distance, 4),
                'matched_image': best_match.get('identity', 'Unknown')
            }

            return result

        except Exception as e:
            logger.error(f"Error identifying face {face_path.name}: {str(e)}")
            return {
                'face_file': face_path.name,
                'status': 'error',
                'reason': str(e),
                'match': None,
                'confidence': None
            }

    def process_all_faces(self) -> Tuple[List[Dict], Dict]:
        """
        Process all extracted faces and identify matches.

        Returns:
            Tuple of (results list, summary dict)
        """
        face_files = self.get_face_files()

        if not face_files:
            logger.warning("No face files found in directory")
            return [], {'total': 0, 'identified': 0, 'no_match': 0, 'errors': 0}

        logger.info(f"Found {len(face_files)} face files to process")

        results = []
        summary = {
            'total': len(face_files),
            'identified': 0,
            'low_confidence': 0,
            'no_match': 0,
            'skipped': 0,
            'errors': 0,
            'timestamp': datetime.now().isoformat()
        }

        for idx, face_file in enumerate(face_files, 1):
            logger.info(f"Processing {idx}/{len(face_files)}: {face_file.name}")
            result = self.identify_face(face_file)
            results.append(result)

            # Update summary
            status = result.get('status')
            if status == 'identified':
                summary['identified'] += 1
            elif status == 'low_confidence':
                summary['low_confidence'] += 1
            elif status == 'no_match':
                summary['no_match'] += 1
            elif status == 'skipped':
                summary['skipped'] += 1
            elif status == 'error':
                summary['errors'] += 1

        return results, summary

    def save_results(self, results: List[Dict], output_dir: Path):
        """Save results to JSON and CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save as JSON
        json_file = output_dir / f'face_identification_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {json_file}")

        # Save as CSV
        csv_file = output_dir / f'face_identification_{timestamp}.csv'
        if results:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            logger.info(f"Results saved to {csv_file}")

    def print_summary(self, results: List[Dict], summary: Dict):
        """Print a nice summary of results."""
        print("\n" + "="*70)
        print("FACE IDENTIFICATION SUMMARY")
        print("="*70)
        print(f"Model: {self.model_name}")
        print(f"Distance Metric: {self.distance_metric}")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"Timestamp: {summary['timestamp']}")
        print("-"*70)
        print(f"Total Faces: {summary['total']}")
        print(f"  ✅ Identified: {summary['identified']}")
        print(f"  ⚠️  Low Confidence: {summary['low_confidence']}")
        print(f"  ❌ No Match: {summary['no_match']}")
        print(f"  ⊘ Skipped: {summary['skipped']}")
        print(f"  ⚠️  Errors: {summary['errors']}")
        print("-"*70)

        # Print identified faces
        identified = [r for r in results if r.get('status') == 'identified']
        if identified:
            print("\n✅ IDENTIFIED FACES:")
            for result in identified:
                print(f"  • {result['face_file']:30} → {result['match']:20} "
                      f"(confidence: {result['confidence']:.1%})")

        # Print low confidence matches
        low_conf = [r for r in results if r.get('status') == 'low_confidence']
        if low_conf:
            print("\n⚠️  LOW CONFIDENCE MATCHES:")
            for result in low_conf:
                print(f"  • {result['face_file']:30} → {result['match']:20} "
                      f"(confidence: {result['confidence']:.1%})")

        # Print unmatched faces
        no_match = [r for r in results if r.get('status') == 'no_match']
        if no_match:
            print(f"\n❌ NO MATCHES FOUND ({len(no_match)} faces):")
            for result in no_match[:10]:  # Show first 10
                print(f"  • {result['face_file']}")
            if len(no_match) > 10:
                print(f"  ... and {len(no_match) - 10} more")

        print("\n" + "="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Process extracted faces and identify matches against a database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python process_extracted_faces.py --faces-dir ./extracted_faces --db-dir ./face_database

  # Use ArcFace model (faster)
  python process_extracted_faces.py --faces-dir ./extracted_faces --db-dir ./face_database --model ArcFace

  # Lower confidence threshold
  python process_extracted_faces.py --faces-dir ./extracted_faces --db-dir ./face_database --confidence-threshold 0.5

  # Save results to custom directory
  python process_extracted_faces.py --faces-dir ./extracted_faces --db-dir ./face_database --output ./results
        """
    )

    parser.add_argument(
        '--faces-dir',
        type=str,
        required=True,
        help='Directory containing extracted face images'
    )

    parser.add_argument(
        '--db-dir',
        type=str,
        required=True,
        help='Directory containing labeled face database'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='Facenet512',
        choices=['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace',
                'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace'],
        help='Face recognition model to use (default: Facenet512)'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='cosine',
        choices=['cosine', 'euclidean', 'euclidean_l2'],
        help='Distance metric for comparison (default: cosine)'
    )

    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.6,
        help='Minimum confidence for accepting matches (default: 0.6)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./identification_results',
        help='Output directory for results (default: ./identification_results)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to files'
    )

    args = parser.parse_args()

    try:
        # Initialize processor
        processor = ExtractedFaceProcessor(
            faces_dir=args.faces_dir,
            database_dir=args.db_dir,
            model_name=args.model,
            distance_metric=args.metric,
            confidence_threshold=args.confidence_threshold
        )

        # Process faces
        logger.info("Starting face identification process...")
        results, summary = processor.process_all_faces()

        # Print summary
        processor.print_summary(results, summary)

        # Save results
        if not args.no_save and results:
            processor.save_results(results, Path(args.output))

        logger.info("Face identification complete!")
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Face Database Management Utility

Manage the face database for identification tasks. Supports:
- Adding new people to the database
- Organizing face images into person directories
- Viewing database contents and statistics
- Generating embeddings for the database
- Finding and removing duplicate faces
- Importing faces from extracted face directories

Usage:
    python manage_face_database.py add-person --db-dir ./face_database --name "John Doe" --images ./john_photos
    python manage_face_database.py list --db-dir ./face_database
    python manage_face_database.py stats --db-dir ./face_database
    python manage_face_database.py generate-embeddings --db-dir ./face_database
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json
import shutil
from datetime import datetime
import csv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from travel_log import FaceEmbeddings, ensure_compatible_image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceDatabaseManager:
    """Manage face database for identification."""

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.heic', '.heif'}

    def __init__(self, database_dir: Path):
        """
        Initialize the database manager.

        Args:
            database_dir: Path to face database directory
        """
        self.database_dir = Path(database_dir)
        self.database_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file for tracking
        self.metadata_file = self.database_dir / 'database_metadata.json'
        self.embeddings_file = self.database_dir / 'embeddings.json'

        logger.info(f"Initialized database manager at {self.database_dir}")

    def add_person(self, name: str, image_paths: List[Path], overwrite: bool = False) -> Dict:
        """
        Add a new person to the database.

        Args:
            name: Person's name
            image_paths: List of image file paths
            overwrite: Whether to overwrite existing person directory

        Returns:
            Dictionary with results
        """
        person_dir = self.database_dir / name


        if person_dir.exists() and not overwrite:
            logger.warning(f"Person '{name}' already exists. Use --overwrite to replace.")
            return {
                'status': 'exists',
                'name': name,
                'message': f"Person '{name}' already exists"
            }

        if person_dir.exists() and overwrite:
            logger.info(f"Overwriting existing person directory: {name}")
            shutil.rmtree(person_dir)

        person_dir.mkdir(parents=True, exist_ok=True)

        # Copy images to person directory
        added_count = 0
        failed_count = 0
        errors = []

        for img_path in image_paths:
            img_path = Path(img_path)

            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                errors.append(f"Not found: {img_path.name}")
                failed_count += 1
                continue

            try:
                # Check if file needs conversion
                if img_path.suffix.lower() in {'.heic', '.heif'}:
                    logger.info(f"Converting HEIC: {img_path.name}")
                    img_path = Path(ensure_compatible_image(str(img_path)))

                # Copy to person directory
                dest = person_dir / img_path.name
                shutil.copy2(img_path, dest)
                added_count += 1
                logger.info(f"Added {img_path.name} for {name}")

            except Exception as e:
                logger.error(f"Error adding image {img_path.name}: {str(e)}")
                errors.append(f"Error: {img_path.name}")
                failed_count += 1

        logger.info(f"Added {added_count} images for '{name}'")

        return {
            'status': 'success',
            'name': name,
            'added': added_count,
            'failed': failed_count,
            'errors': errors
        }

    def remove_person(self, name: str, confirm: bool = False) -> Dict:
        """
        Remove a person from the database.

        Args:
            name: Person's name
            confirm: Whether to confirm deletion

        Returns:
            Dictionary with results
        """
        person_dir = self.database_dir / name

        if not person_dir.exists():
            logger.warning(f"Person '{name}' not found")
            return {
                'status': 'not_found',
                'name': name,
                'message': f"Person '{name}' not found"
            }

        if not confirm:
            return {
                'status': 'pending_confirmation',
                'name': name,
                'message': f"Use --confirm to delete '{name}' and all their images"
            }

        try:
            shutil.rmtree(person_dir)
            logger.info(f"Removed person: {name}")
            return {
                'status': 'success',
                'name': name,
                'message': f"Successfully removed '{name}'"
            }
        except Exception as e:
            logger.error(f"Error removing person {name}: {str(e)}")
            return {
                'status': 'error',
                'name': name,
                'message': str(e)
            }

    def list_people(self) -> Dict:
        """
        List all people in the database.

        Returns:
            Dictionary with people and image counts
        """
        people = {}

        for person_dir in self.database_dir.iterdir():
            if person_dir.is_dir() and person_dir.name != '.metadata':
                image_count = len([
                    f for f in person_dir.glob('*')
                    if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
                ])
                people[person_dir.name] = image_count

        return {
            'total_people': len(people),
            'people': people,
            'timestamp': datetime.now().isoformat()
        }

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        people_data = self.list_people()
        people = people_data['people']

        total_images = sum(people.values())
        avg_images_per_person = total_images / len(people) if people else 0

        return {
            'total_people': people_data['total_people'],
            'total_images': total_images,
            'average_images_per_person': round(avg_images_per_person, 2),
            'people': people,
            'database_size_bytes': self._get_directory_size(),
            'timestamp': datetime.now().isoformat()
        }

    def _get_directory_size(self) -> int:
        """Calculate total directory size in bytes."""
        total = 0
        for file in self.database_dir.rglob('*'):
            if file.is_file():
                total += file.stat().st_size
        return total

    def import_extracted_faces(self, faces_dir: Path, organize_by_date: bool = False) -> Dict:
        """
        Import extracted faces from a directory.

        Args:
            faces_dir: Directory containing extracted faces
            organize_by_date: Whether to organize by extraction date

        Returns:
            Dictionary with import results
        """
        faces_dir = Path(faces_dir)

        if not faces_dir.exists():
            return {
                'status': 'error',
                'message': f"Faces directory not found: {faces_dir}"
            }

        # Find all face images
        face_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            face_files.extend(faces_dir.glob(f'*{ext}'))
            face_files.extend(faces_dir.glob(f'*{ext.upper()}'))

        if not face_files:
            return {
                'status': 'no_faces',
                'message': f"No face images found in {faces_dir}"
            }

        logger.info(f"Found {len(face_files)} faces to import")

        # Create 'unknown' directory for unorganized faces
        unknown_dir = self.database_dir / 'unknown'
        unknown_dir.mkdir(parents=True, exist_ok=True)

        imported_count = 0
        for face_file in face_files:
            try:
                dest = unknown_dir / face_file.name
                shutil.copy2(face_file, dest)
                imported_count += 1
            except Exception as e:
                logger.error(f"Error importing {face_file.name}: {str(e)}")

        logger.info(f"Imported {imported_count} faces to 'unknown' directory")

        return {
            'status': 'success',
            'imported': imported_count,
            'total': len(face_files),
            'destination': str(unknown_dir),
            'message': f"Imported {imported_count} faces. Manually organize them into person directories."
        }

    def generate_embeddings(self, model_name: str = 'Facenet512') -> Dict:
        """
        Generate face embeddings for database.

        Args:
            model_name: Face recognition model to use

        Returns:
            Dictionary with embedding results
        """
        logger.info(f"Generating embeddings using {model_name}")

        embeddings_gen = FaceEmbeddings(model_name=model_name)
        embeddings_dict = {}
        errors = []

        people_data = self.list_people()

        for person_name, image_count in people_data['people'].items():
            person_dir = self.database_dir / person_name
            embeddings_dict[person_name] = []

            for img_file in person_dir.glob('*'):
                if img_file.is_file() and img_file.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    try:
                        embedding = embeddings_gen.generate_embedding(str(img_file))
                        embeddings_dict[person_name].append({
                            'image': img_file.name,
                            'embedding': embedding.tolist()  # Convert numpy array to list
                        })
                    except Exception as e:
                        logger.error(f"Error generating embedding for {img_file.name}: {str(e)}")
                        errors.append(f"{img_file.name}: {str(e)}")

        # Save embeddings
        with open(self.embeddings_file, 'w') as f:
            json.dump(embeddings_dict, f, indent=2)

        logger.info(f"Saved embeddings to {self.embeddings_file}")

        return {
            'status': 'success',
            'model': model_name,
            'people_processed': len(embeddings_dict),
            'embeddings_file': str(self.embeddings_file),
            'errors': errors
        }

    def print_statistics(self, stats: Dict):
        """Print formatted statistics."""
        print("\n" + "="*70)
        print("FACE DATABASE STATISTICS")
        print("="*70)
        print(f"Timestamp: {stats['timestamp']}")
        print("-"*70)
        print(f"Total People: {stats['total_people']}")
        print(f"Total Images: {stats['total_images']}")
        print(f"Avg Images per Person: {stats['average_images_per_person']}")
        print(f"Database Size: {stats['database_size_bytes'] / (1024*1024):.2f} MB")
        print("-"*70)

        if stats['people']:
            print("\n📋 PEOPLE IN DATABASE:")
            for person, count in sorted(stats['people'].items()):
                print(f"  • {person:30} ({count} images)")

        print("\n" + "="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Manage face database for identification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  add-person         Add a new person to the database
  remove-person      Remove a person from the database
  list               List all people in database
  stats              Show database statistics
  generate-embeddings Generate face embeddings for all people
  import-extracted   Import extracted faces from a directory

Examples:
  # Add a person with photos
  python manage_face_database.py add-person --db-dir ./face_database \\
      --name "John Doe" --images ./john_photos/*.jpg

  # List all people
  python manage_face_database.py list --db-dir ./face_database

  # Show statistics
  python manage_face_database.py stats --db-dir ./face_database

  # Generate embeddings
  python manage_face_database.py generate-embeddings --db-dir ./face_database

  # Import extracted faces
  python manage_face_database.py import-extracted --db-dir ./face_database \\
      --faces-dir ./extracted_faces
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Common arguments
    common_group = parser.add_argument_group('common')
    common_group.add_argument(
        '--db-dir',
        type=str,
        default='./face_database',
        help='Face database directory (default: ./face_database)'
    )

    # Add person command
    add_parser = subparsers.add_parser('add-person', help='Add a new person to database')
    add_parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Person\'s name'
    )
    add_parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        required=True,
        help='Image file paths (can use wildcards)'
    )
    add_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing person directory'
    )

    # Remove person command
    remove_parser = subparsers.add_parser('remove-person', help='Remove a person from database')
    remove_parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Person\'s name to remove'
    )
    remove_parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm deletion'
    )

    # List command
    subparsers.add_parser('list', help='List all people in database')

    # Stats command
    subparsers.add_parser('stats', help='Show database statistics')

    # Generate embeddings command
    embed_parser = subparsers.add_parser('generate-embeddings', help='Generate face embeddings')
    embed_parser.add_argument(
        '--model',
        type=str,
        default='Facenet512',
        choices=['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace',
                'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace'],
        help='Face recognition model (default: Facenet512)'
    )

    # Import extracted faces command
    import_parser = subparsers.add_parser('import-extracted', help='Import extracted faces')
    import_parser.add_argument(
        '--faces-dir',
        type=str,
        required=True,
        help='Directory containing extracted faces'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        manager = FaceDatabaseManager(Path(args.db_dir))

        if args.command == 'add-person':
            # Expand image paths
            import glob
            image_paths = []
            for pattern in args.images:
                image_paths.extend(glob.glob(pattern))

            result = manager.add_person(args.name, image_paths, args.overwrite)
            print(f"\n{result['message']}")
            if result.get('added'):
                print(f"✅ Added {result['added']} images")
            if result.get('failed'):
                print(f"⚠️  Failed: {result['failed']} images")

        elif args.command == 'remove-person':
            result = manager.remove_person(args.name, args.confirm)
            print(f"\n{result['message']}")

        elif args.command == 'list':
            people_data = manager.list_people()
            print(f"\n📋 PEOPLE IN DATABASE ({people_data['total_people']} total):")
            for person, count in sorted(people_data['people'].items()):
                print(f"  • {person:30} ({count} images)")

        elif args.command == 'stats':
            stats = manager.get_statistics()
            manager.print_statistics(stats)

        elif args.command == 'generate-embeddings':
            result = manager.generate_embeddings(args.model)
            print(f"\n✅ {result['status'].upper()}")
            print(f"Model: {result['model']}")
            print(f"People processed: {result['people_processed']}")
            if result['errors']:
                print(f"Errors: {len(result['errors'])}")

        elif args.command == 'import-extracted':
            result = manager.import_extracted_faces(Path(args.faces_dir))
            print(f"\n{result['message']}")
            if result.get('imported'):
                print(f"✅ Imported {result['imported']} faces")

        return 0

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

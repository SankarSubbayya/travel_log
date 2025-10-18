#!/bin/bash
# Project Cleanup Script

echo "=========================================="
echo "Cleaning up Travel Log Project"
echo "=========================================="

# Create a cleanup directory for backup
mkdir -p .cleanup_backup
echo "✓ Created backup directory: .cleanup_backup/"

# Files to DELETE (temporary test files)
TEST_FILES=(
    "test_basic.py"
    "test_basic_working.py"
    "test_debug.py"
    "test_deepface.py"
    "test_deepface_smart.py"
    "test_download.py"
    "test_fast.py"
    "test_local_verification.py"
    "test_minimal.py"
    "test_mock.py"
    "test_network.py"
    "test_no_deepface.py"
    "test_opencv_only.py"
    "test_simple.py"
    "test_with_deepface.py"
    "test_without_faces.py"
    "pre_download_cpu.py"
    "pre_download_models.py"
    "download_models.py"
    "safe_download_models.py"
    "check_deepface_status.py"
    "check_download_size.py"
    "travel_log_lite.py"
)

echo ""
echo "Removing temporary test files..."
for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" .cleanup_backup/
        echo "  ✓ Removed: $file"
    fi
done

# Remove temporary markdown files
TEMP_DOCS=(
    "DOWNLOAD_STATUS.md"
    "FIXING_MUTEX_CRASH.md"
)

echo ""
echo "Removing outdated documentation..."
for file in "${TEMP_DOCS[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" .cleanup_backup/
        echo "  ✓ Removed: $file"
    fi
done

# Remove nohup.out if exists
if [ -f "nohup.out" ]; then
    mv nohup.out .cleanup_backup/
    echo "  ✓ Removed: nohup.out"
fi

# Keep these essential files:
KEEP_FILES=(
    "demo_face_detection.py"
    "verify_installation.py"
    "test_installation.py"
    "simple_download.py"
    "check_deepface_models.py"
    "template_script.py"
    "setup_suppress_warnings.py"
)

echo ""
echo "=========================================="
echo "Keeping essential files:"
echo "=========================================="
for file in "${KEEP_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    fi
done

echo ""
echo "=========================================="
echo "✅ Cleanup complete!"
echo "=========================================="
echo ""
echo "Kept files:"
echo "  - Core modules: src/travel_log/"
echo "  - Examples: examples/"
echo "  - Documentation: docs/"
echo "  - Utilities: demo_face_detection.py, verify_installation.py, etc."
echo ""
echo "Removed files backed up to: .cleanup_backup/"
echo ""
echo "To restore a file:"
echo "  mv .cleanup_backup/filename.py ."
echo ""
echo "To permanently delete backup:"
echo "  rm -rf .cleanup_backup/"
echo ""


#!/usr/bin/env python3
"""
Check DeepFace model download status and locations.

This script shows:
1. Where DeepFace downloads models
2. Which models are already downloaded
3. How to pre-download models
"""

import os
from pathlib import Path

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DEEPFACE MODEL DOWNLOAD STATUS")
print("="*70)

# Check DeepFace weights directory
home = Path.home()
deepface_dir = home / ".deepface" / "weights"

print(f"\n📂 DeepFace models location:")
print(f"   {deepface_dir}")

if deepface_dir.exists():
    print(f"\n✅ Directory exists")
    
    # List all model files
    models = []
    for ext in ['*.h5', '*.pb', '*.pth', '*.pkl', '*.npy']:
        models.extend(deepface_dir.glob(ext))
    
    if models:
        print(f"\n📦 Downloaded models ({len(models)} files):")
        total_size = 0
        for model in sorted(models):
            size_mb = model.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"   - {model.name}")
            print(f"     Size: {size_mb:.1f} MB")
        
        print(f"\n💾 Total size: {total_size:.1f} MB")
    else:
        print(f"\n⚠️  No models downloaded yet")
        print(f"   Models will download on first use")
else:
    print(f"\n⚠️  Directory doesn't exist yet")
    print(f"   Will be created on first DeepFace use")

# Show what will be downloaded
print("\n" + "="*70)
print("MODELS THAT WILL DOWNLOAD ON FIRST USE")
print("="*70)

print("\n📥 Detection Backends:")
print("   • opencv: Built-in (no download)")
print("   • ssd: ~2 MB")
print("   • mtcnn: ~2 MB")
print("   • retinaface: ~1 MB")
print("   • dlib: ~99 MB")

print("\n📥 Recognition Models:")
print("   • VGG-Face: ~500 MB")
print("   • Facenet: ~90 MB")
print("   • Facenet512: ~90 MB")
print("   • OpenFace: ~30 MB")
print("   • DeepFace: ~150 MB")
print("   • DeepID: ~17 MB")
print("   • ArcFace: ~160 MB")
print("   • Dlib: ~99 MB")
print("   • SFace: ~37 MB")

print("\n💡 Note: Only downloads what you actually use!")

# Check for nohup output
print("\n" + "="*70)
print("CHECKING BACKGROUND PROCESS")
print("="*70)

nohup_file = Path("nohup.out")
if nohup_file.exists():
    size = nohup_file.stat().st_size
    print(f"\n✅ Found nohup.out ({size} bytes)")
    
    # Show last few lines
    with open(nohup_file, 'r') as f:
        lines = f.readlines()
        if lines:
            print(f"\n📄 Last 10 lines of nohup.out:")
            for line in lines[-10:]:
                print(f"   {line.rstrip()}")
        else:
            print("\n📄 nohup.out is empty (process may still be running)")
else:
    print("\n⚠️  No nohup.out file found")
    print("   Your background process may not have started yet")

# Check for running background processes
print("\n" + "="*70)
print("TIPS")
print("="*70)

print("\n🔍 To check if your background process is still running:")
print("   ps aux | grep deepface")

print("\n📥 To pre-download models manually:")
print("   python pre_download_models.py")

print("\n📂 To see downloaded models:")
print(f"   ls -lh {deepface_dir}")

print("\n🧹 To clear downloaded models (if needed):")
print(f"   rm -rf {deepface_dir}")
print("   (They'll re-download on next use)")

print("\n" + "="*70)


# Travel Log Documentation

Complete guide to using the Travel Log face recognition system.

## 🚀 Quick Start

**New to Travel Log?** Start here:
- **[Quick Start Guide](FACE_RECOGNITION_QUICKSTART.md)** - Get started in 5 minutes
- **[Testing Guide](TESTING_GUIDE.md)** - Verify your installation works

## 📚 Core Documentation

### User Guides
- **[Streamlit Web App](STREAMLIT_APP.md)** - Interactive web interface 🌐
- **[HEIC Image Support](HEIC_SUPPORT.md)** - Use Apple HEIC/HEIF photos 📱
- **[Configuration Guide](CONFIGURATION.md)** - Configure paths and settings
- **[Project Overview](PROJECT_OVERVIEW.md)** - Complete project structure and features
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details

### Troubleshooting & Reference
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Fix common app errors
- **[Network Access Guide](NETWORK_ACCESS.md)** - Configure network access
- **[TensorFlow Warning Fix](TENSORFLOW_WARNING_FIX.md)** - Fix TensorFlow mutex warnings
- **[DeepFace Downloads](DEEPFACE_DOWNLOADS.md)** - Understanding model downloads
- **[DeepFace Alternatives](DEEPFACE_ALTERNATIVES.md)** - Alternative face recognition libraries

### Technical Documentation
- **[Comprehensive Face Recognition Guide](../docs/face-recognition-guide.md)** - Full detailed technical guide

## 📖 Documentation by Topic

### Getting Started
1. Read the [Quick Start Guide](FACE_RECOGNITION_QUICKSTART.md)
2. Run `uv run python test_installation.py`
3. Try `uv run python demo_face_detection.py your_photo.jpg`
4. Explore [examples](../examples/)

### Face Detection
- Basic detection: [Quick Start Guide](FACE_RECOGNITION_QUICKSTART.md)
- Advanced features: [Comprehensive Guide](../docs/face-recognition-guide.md)
- Example code: [face_detection_example.py](../examples/face_detection_example.py)

### Face Recognition & Labeling
- Overview: [Project Overview](PROJECT_OVERVIEW.md)
- Detailed guide: [Comprehensive Guide](../docs/face-recognition-guide.md)
- Example code: [face_labeling_example.py](../examples/face_labeling_example.py)

### Face Embeddings
- Technical details: [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- Full guide: [Comprehensive Guide](../docs/face-recognition-guide.md)
- Example code: [face_embeddings_example.py](../examples/face_embeddings_example.py)

### Troubleshooting
- **App errors?** → [Troubleshooting Guide](TROUBLESHOOTING.md)
- **Network issues?** → [Network Access Guide](NETWORK_ACCESS.md)
- **TensorFlow warnings?** → [TensorFlow Warning Fix](TENSORFLOW_WARNING_FIX.md)
- **Models not downloading?** → [DeepFace Downloads](DEEPFACE_DOWNLOADS.md)
- **Installation issues?** → [Testing Guide](TESTING_GUIDE.md)
- **DeepFace crashing?** → [Alternatives](DEEPFACE_ALTERNATIVES.md)

## 🎯 Common Questions

### "I want a visual interface"
→ [Streamlit Web App](STREAMLIT_APP.md) - No coding required!

### "Port 8501 is already in use"
→ [Troubleshooting Guide](TROUBLESHOOTING.md#-error-port-8501-is-already-in-use)

### "Can't reach external URL"
→ [Network Access Guide](NETWORK_ACCESS.md#the-problem-with-external-url)

### "I just want to get started"
→ [Quick Start Guide](FACE_RECOGNITION_QUICKSTART.md)

### "How do I configure paths and settings?"
→ [Configuration Guide](CONFIGURATION.md)

### "How do I test if it works?"
→ [Testing Guide](TESTING_GUIDE.md)

### "I see TensorFlow warnings"
→ [TensorFlow Warning Fix](TENSORFLOW_WARNING_FIX.md)

### "Where are models downloaded?"
→ [DeepFace Downloads](DEEPFACE_DOWNLOADS.md)

### "I want to understand everything"
→ [Comprehensive Guide](../docs/face-recognition-guide.md)

## 💡 Quick Links

- **Main README**: [../README.md](../README.md)
- **Examples Directory**: [../examples/](../examples/)
- **Source Code**: [../src/travel_log/](../src/travel_log/)
- **API Documentation**: [../docs/](../docs/)

## 📂 File Index

```
documentation/
├── README.md                        # This file - documentation index
│
├── User Guides
│   ├── STREAMLIT_APP.md            # Web app guide 🌐
│   ├── CONFIGURATION.md            # Configuration guide
│   ├── FACE_RECOGNITION_QUICKSTART.md  # 5-minute quick start
│   ├── PROJECT_OVERVIEW.md         # Complete project overview
│   ├── IMPLEMENTATION_SUMMARY.md   # Technical implementation
│   └── TESTING_GUIDE.md            # Testing guide
│
└── Troubleshooting & Reference
    ├── TROUBLESHOOTING.md          # Fix common app errors
    ├── NETWORK_ACCESS.md           # Network access guide
    ├── TENSORFLOW_WARNING_FIX.md   # Fix TensorFlow warnings
    ├── DEEPFACE_DOWNLOADS.md       # Model downloads
    └── DEEPFACE_ALTERNATIVES.md    # Alternative libraries
```

## 📞 Support

For questions during your weekly meetings with Chander and Asif, reference these docs!

---

**Quick Test Command:**
```bash
cd /home/sankar/travel_log
uv run python test_installation.py
```

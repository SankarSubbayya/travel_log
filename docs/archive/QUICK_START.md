# Quick Start: Ollama Caption Generation

## TL;DR - Get Started in 5 Minutes

### Prerequisites
- GPU with 6GB+ VRAM (NVIDIA/AMD)
- Terminal access
- Python 3.12+

### Installation

**1. Install Ollama** (2 minutes)
```bash
# Download from https://ollama.ai or run:
curl -fsSL https://ollama.ai/install.sh | sh
```

**2. Pull LLaVA Model** (5-10 minutes)
```bash
ollama pull llava:7b
```

**3. Start Ollama Service**
```bash
# In one terminal, keep running:
ollama serve
```

**4. Update Travel Log** (in new terminal)
```bash
cd /path/to/travel_log
uv sync
```

**5. Run the App**
```bash
streamlit run app.py
```

**6. Use Caption Generation**
- Open http://localhost:8501
- Go to "Captions" tab
- Click "🚀 Load Caption Generator"
- Upload image
- Click "✨ Generate Captions"
- Wait 30-60 seconds

## What You Get

Now your image captions actually **analyze your photos**! 🎉

### Example:

**Old** (no vision):
```
"A beautiful beach scene with interesting details"
```

**New** (actual image analysis):
```
"The Golden Gate Bridge stands majestically against a brilliant orange sunset,
with fog rolling through the bay. Sailboats dot the water, and the iconic
red-orange suspension cables catch the warm evening light perfectly."
```

## Quick Usage

### Web Interface (Easiest)
1. Open http://localhost:8501
2. Captions tab → Upload → Generate → Done

### Python Script
```python
from travel_log import CaptionGenerator
from PIL import Image

gen = CaptionGenerator()
img = Image.open("photo.jpg")
captions = gen.generate_all(img)
```

## Performance

- ⏱️ First: 30-60 sec (loads model to GPU)
- ⏱️ After: 10-30 sec each
- 📸 Formats: JPG, PNG, HEIC, BMP, GIF

## Help!

### "Cannot connect to Ollama"
→ Run `ollama serve` in another terminal

### "Model not found"
→ Run `ollama pull llava:7b`

### "Out of memory"
→ Close apps, need 8GB+ VRAM

### Takes too long
→ Normal! First caption is slowest.

## Learn More

- Full guide: [OLLAMA_INTEGRATION_GUIDE.md](OLLAMA_INTEGRATION_GUIDE.md)
- What changed: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- Examples: `python examples/caption_generation_example.py`

---

**Ready?** Start with step 1 above! ✅

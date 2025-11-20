# Caption Generator Guide

## Important Update

The caption generator implementation has been improved with **lazy loading** to work better with Streamlit.

## How It Works

### Initialization (Fast - No GPU)
```python
gen = CaptionGenerator()
```
- **Time:** <1 second
- **Memory:** Minimal
- Model is **NOT** loaded yet

### First Caption Generation (Slow - Loads GPU)
```python
caption = gen.generate_caption(image)
```
- **First run:** 1-2 minutes (downloads and loads 4GB model into VRAM)
- **Subsequent runs:** 10-30 seconds (model stays in memory)

## Using in Streamlit App

### Step 1: Initialize the Generator
1. Go to **Image Captions** tab
2. Click **"🚀 Load Caption Generator"** button
   - You'll see: "✅ Caption generator ready! Model will load when you generate first caption."
   - This takes <1 second

### Step 2: Upload Image
1. Upload your photo (JPEG, PNG, HEIC, etc.)

### Step 3: Generate Captions
1. Click **"✨ Generate Captions"** button
2. First time: You'll see progress
   ```
   ⏳ Generating captions...
   
   First run: Loading model into GPU (1-2 min)
   Subsequent runs: ~10-30 seconds
   ```
3. Wait 1-2 minutes while model loads
4. Once done: Captions appear below

## Important Notes

### ⚠️ Do NOT Close the App During Caption Generation
- The model stays loaded in GPU memory
- Reloading the app = reloading the model = 1-2 minute wait again
- Keep the app running between captions for fast generation

### GPU Memory
- First caption: Uses ~4-6GB VRAM
- Subsequent captions: Same ~4-6GB (model stays loaded)
- Your 24GB GPU has plenty of room

### Model Details
- **Model:** Llava v1.5 7B (Quantized 4-bit)
- **Size:** ~4GB download
- **Cache:** `~/.cache/huggingface/hub/`
- **Speed:** 10-30 seconds per caption (after loading)

## Troubleshooting

### "Error loading model" or "CUDA error"
1. Check GPU has free memory: `nvidia-smi`
2. Close other GPU applications
3. Restart the app and try again

### "Caption generation timed out"
1. Model took >2 minutes to load (rare)
2. Restart the app
3. Try with a smaller image

### Model downloads slowly
1. Check internet connection
2. Model is 4GB (~1-3 minutes typical)
3. Downloads to: `~/.cache/huggingface/hub/`

## Current Limitations

⚠️ **Important:** The current implementation generates **plausible text** based on patterns, not actual image analysis. This is a limitation of the text-only setup.

**Examples of what you get:**
- Generic beach descriptions
- Common travel scenarios
- Pattern-based text

**Why:** The model doesn't actually "see" your images - it generates text based on the prompt structure.

## Better Solutions

### Option 1: Use Ollama (Recommended)
For true image understanding:
```bash
# Install from https://ollama.ai
ollama run llava:7b
# Then use: http://localhost:11434
```

### Option 2: Use Transformers Pipeline
For direct multimodal support (slower but accurate):
```bash
pip install transformers[torch]
```

### Option 3: Keep Current Setup
- Works fine for face detection/identification
- Disable captions if not needed
- Focus on face features which actually work well

## Summary

| Task | Status |
|------|--------|
| Face Detection | ✅ Works great |
| Face Identification | ✅ Works great |
| Image Captions | ⚠️ Text-only (not image-aware) |
| Performance | ✅ Fast after first load |

**Face features (detection/identification) are fully functional and use actual image analysis.**

**Caption generation is structure-complete but text-only due to llama-cpp-python limitations.**

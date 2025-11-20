# Ollama Integration Guide - Caption Generation with LLaVA

## Overview

The Travel Log application now uses **Ollama's LLaVA model** for true multimodal image analysis and caption generation. This replaces the previous text-only approach with actual computer vision capabilities that analyze the content of your travel photos.

## Key Features

✅ **True Image Analysis**: LLaVA actually analyzes image content (not just generating generic text patterns)
✅ **Multimodal Processing**: Combines vision and language understanding
✅ **Fast Local Processing**: Runs on your machine with GPU support
✅ **Easy Setup**: Simple Ollama installation and model pulling
✅ **Three Caption Types**:
  - **Title**: Short descriptive title (2-4 words)
  - **Caption**: Detailed description of the image
  - **Travel Caption**: Travel-specific description with location/activity details

## Prerequisites

### System Requirements
- **GPU**: NVIDIA or AMD GPU with 6GB+ VRAM (16GB+ recommended for best performance)
- **Memory**: 16GB+ system RAM
- **Storage**: 6GB+ free space for LLaVA model
- **OS**: Linux, macOS, or Windows

### Python Version
- Python 3.12+

## Installation & Setup

### Step 1: Install Ollama

Visit [https://ollama.ai](https://ollama.ai) and download Ollama for your OS.

**For Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**For Windows:**
Download the installer from https://ollama.ai/download

### Step 2: Pull the LLaVA Model

```bash
ollama pull llava:7b
```

This downloads the 7B parameter quantized LLaVA model (~4.2 GB).

**Optional**: For better quality (slower), try the larger model:
```bash
ollama pull llava:13b  # ~8 GB, slower but higher quality
```

### Step 3: Start Ollama Service

```bash
ollama serve
```

This starts the Ollama API server on `http://localhost:11434`. Keep this terminal running while using the app.

### Step 4: Update Dependencies

In the travel_log directory:
```bash
uv sync
```

This installs the updated dependencies including the `requests` library for Ollama API calls.

### Step 5: Run the Application

In a new terminal (keeping Ollama running):
```bash
streamlit run app.py
```

## Using Caption Generation

### Via Streamlit Web Interface

1. Navigate to the **Captions** tab in the web app
2. Click **"🚀 Load Caption Generator"** button
3. Upload an image (JPG, PNG, HEIC, etc.)
4. Choose caption type:
   - **All (caption, title, travel)**: Generate all three types
   - **Title only**: Just the short title
   - **Detailed caption**: Full description
   - **Travel caption only**: Travel-focused description
5. Click **"✨ Generate Captions"**
6. Wait 30-60 seconds for analysis and caption generation
7. Copy captions to clipboard if desired

### Via Python Script

```python
from src.travel_log import CaptionGenerator
from PIL import Image

# Initialize with Ollama
generator = CaptionGenerator()

# Load an image
image = Image.open("path/to/travel/photo.jpg")

# Generate all captions
captions = generator.generate_all(image)

print(f"Title: {captions['title']}")
print(f"Caption: {captions['caption']}")
print(f"Travel Caption: {captions['travel_caption']}")
```

### Individual Caption Methods

```python
# Generate just a title
title = generator.generate_title(image)

# Generate detailed caption
caption = generator.generate_caption(image)

# Generate travel-specific caption
travel_caption = generator.generate_travel_caption(image)
```

### Custom Parameters

```python
# Adjust temperature (0.3=deterministic, 1.0=creative)
caption = generator.generate_caption(
    image,
    max_tokens=120,
    temperature=0.7
)

# Adjust token limits
title = generator.generate_title(
    image,
    max_tokens=8,
    temperature=0.5
)
```

## Configuration

### Default Settings

The `CaptionGenerator` class uses these defaults:
- **Ollama URL**: `http://localhost:11434` (change with `ollama_base_url` parameter)
- **Model**: `llava:7b` (change with `model_name` parameter)
- **Timeout**: 120 seconds (change with `timeout` parameter)

### Custom Configuration

```python
generator = CaptionGenerator(
    ollama_base_url="http://localhost:11434",
    model_name="llava:7b",
    timeout=180  # Increase timeout for slower GPUs
)
```

## Troubleshooting

### Problem: "Cannot connect to Ollama"

**Solution**:
1. Ensure Ollama is running: `ollama serve` in terminal
2. Check Ollama is on expected port: `http://localhost:11434/api/tags`
3. If custom port/URL, pass to `CaptionGenerator(ollama_base_url="...")`

### Problem: Request times out

**Solution**:
1. Increase timeout: `CaptionGenerator(timeout=300)`
2. Ensure GPU has enough memory
3. Try reducing image size before processing
4. Check if Ollama process is using GPU: `watch nvidia-smi`

### Problem: Out of memory errors

**Solution**:
1. Ensure sufficient VRAM (16GB+ recommended)
2. Close other GPU-intensive applications
3. Use smaller model if needed (though 7B is already quantized)
4. Increase swap space if on Linux

### Problem: LLaVA model not found

**Solution**:
```bash
# List available models
ollama list

# If llava:7b not listed, pull it
ollama pull llava:7b

# Try alternative model
ollama pull llava:13b
```

### Problem: Slow caption generation

**Solution**:
1. First run is slower (30-60s) as models are loaded into VRAM
2. Subsequent runs are faster (~10-30s)
3. Reduce image size for faster processing
4. Use GPU instead of CPU: check with `nvidia-smi`

## Performance Tips

1. **First Run**: Model loading takes 1-2 minutes. Subsequent runs are much faster.

2. **Image Size**: Larger images take longer to process
   - Recommended: 800x600 or smaller
   - Max processed: ~1024x1024

3. **GPU Utilization**: Monitor with:
   ```bash
   # NVIDIA
   nvidia-smi -l 1

   # AMD (if supported)
   rocm-smi --watch
   ```

4. **Temperature Parameter**:
   - `0.3` = Deterministic, factual descriptions
   - `0.7` = Balanced (default)
   - `1.0` = Creative, varied descriptions

## Example Scripts

### Run the example script

```bash
python examples/caption_generation_example.py
```

This demonstrates:
1. Single image caption generation
2. Batch processing multiple images
3. Integration with face detection
4. Custom temperature variations

## Advanced Usage

### Batch Processing

```python
from pathlib import Path
from travel_log import CaptionGenerator
from PIL import Image

generator = CaptionGenerator()
photos_dir = Path("path/to/photos")

for image_path in photos_dir.glob("*.jpg"):
    image = Image.open(image_path)
    captions = generator.generate_all(image)
    print(f"{image_path.name}: {captions['title']}")
```

### Integration with Face Recognition

```python
from travel_log import TravelLogFaceManager

manager = TravelLogFaceManager(
    workspace_dir="./workspace",
    enable_caption_generator=True
)

result = manager.process_photo(
    "photo.jpg",
    extract_faces=True,
    generate_captions=True
)

print(result['captions'])
```

## Model Information

### LLaVA 7B Quantized (Q4_K_M)
- **Size**: ~4.2 GB
- **Parameters**: 7 Billion
- **Quantization**: 4-bit (Q4_K_M)
- **Speed**: ~20-40 tokens/second on good GPU
- **Quality**: Good balance of quality and speed
- **VRAM**: 6-8 GB during inference

### LLaVA 13B Quantized (Q4_K_M) - Optional
- **Size**: ~8 GB
- **Parameters**: 13 Billion
- **Quantization**: 4-bit (Q4_K_M)
- **Speed**: ~10-20 tokens/second on good GPU
- **Quality**: Better quality, slower
- **VRAM**: 10-12 GB during inference

## Architecture

The implementation uses:
1. **PIL (Pillow)**: Image handling and preprocessing
2. **requests**: HTTP API calls to Ollama
3. **base64**: Image encoding for transmission
4. **Ollama API**: `/api/generate` endpoint with vision support

## API Reference

### CaptionGenerator Class

```python
class CaptionGenerator:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "llava:7b",
        timeout: int = 120,
    )

    def generate_caption(
        image: Image.Image,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str

    def generate_title(
        image: Image.Image,
        max_tokens: int = 15,
        temperature: float = 0.7,
    ) -> str

    def generate_travel_caption(
        image: Image.Image,
        max_tokens: int = 120,
        temperature: float = 0.7,
    ) -> str

    def generate_all(
        image: Image.Image,
        max_caption_tokens: int = 100,
        max_title_tokens: int = 15,
        temperature: float = 0.7,
    ) -> dict[str, str]
```

## Known Limitations

1. **Internet**: Ollama runs locally, no internet required
2. **Model Size**: LLaVA is not as capable as larger models like GPT-4V
3. **Speed**: Processing takes 30-60 seconds per image
4. **Memory**: Requires 6GB+ VRAM for smooth operation
5. **Accuracy**: Vision understanding limited by model size

## Future Improvements

Possible enhancements:
- Support for larger models (13B, 34B)
- Streaming responses for faster initial output
- Caching of captions
- Multi-image comparison
- Custom fine-tuning for travel-specific captions

## References

- Ollama: https://ollama.ai
- LLaVA: https://github.com/haotian-liu/LLaVA
- GGUF Format: https://github.com/ggerganov/ggml
- Quantization: https://huggingface.co/docs/transformers/quantization

## Support

For issues or questions:
1. Check troubleshooting section above
2. Verify Ollama is running and accessible
3. Check error messages in application
4. Review example scripts for usage patterns

---

**Last Updated**: November 2024
**Travel Log Version**: 0.1.0
**Ollama Integration**: Initial Release

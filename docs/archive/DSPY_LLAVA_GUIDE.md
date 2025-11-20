# DSPy + LLaVA Integration Guide

## Overview

This guide shows how to combine **DSPy** (optimization framework) with **LLaVA** (vision-language model) in Travel Log to create intelligent, context-aware photo captions.

## What is DSPy?

DSPy is a framework for **programming with language models**, not just prompting them. It allows you to:
- Define structured signatures for LM tasks
- Chain multiple reasoning steps
- Optimize prompts automatically
- Learn from examples and feedback

## Why Combine DSPy with LLaVA?

### LLaVA Alone:
- ✅ **Vision**: Sees and describes what's IN the image
- ❌ Limited context integration
- ❌ Single-purpose outputs
- ❌ No learning from feedback

### LLaVA + DSPy:
- ✅ **Vision + Reasoning**: Understands what the image MEANS
- ✅ Integrates face recognition, GPS, timestamps
- ✅ Multi-format outputs (social, personal, memory notes)
- ✅ Learns and optimizes from user feedback
- ✅ Adaptive prompts based on scene type

## Architecture

```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  LLaVA (Vision)     │  ← Ollama: llava:7b
│  "I see..."         │
└──────┬──────────────┘
       │ visual_description
       ▼
┌─────────────────────────────────┐
│  DSPy (Reasoning & Context)     │  ← Ollama: llama3
│  + Face names                   │
│  + GPS location                 │
│  + Timestamp                    │
│  + Nearby photos                │
│  → Intelligent captions         │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Rich Outputs       │
│  - Title            │
│  - Caption          │
│  - Social media     │
│  - Memory note      │
│  - Hashtags         │
└─────────────────────┘
```

## Prerequisites

### 1. Install DSPy

```bash
cd /home/sankar/travel_log
uv add dspy-ai

# Or with pip
pip install dspy-ai
```

### 2. Ensure Ollama Models

```bash
# Check installed models
ollama list

# Install if needed
ollama pull llava:7b    # For vision
ollama pull llama3      # For text reasoning
```

### 3. Start Ollama

```bash
ollama serve
```

## Usage

### Basic Example

```python
from travel_log.dspy_llava_integration import generate_smart_captions
from PIL import Image
from datetime import datetime

# Generate smart captions
captions = generate_smart_captions(
    image="vacation_photo.jpg",
    face_names=["Sarah", "John"],          # From face recognition
    location="Golden Gate Bridge, SF",      # From GPS/EXIF
    timestamp=datetime(2024, 11, 8, 15, 30) # From EXIF
)

print(f"Title: {captions['title']}")
print(f"Caption: {captions['caption']}")
print(f"Mood: {captions['mood']}")
print(f"Hashtags: {captions['hashtags']}")
```

### Advanced Example with Face Recognition

```python
from travel_log import FaceDetector, FaceLabeler
from travel_log.dspy_llava_integration import DSPyLLaVACaptionGenerator
from PIL import Image

# Initialize face recognition
detector = FaceDetector(detector_backend='mtcnn')
labeler = FaceLabeler(
    database_path='./face_database',
    model_name='Facenet512'
)

# Process photo
image_path = "group_photo.jpg"
image = Image.open(image_path)

# Detect and identify faces
faces = detector.extract_faces(image_path)
matches = labeler.find_face(image_path)

# Extract names
face_names = []
if matches and len(matches) > 0:
    for idx, match in matches[0].iterrows():
        person_name = Path(match['identity']).parent.name
        face_names.append(person_name)

# Generate contextual captions with DSPy + LLaVA
generator = DSPyLLaVACaptionGenerator()
result = generator.forward(
    image=image,
    face_names=face_names,
    location="Yosemite National Park",
    timestamp=datetime.now()
)

print(f"Personalized Caption: {result['caption']}")
# "Sarah and John enjoying the stunning views at Yosemite National Park
#  on a beautiful autumn afternoon"
```

### Scene-Adaptive Captions

```python
from travel_log.dspy_llava_integration import SmartTravelPhotoCaptioner

captioner = SmartTravelPhotoCaptioner()

result = captioner.forward(
    image=Image.open("sunset.jpg"),
    face_names=["Alex"],
    metadata={
        "location": "Malibu Beach",
        "time": "Evening"
    }
)

# Automatically adapts based on scene type
print(f"Scene: {result['type']}")        # "landscape"
print(f"Caption: {result['caption']}")   # Scenic description
print(f"Extra: {result['extra']}")       # Photography tip
```

### Batch Photo Analysis

```python
from travel_log.dspy_llava_integration import BatchPhotoAnalyzer

analyzer = BatchPhotoAnalyzer()

# Analyze trip photos
photos = [
    {"image": img1, "timestamp": "2024-11-01 10:00", "location": "Airport"},
    {"image": img2, "timestamp": "2024-11-01 14:00", "location": "Hotel"},
    {"image": img3, "timestamp": "2024-11-02 09:00", "location": "Beach"},
]

trip_analysis = analyzer.analyze_photo_collection(photos)

print(f"Events: {trip_analysis['event_names']}")
print(f"Story: {trip_analysis['narrative']}")
# "Our adventure began with an early morning flight..."
```

## Configuration

Add DSPy settings to your `config.yaml`:

```yaml
# DSPy + LLaVA Configuration
caption_generation:
  # Enable DSPy enhancement
  use_dspy: true

  # Ollama settings
  ollama_base_url: http://localhost:11434

  # Models
  vision_model: llava:7b      # For visual analysis
  reasoning_model: llama3     # For contextual reasoning

  # Caption styles
  default_style: balanced     # Options: casual, professional, poetic, humorous

  # Output formats
  generate_social_captions: true
  generate_memory_notes: true
  generate_hashtags: true
```

## DSPy Signatures

### Basic Caption Signature

```python
import dspy

class TravelCaptionSignature(dspy.Signature):
    """Generate travel photo captions."""

    visual_content = dspy.InputField(desc="what's in the image")
    people_names = dspy.InputField(desc="identified people")
    location = dspy.InputField(desc="where photo was taken")

    title = dspy.OutputField(desc="catchy 2-4 word title")
    caption = dspy.OutputField(desc="engaging 2-3 sentence caption")
    hashtags = dspy.OutputField(desc="relevant hashtags")
```

### Using Chain-of-Thought

```python
class SmartCaptioner(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(TravelCaptionSignature)

    def forward(self, visual, names, location):
        return self.generate(
            visual_content=visual,
            people_names=names,
            location=location
        )
```

## Optimization with Examples

### Train from Examples

```python
from dspy.teleprompt import BootstrapFewShot

# Define training examples
training_examples = [
    dspy.Example(
        visual_content="Beach sunset with couple",
        people_names="Sarah, John",
        location="Malibu",
        caption="Sarah and John enjoying a romantic sunset at Malibu Beach"
    ),
    # More examples...
]

# Define quality metric
def caption_quality(example, prediction, trace=None):
    score = 0.0
    # Check if names mentioned
    if all(name in prediction.caption for name in example.people_names.split(", ")):
        score += 0.5
    # Check if location mentioned
    if example.location in prediction.caption:
        score += 0.5
    return score

# Optimize
generator = SmartCaptioner()
optimizer = BootstrapFewShot(metric=caption_quality)
optimized_generator = optimizer.compile(generator, trainset=training_examples)
```

### Learn from User Feedback

```python
class FeedbackLearner:
    def __init__(self):
        self.generator = DSPyLLaVACaptionGenerator()
        self.feedback_examples = []

    def generate_and_learn(self, image, context):
        # Generate caption
        result = self.generator.forward(image=image, **context)

        # Get user feedback
        user_rating = get_user_rating()  # 1-5 stars
        user_edit = get_user_edit()      # User's edited version

        # Store as training example
        if user_rating >= 4:
            self.feedback_examples.append({
                "input": context,
                "good_output": user_edit
            })

        # Periodically retrain
        if len(self.feedback_examples) >= 10:
            self.retrain()

    def retrain(self):
        # Convert feedback to DSPy examples
        # Optimize with BootstrapFewShot
        # Update generator
        pass
```

## Integration with Streamlit App

Update [app.py](../app.py) to use DSPy:

```python
# In app.py
from travel_log.dspy_llava_integration import generate_smart_captions

# In the caption tab
if use_dspy:
    # Enhanced with DSPy
    captions = generate_smart_captions(
        image=image,
        face_names=identified_faces,
        location=metadata.get('location'),
        timestamp=metadata.get('datetime')
    )

    st.write(f"**Title:** {captions['title']}")
    st.write(f"**Caption:** {captions['caption']}")
    st.write(f"**Mood:** {captions['mood']}")
    st.write(f"**Hashtags:** {captions['hashtags']}")
else:
    # Basic LLaVA only
    basic_caption = caption_generator.generate_caption(image)
    st.write(f"**Caption:** {basic_caption}")
```

## Performance Comparison

### LLaVA Alone
```
Input: Beach photo
Output: "A beach scene with people and sunset"
Time: ~30 seconds
```

### LLaVA + DSPy
```
Input: Same beach photo + [names, location, time]
Output:
  Title: "Sunset Memories"
  Caption: "Sarah and John capturing the golden hour at Malibu Beach
            after a perfect day exploring the California coast"
  Mood: "Romantic and peaceful"
  Hashtags: "#MalibuSunset #CaliforniaLove #GoldenHour"
Time: ~35 seconds (5s overhead for reasoning)
```

**Worth it?** Yes! The 5-second overhead gives you:
- Personalized captions with names
- Location context
- Multiple output formats
- Emotional understanding
- Social media ready content

## Common Use Cases

### 1. Personal Photo Albums
Generate memory-focused captions that mention people by name and recall specific moments.

### 2. Social Media Posts
Create Instagram/Facebook ready captions with emojis and hashtags.

### 3. Travel Blogs
Generate narrative descriptions that connect photos into a story.

### 4. Photo Organization
Automatically categorize and tag photos based on content and context.

### 5. Smart Search
Create detailed descriptions that make photos searchable by natural language queries.

## Troubleshooting

### DSPy Not Working

```bash
# Check installation
python -c "import dspy; print(dspy.__version__)"

# Reinstall if needed
pip install --upgrade dspy-ai
```

### Ollama Connection Errors

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Check models
ollama list

# Restart Ollama
pkill ollama
ollama serve
```

### Slow Performance

1. **Use smaller models**:
   ```python
   # Instead of llama3:70b, use llama3:8b
   text_model = dspy.OllamaLocal(model="llama3:8b")
   ```

2. **Reduce reasoning steps**:
   ```python
   # Use Predict instead of ChainOfThought
   self.generator = dspy.Predict(signature)
   ```

3. **Cache results**:
   ```python
   import functools

   @functools.lru_cache(maxsize=100)
   def cached_caption(image_hash, ...):
       return generate_smart_captions(...)
   ```

## Example Scripts

Run the demo:

```bash
cd /home/sankar/travel_log
python examples/dspy_llava_demo.py
```

This demonstrates:
1. Basic caption generation
2. Comparison with LLaVA-only
3. Adaptive captions by scene type
4. Integration with face recognition

## Best Practices

### 1. Start Simple
Begin with basic DSPy signatures, then add complexity as needed.

### 2. Collect Examples
Save good caption examples to train/optimize your system.

### 3. Monitor Performance
Track generation time and quality metrics.

### 4. User Feedback
Let users rate and edit captions, use as training data.

### 5. Experiment with Prompts
DSPy helps optimize, but start with clear, specific prompts.

## Summary

DSPy + LLaVA gives you:

- ✅ **Vision understanding** (LLaVA sees the image)
- ✅ **Contextual reasoning** (DSPy understands what it means)
- ✅ **Personalization** (Names, locations, relationships)
- ✅ **Multiple outputs** (Social, personal, memory notes)
- ✅ **Optimization** (Learns from examples and feedback)
- ✅ **Adaptability** (Different styles for different scenes)

**Result:** Smarter, more useful photo captions that go beyond simple descriptions!

## References

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Travel Log OLLAMA Guide](./OLLAMA_INTEGRATION_GUIDE.md)

---

**Last Updated:** November 2024
**Version:** 0.1.0
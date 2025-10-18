# COMPLETE FIX: TensorFlow Mutex Warning

## The Warning

```
[mutex.cc : 452] RAW: Lock blocking 0x1140e7cb8
```

This TensorFlow mutex warning appears on macOS and is **completely harmless** but annoying.

## ✅ GUARANTEED FIX - Choose One Method

### Method 1: Use the Suppression Utility (EASIEST)

Import `setup_suppress_warnings` **BEFORE** importing travel_log:

```python
# YOUR SCRIPT
import setup_suppress_warnings  # Must be FIRST!
from travel_log import TravelLogFaceManager  # Now import

# Your code here - no warnings!
manager = TravelLogFaceManager("workspace")
```

### Method 2: Set Environment Variables at Script Start (MOST RELIABLE)

Put these lines at the **VERY TOP** of your script (before any imports):

```python
# Put at the VERY TOP - line 1!
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

# Now import warnings suppression
import warnings
warnings.filterwarnings('ignore')

# NOW you can import travel_log
from travel_log import TravelLogFaceManager
```

**Complete Example:**

```python
#!/usr/bin/env python3
"""Example with warnings suppressed."""

# STEP 1: Set env vars FIRST (must be before ANY imports)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# STEP 2: Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# STEP 3: Now import and use
from travel_log import TravelLogFaceManager

manager = TravelLogFaceManager("workspace")
print("✓ No warnings!")
```

### Method 3: Set Environment Variables in Terminal (ONE-TIME)

Before running your Python script:

```bash
# macOS/Linux
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
python your_script.py

# Or in one line:
TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0 python your_script.py
```

### Method 4: Set Permanently in Shell Profile

Add to `~/.zshrc` (macOS) or `~/.bashrc` (Linux):

```bash
# Add these lines
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_VLOG_LEVEL=3
```

Then reload:
```bash
source ~/.zshrc  # or source ~/.bashrc
```

### Method 5: In Jupyter Notebook

Put in the **first cell**:

```python
# First cell - run this first!
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

print("✓ Warnings suppressed")
```

Then in the next cell:
```python
# Second cell - now import
from travel_log import TravelLogFaceManager
```

## Why Is This Happening?

The warning appears because:

1. **TensorFlow** is used by DeepFace for neural network models
2. On **macOS**, TensorFlow's threading/mutex system logs verbose warnings
3. The warning is about **internal thread locking** - completely safe to ignore
4. It appears during **model initialization** (first time using DeepFace)

## What If It Still Appears?

If you're still seeing the warning after trying the methods above:

### Check 1: Verify Environment Variables Are Set

```python
import os
print("TF_CPP_MIN_LOG_LEVEL:", os.environ.get('TF_CPP_MIN_LOG_LEVEL', 'NOT SET'))
print("TF_ENABLE_ONEDNN_OPTS:", os.environ.get('TF_ENABLE_ONEDNN_OPTS', 'NOT SET'))
```

If it says "NOT SET", the environment variables aren't being set early enough.

### Check 2: Look for Early TensorFlow Imports

If another module imports TensorFlow before your suppression code, you'll still see warnings. 

**Solution**: Set environment variables in terminal before running:
```bash
TF_CPP_MIN_LOG_LEVEL=3 python your_script.py
```

### Check 3: Restart Python Kernel

If using Jupyter or IPython:
- Restart the kernel completely
- Run suppression code in first cell
- Then import travel_log

### Check 4: Check for Other TensorFlow Users

```bash
# See what's importing TensorFlow
grep -r "import tensorflow" .
grep -r "from tensorflow" .
```

## Is It Actually a Problem?

**NO!** This warning:
- ✅ Does NOT affect functionality
- ✅ Does NOT cause errors or crashes
- ✅ Does NOT slow down your code
- ✅ Does NOT affect accuracy
- ✅ Is ONLY verbose logging

You can safely ignore it if the fixes don't work for your specific setup.

## Test the Fix

Run the test script to verify:

```bash
cd /Users/sankar/sankar/courses/llm/travel_log
python examples/test_no_warnings.py
```

If no warnings appear, your fix worked! ✅

## Additional TensorFlow Configuration

For more control over TensorFlow behavior:

```python
import os

# Suppress all TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable oneDNN custom ops
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Force CPU mode (avoid GPU warnings)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Memory growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Additional verbose logging suppression
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
```

## The Nuclear Option: Redirect stderr

If ALL else fails, redirect stderr (captures ALL output):

```python
import sys
import os

# Redirect stderr to suppress mutex warnings
class SuppressOutput:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

# Use it
with SuppressOutput():
    from travel_log import TravelLogFaceManager

# Now use normally
manager = TravelLogFaceManager("workspace")
```

**Warning**: This suppresses ALL stderr output, including real errors!

## Platform-Specific Notes

### macOS (Your System)
- Most likely to see this warning
- TensorFlow + macOS + ARM/Intel = verbose logging
- Fix: Use Method 1 or Method 2 above

### Linux
- Less common
- Usually fixed by setting `TF_CPP_MIN_LOG_LEVEL=3`

### Windows
- Rarely occurs
- If it does, use Method 2

## Summary: The Golden Rule

🥇 **ALWAYS set environment variables BEFORE any TensorFlow import!**

```python
# Line 1-2 of your script
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Line 3-4
import warnings
warnings.filterwarnings('ignore')

# Now rest of imports
from travel_log import TravelLogFaceManager
```

## Quick Reference

| Problem | Solution |
|---------|----------|
| Warnings in script | Put env vars at top of script |
| Warnings in notebook | Put env vars in first cell |
| Warnings persist | Use `setup_suppress_warnings.py` |
| Still seeing it | Run with `TF_CPP_MIN_LOG_LEVEL=3 python script.py` |
| Nothing works | It's harmless - ignore it! |

## Need Help?

1. Try `examples/test_no_warnings.py`
2. Use `setup_suppress_warnings.py`
3. Ask Chander and Asif in your weekly meeting
4. Remember: **The warning is harmless!**

## Environment Variable Levels

```
TF_CPP_MIN_LOG_LEVEL values:
  0 = Show all (INFO, WARNING, ERROR, FATAL) - default
  1 = Hide INFO
  2 = Hide INFO and WARNING
  3 = Hide INFO, WARNING, and ERROR (recommended)
```

Use level **3** to hide the mutex warning.

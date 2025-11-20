# Streamlit App Troubleshooting Guide

## Common Errors and Solutions

### ❌ Error: "Port 8501 is already in use"

This happens when a Streamlit process is already running.

#### Quick Fix (Automatic)

The launch scripts now handle this automatically! Just run:

```bash
./run_app.sh
```

The script will detect and stop any existing processes.

#### Manual Fix

**Option 1: Use the kill script**
```bash
./kill_streamlit.sh
```

**Option 2: Kill manually**
```bash
# Find what's using port 8501
lsof -i :8501

# Kill the process
pkill -f "streamlit run"

# Verify it's stopped
lsof -i :8501
```

**Option 3: Use a different port**
```bash
uv run streamlit run app.py --server.port 8502
# Access at: http://localhost:8502
```

### ❌ Error: "Cannot reach external URL"

See `NETWORK_ACCESS.md` for complete details.

**Quick Solution:**
```bash
# Use localhost instead
http://localhost:8501

# NOT the external URL
# ❌ http://99.46.139.88:8501
```

### ❌ Error: "Address already in use"

Same as "Port in use" - another process is using port 8501.

**Solution:**
```bash
./kill_streamlit.sh
./run_app.sh
```

### ❌ Error: "Connection refused"

The app isn't running or crashed.

**Solutions:**

1. **Check if app is running:**
   ```bash
   ps aux | grep streamlit
   ```

2. **Check for errors in terminal where you started it**

3. **Restart the app:**
   ```bash
   ./kill_streamlit.sh
   ./run_app.sh
   ```

4. **Check the logs:**
   ```bash
   # If you see errors, they'll show in the terminal
   ```

### ❌ Error: "Can't register atexit after shutdown"

This is a harmless warning during shutdown. Ignore it - the app works fine!

**It appears when:**
- You stop the app with Ctrl+C
- The system is cleaning up threads

**No action needed** - it's just verbose logging during shutdown.

### ❌ Error: "Module 'tensorflow' has no attribute..."

TensorFlow compatibility issue.

**Solution:**
```bash
# Already fixed in pyproject.toml with tf-keras
uv sync

# Verify
uv run python -c "import tensorflow as tf; print(tf.__version__)"
```

### ❌ Error: "No module named 'streamlit'"

Streamlit not installed.

**Solution:**
```bash
uv sync
# Or
uv add streamlit
```

### ❌ Error: "Configuration file not found"

Running from wrong directory.

**Solution:**
```bash
# Always run from project root
cd /home/sankar/travel_log
./run_app.sh
```

### ⚠️ Warning: "You may see mutex.cc warnings"

These are harmless TensorFlow logging messages.

**No action needed** - face detection still works perfectly.

## Checking System Status

### Check if Streamlit is Running

```bash
# Check process
ps aux | grep streamlit

# Check port
lsof -i :8501

# Check if responding
curl http://localhost:8501
```

### Kill All Streamlit Processes

```bash
# Easy way
./kill_streamlit.sh

# Manual way
pkill -9 -f "streamlit run"

# Nuclear option (kills all Python)
# ⚠️ Only use if above doesn't work
pkill -9 python
```

### Check Port Availability

```bash
# See what's on port 8501
lsof -i :8501

# See all listening ports
ss -tuln

# Check if port is free
nc -zv localhost 8501
```

## Performance Issues

### App is Slow

**Causes:**
- First run (downloading models)
- CPU-only processing
- Large images
- Heavy backend (RetinaFace)

**Solutions:**

1. **Use faster backend:**
   - In app, select "opencv" or "ssd" backend
   
2. **Check GPU usage:**
   ```bash
   nvidia-smi
   ```

3. **Reduce image size before upload**

4. **Lower confidence threshold**

### App Crashes During Detection

**Possible causes:**
- Out of memory
- Image too large
- Model download interrupted

**Solutions:**

1. **Check available memory:**
   ```bash
   free -h
   ```

2. **Use lighter backend:**
   - Select "opencv" instead of "retinaface"

3. **Restart app:**
   ```bash
   ./kill_streamlit.sh
   ./run_app.sh
   ```

## Models Not Downloading

### Check Download Status

```bash
# See what models are downloaded
ls -lh ~/.deepface/weights/

# Check disk space
df -h ~/.deepface
```

### Fix Download Issues

```bash
# Check internet
ping github.com

# Remove corrupted downloads
rm -rf ~/.deepface/weights/

# Restart app (models will re-download)
./run_app.sh
```

## Network Access Issues

### Can't Access from Other Devices

**Checklist:**

1. ✅ **Using network script:**
   ```bash
   ./run_app_network.sh
   ```

2. ✅ **Same WiFi network:**
   - Check device is on same network as server

3. ✅ **Using correct URL:**
   ```bash
   # Use internal IP, not external
   http://192.168.0.140:8501
   ```

4. ✅ **Firewall open:**
   ```bash
   sudo ufw allow 8501/tcp
   ```

5. ✅ **App is running:**
   ```bash
   lsof -i :8501
   ```

### Test Network Connectivity

```bash
# From other device, try:
ping 192.168.0.140

# Check if port is reachable
nc -zv 192.168.0.140 8501
```

## Browser Issues

### Page Won't Load

1. **Clear browser cache:**
   - Ctrl+Shift+Delete
   - Clear cache and reload

2. **Try different browser:**
   - Chrome, Firefox, Edge

3. **Check console errors:**
   - F12 → Console tab

4. **Disable extensions:**
   - Try incognito/private mode

### Page Loads but App Doesn't Work

1. **Check browser console for errors** (F12)

2. **Try hard refresh:**
   - Ctrl+Shift+R (Linux/Windows)
   - Cmd+Shift+R (Mac)

3. **Restart the app:**
   ```bash
   ./kill_streamlit.sh
   ./run_app.sh
   ```

## Complete Reset

If nothing else works:

```bash
# 1. Kill everything
./kill_streamlit.sh

# 2. Clean Python cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# 3. Reinstall dependencies
uv sync

# 4. Clear Streamlit cache
rm -rf ~/.streamlit/cache

# 5. Start fresh
./run_app.sh
```

## Getting Help

### Collect Diagnostic Info

```bash
# System info
uname -a
python --version

# Package versions
uv pip list | grep -E "streamlit|tensorflow|deepface"

# Port status
lsof -i :8501

# Process status
ps aux | grep streamlit

# Recent logs
journalctl --user -u streamlit --since "1 hour ago"
```

### Common Command Reference

| Task | Command |
|------|---------|
| Start app (local) | `./run_app.sh` |
| Start app (network) | `./run_app_network.sh` |
| Stop app | `Ctrl+C` or `./kill_streamlit.sh` |
| Kill all Streamlit | `./kill_streamlit.sh` |
| Check if running | `lsof -i :8501` |
| Check processes | `ps aux \| grep streamlit` |
| View models | `ls -lh ~/.deepface/weights/` |
| Test connection | `curl http://localhost:8501` |
| Change port | `--server.port 8502` |
| Clear cache | `rm -rf ~/.streamlit/cache` |

## Prevention Tips

### Avoid "Port in Use" Errors

1. **Always stop with Ctrl+C** when done
2. **Use the launch scripts** (they auto-clean)
3. **Don't close terminal** without stopping app

### Avoid Network Issues

1. **Start with local access** for testing
2. **Use network access** only when needed
3. **Close firewall port** when done
4. **Document your network setup**

### Avoid Performance Issues

1. **Use appropriate backend** for your needs
2. **Start with opencv** for testing
3. **Upgrade to mtcnn** for accuracy
4. **Monitor GPU usage**
5. **Resize large images** before upload

## Quick Fixes Summary

```bash
# Port in use
./kill_streamlit.sh && ./run_app.sh

# Can't connect
# Use http://localhost:8501 (not external IP)

# App crashed
./kill_streamlit.sh
./run_app.sh

# Slow performance
# Change backend to 'opencv' in app

# Models not downloading
rm -rf ~/.deepface/weights/
./run_app.sh

# Complete reset
./kill_streamlit.sh
uv sync
rm -rf ~/.streamlit/cache
./run_app.sh
```

## Still Having Issues?

1. Check the error message carefully
2. Look in this troubleshooting guide
3. Check `NETWORK_ACCESS.md` for network issues
4. Review terminal output for specific errors
5. Try the complete reset procedure above

Most issues are solved by:
- Using the right URL (localhost:8501)
- Killing old processes
- Running from correct directory
- Using the provided launch scripts

The app is working correctly - just needs the right commands! 🚀


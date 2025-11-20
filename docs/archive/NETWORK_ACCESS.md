# Network Access Guide for Streamlit App

## Understanding the URLs

When you start the Streamlit app, you'll see three URLs:

1. **Local URL**: `http://localhost:8501` ✅ **Use this one!**
2. **Network URL**: `http://192.168.0.140:8501` (Local network only)
3. **External URL**: `http://99.46.139.88:8501` ❌ **Won't work by default**

## The Problem with External URL

The external URL (`http://99.46.139.88:8501`) doesn't work because:

1. **Firewall**: Your system/router blocks incoming connections on port 8501
2. **NAT/Router**: Your router doesn't forward external traffic to port 8501
3. **Security**: By default, Streamlit only binds to localhost for security
4. **ISP**: Many ISPs block incoming connections on non-standard ports

**This is intentional for security!** You don't want random internet users accessing your app.

## Solutions

### ✅ Solution 1: Access Locally (Recommended)

If you're on the same machine, use:
```bash
# Start the app
./run_app.sh

# Access in your browser
http://localhost:8501
```

### ✅ Solution 2: Local Network Access

If you want to access from other devices on your **local network** (e.g., phone, tablet):

```bash
# Start with network access
./run_app_network.sh

# From other devices on same WiFi, use
http://192.168.0.140:8501
```

**Note**: You may need to open the firewall:
```bash
sudo ufw allow 8501/tcp
```

### ✅ Solution 3: SSH Tunnel (Secure Remote Access)

If you need to access from a remote location securely:

**From your remote machine:**
```bash
# Create SSH tunnel
ssh -L 8501:localhost:8501 sankar@99.46.139.88

# Then access in your browser
http://localhost:8501
```

This creates a secure tunnel through SSH.

### ⚠️ Solution 4: Public Access (Not Recommended)

If you really need public internet access (use with caution):

1. **Open firewall:**
   ```bash
   sudo ufw allow 8501/tcp
   ```

2. **Configure router:**
   - Log into your router admin panel
   - Set up port forwarding: External port 8501 → 192.168.0.140:8501

3. **Start app with network binding:**
   ```bash
   ./run_app_network.sh
   ```

**Security Warnings:**
- Anyone on the internet could access your app
- No authentication/password protection
- Could expose your file system
- Use only temporarily and behind a VPN/firewall

### ✅ Solution 5: Cloud Deployment (Best for Production)

For real production access, deploy to:

- **Streamlit Cloud** (free tier available)
- **Heroku**
- **AWS/GCP/Azure**
- **Docker + reverse proxy (nginx)**

## Quick Reference

| Access Type | Command | URL | Security |
|-------------|---------|-----|----------|
| **Local only** | `./run_app.sh` | `http://localhost:8501` | ✅ Secure |
| **Local network** | `./run_app_network.sh` | `http://192.168.0.140:8501` | ⚠️ Trusted network only |
| **SSH Tunnel** | `ssh -L 8501:localhost:8501 ...` | `http://localhost:8501` | ✅ Secure |
| **Public** | Not provided | `http://99.46.139.88:8501` | ❌ Not recommended |

## Checking Firewall Status

### Ubuntu/Debian (UFW)
```bash
# Check status
sudo ufw status

# Allow port 8501 (if needed)
sudo ufw allow 8501/tcp

# Remove rule later
sudo ufw delete allow 8501/tcp
```

### Check if Port is Open
```bash
# Check if Streamlit is listening
ss -tuln | grep 8501

# Test connection from same machine
curl http://localhost:8501

# Test from another machine on network
curl http://192.168.0.140:8501
```

## Troubleshooting

### "Cannot be reached" error

**For external URL (99.46.139.88:8501):**
- ✅ **Expected behavior** - this is blocked for security
- ✅ **Use localhost instead**: `http://localhost:8501`
- ✅ **Or use SSH tunnel** for remote access

**For network URL (192.168.0.140:8501):**
- Check firewall: `sudo ufw status`
- Open port: `sudo ufw allow 8501/tcp`
- Restart app with: `./run_app_network.sh`
- Check same WiFi network

**For local URL (localhost:8501):**
- Check if app is running: `ss -tuln | grep 8501`
- Try 127.0.0.1:8501 instead
- Restart the app

### App shows "Stopping..." immediately

- Old instance may still be running
- Kill it: `pkill -f streamlit`
- Start fresh: `./run_app.sh`

### "Address already in use"

```bash
# Find what's using port 8501
sudo lsof -i :8501

# Kill the process
kill <PID>

# Or use different port
uv run streamlit run app.py --server.port 8502
```

## Recommended Setup

### For Development (Your Scenario)
```bash
# Just use local access
./run_app.sh

# Access at
http://localhost:8501
```

### For Showing to Team on Same Network
```bash
# Use network access
./run_app_network.sh

# Share this URL with team
http://192.168.0.140:8501

# When done, stop with Ctrl+C
```

### For Remote Demo
```bash
# On server
./run_app.sh

# From your laptop
ssh -L 8501:localhost:8501 sankar@99.46.139.88

# Access on laptop
http://localhost:8501
```

## Security Best Practices

1. **Default to localhost** - Use `./run_app.sh` for development
2. **Temporary network access** - Only use `./run_app_network.sh` when needed
3. **Close port after** - Remove firewall rule when done
4. **Never expose to internet** - Without proper authentication
5. **Use SSH tunnels** - For remote access
6. **Consider authentication** - For production deployments

## Summary

**Your current error (`http://99.46.139.88:8501` cannot be reached):**

✅ **This is normal and expected for security!**

**What to do instead:**

```bash
# On the same machine:
./run_app.sh
# Access: http://localhost:8501

# From another device on same network:
./run_app_network.sh
# Access: http://192.168.0.140:8501

# From remote location:
ssh -L 8501:localhost:8501 sankar@99.46.139.88
# Access: http://localhost:8501
```

The external URL is shown by Streamlit but blocked by design for your security! 🔒


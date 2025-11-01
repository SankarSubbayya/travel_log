#!/bin/bash
#-------------------------------------------------------------------------------------------
#  Copyright (c) 2016-2025.  SupportVectors AI Lab
#
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: SupportVectors AI Training
#-------------------------------------------------------------------------------------------

# Run the Travel Log Face Recognition Streamlit App (Network Access)
# WARNING: This allows access from other devices on your network

echo "🚀 Starting Travel Log Face Recognition App (Network Access)..."
echo ""

# Check if port 8501 is in use
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8501 is already in use. Stopping existing process..."
    pkill -f "streamlit run" 2>/dev/null
    sleep 2
    echo "✅ Cleaned up old processes"
    echo ""
fi

echo "⚠️  WARNING: App will be accessible from other devices on your network!"
echo "📝 Note: Models will download automatically on first use (~100-500MB)"
echo ""
echo "🌐 Access URLs:"
echo "   Local:   http://localhost:8501"
echo "   Network: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
echo "🔒 Security: Only use on trusted networks"
echo "🛑 To stop the server, press Ctrl+C"
echo ""

# Check if firewall might be blocking
if command -v ufw &> /dev/null; then
    echo "💡 If you can't access from other devices, you may need to open the firewall:"
    echo "   sudo ufw allow 8501/tcp"
    echo ""
fi

uv run streamlit run app.py --server.address 0.0.0.0 --server.port 8501


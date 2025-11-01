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

# Run the Travel Log Face Recognition Streamlit App (Local Access - Default)

echo "🚀 Starting Travel Log Face Recognition App..."
echo ""

# Check if port 8501 is in use
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8501 is already in use. Stopping existing process..."
    pkill -f "streamlit run" 2>/dev/null
    sleep 2
    echo "✅ Cleaned up old processes"
    echo ""
fi

echo "📝 Note: Models will download automatically on first use (~100-500MB)"
echo "🌐 The app will open in your browser at http://localhost:8501"
echo "🔒 Access: Localhost only (most secure)"
echo ""
echo "💡 For network access, use: ./run_app_network.sh"
echo "🛑 To stop the app, press Ctrl+C"
echo ""

uv run streamlit run app.py --server.address localhost --server.port 8501


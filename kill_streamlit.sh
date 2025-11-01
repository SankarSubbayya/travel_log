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

# Kill all Streamlit processes

echo "🔍 Checking for running Streamlit processes..."

# Find Streamlit processes
PIDS=$(pgrep -f "streamlit run")

if [ -z "$PIDS" ]; then
    echo "✅ No Streamlit processes found running"
    exit 0
fi

echo "📋 Found Streamlit processes:"
ps aux | grep -E "streamlit run|PID" | grep -v grep

echo ""
echo "🛑 Killing Streamlit processes..."
pkill -9 -f "streamlit run"

sleep 1

# Verify
REMAINING=$(pgrep -f "streamlit run")
if [ -z "$REMAINING" ]; then
    echo "✅ All Streamlit processes stopped successfully"
else
    echo "⚠️  Some processes may still be running. Trying again..."
    kill -9 $REMAINING 2>/dev/null
    echo "✅ Done"
fi

echo ""
echo "💡 You can now start the app with: ./run_app.sh"


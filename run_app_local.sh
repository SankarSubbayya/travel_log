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

# Run the Travel Log Face Recognition Streamlit App (Local Access Only)

echo "🚀 Starting Travel Log Face Recognition App (Local Access)..."
echo ""
echo "📝 Note: Models will download automatically on first use (~100-500MB)"
echo "🌐 The app will open in your browser at http://localhost:8501"
echo "🔒 Security: Access restricted to this machine only"
echo ""

uv run streamlit run app.py --server.address localhost --server.port 8501


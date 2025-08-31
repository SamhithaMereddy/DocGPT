#!/bin/bash
# Run script for Intelligent Document Search System

echo "🚀 Starting Intelligent Document Search System..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the application
echo "📄 Launching web interface at http://localhost:8501"
streamlit run search_interface.py --server.port=8501 --server.address=0.0.0.0


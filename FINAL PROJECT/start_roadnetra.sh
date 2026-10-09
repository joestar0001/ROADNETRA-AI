#!/bin/bash

# Ensure script is executable
chmod +x "$0" 2>/dev/null

echo "=========================================================="
echo "🚀 Starting ROADNETRA AI - Professional Multi-Portal Platform"
echo "IEEE Hackathon 2026 - Problem Statement 03.1 (Team ESPADA)"
echo "Apple Design System · Live Edge Vision · Multi-Agency ICCC"
echo "=========================================================="
echo ""
echo "📍 Web Interface: http://localhost:8080"
echo "🌐 Automatically opening in your browser..."
echo "=========================================================="

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Auto-open browser on macOS after a brief delay
(sleep 1.5 && open "http://localhost:8080" 2>/dev/null || open "http://127.0.0.1:8080" 2>/dev/null) &

# Run server using virtual environment python if present, else system python3
if [ -f "$DIR/.venv/bin/python3" ]; then
    PORT=8080 "$DIR/.venv/bin/python3" server.py
elif [ -f "/Users/bhaweshkumargautam/.gemini/antigravity/scratch/new_pothole_detection/.venv/bin/python3" ]; then
    PORT=8080 /Users/bhaweshkumargautam/.gemini/antigravity/scratch/new_pothole_detection/.venv/bin/python3 server.py
else
    PORT=8080 python3 server.py
fi

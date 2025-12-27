#!/bin/bash
# Run the OmniBridge server with threading fixes for macOS

# Fix macOS mutex lock issue
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "Starting OmniBridge Caption Server..."
echo "Open http://localhost:5000 in your browser"
echo ""

python3 server.py

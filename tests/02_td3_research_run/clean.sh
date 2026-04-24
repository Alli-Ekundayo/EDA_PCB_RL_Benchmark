#!/bin/bash
# Cleanup script for 02_td3_research_run

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Cleaning up 02_td3_research_run results..."

# Remove results directory
if [ -d "results" ]; then
    rm -rf results
    echo "  Removed: results/"
fi

# Remove any other temporary files if they exist
rm -f *.log
rm -f *.mp4
rm -f *.pdf

echo "Cleanup complete."

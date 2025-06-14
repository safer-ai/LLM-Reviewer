#!/bin/bash

echo "Starting Document Reviewer..."
echo

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3 from https://www.python.org/"
    exit 1
fi

# Set API key if provided as argument
if [ ! -z "$1" ]; then
    export ANTHROPIC_API_KEY="$1"
fi

# Run the GUI
python3 gui_reviewer.py

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Failed to start the reviewer."
    echo "Make sure all files are in the correct location."
    read -p "Press Enter to continue..."
fi

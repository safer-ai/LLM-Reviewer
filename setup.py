#!/usr/bin/env python3
"""
Setup Script for Document Reviewer
This script helps set up the reviewer for non-technical users
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def create_windows_executable():
    """Create a Windows executable using PyInstaller"""
    print("Creating Windows executable...")
    
    # Install PyInstaller if needed
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Create the executable
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",  # No console window
        "--name", "DocumentReviewer",
        "--icon", "icon.ico",  # You'll need to provide an icon
        "--add-data", "prompts;prompts",  # Include prompts folder
        "--add-data", "parameters.yaml;.",  # Include config
        "--hidden-import", "tiktoken_ext.openai_public",
        "--hidden-import", "tiktoken_ext",
        "gui_reviewer.py"
    ]
    
    subprocess.check_call(cmd)
    print("Executable created in 'dist' folder!")


def create_mac_app():
    """Create a Mac application bundle"""
    print("Creating Mac application...")
    
    # Create app structure
    app_name = "DocumentReviewer"
    app_path = Path(f"{app_name}.app")
    contents = app_path / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    
    # Create directories
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(exist_ok=True)
    
    # Create Info.plist
    info_plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Document Reviewer</string>
    <key>CFBundleIdentifier</key>
    <string>com.yourcompany.documentreviewer</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleExecutable</key>
    <string>run.sh</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
</dict>
</plist>"""
    
    with open(contents / "Info.plist", "w") as f:
        f.write(info_plist)
    
    # Create run script
    run_script = """#!/bin/bash
cd "$(dirname "$0")"
cd ../Resources
python3 gui_reviewer.py
"""
    
    run_file = macos / "run.sh"
    with open(run_file, "w") as f:
        f.write(run_script)
    
    # Make executable
    os.chmod(run_file, 0o755)
    
    # Copy files
    import shutil
    shutil.copy("gui_reviewer.py", resources)
    shutil.copy("reviewer.py", resources)
    shutil.copy("parameters.yaml", resources)
    shutil.copytree("prompts", resources / "prompts", dirs_exist_ok=True)
    
    print(f"Mac app created: {app_path}")


def create_batch_files():
    """Create simple batch/shell files for running"""
    
    # Windows batch file
    windows_batch = """@echo off
title Document Reviewer

echo Starting Document Reviewer...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
python -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo ERROR: tkinter not available. Please reinstall Python with tkinter support.
    pause
    exit /b 1
)

REM Set API key if provided as argument
if not "%1"=="" (
    set ANTHROPIC_API_KEY=%1
)

REM Run the GUI
python gui_reviewer.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start the reviewer.
    echo Make sure all files are in the correct location.
    pause
)
"""
    
    with open("RunReviewer.bat", "w") as f:
        f.write(windows_batch)
    
    # Mac/Linux shell script
    unix_script = """#!/bin/bash

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
"""
    
    with open("RunReviewer.sh", "w") as f:
        f.write(unix_script)
    
    os.chmod("RunReviewer.sh", 0o755)
    
    print("Created RunReviewer.bat (Windows) and RunReviewer.sh (Mac/Linux)")


def install_requirements():
    """Install required Python packages"""
    print("Installing required packages...")
    
    requirements = [
        "anthropic",
        "PyYAML", 
        "tqdm",
        "PyMuPDF",  # For PDF support
        "python-docx",  # For Word support
        "pandas",  # For Excel export
        "openpyxl",  # For Excel export
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"WARNING: Failed to install {package}. Some features may not work.")


def main():
    print("Document Reviewer Setup")
    print("=" * 50)
    print()
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7 or higher is required.")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()}")
    print()
    
    # Install requirements
    response = input("Install required packages? (y/n): ")
    if response.lower() == 'y':
        install_requirements()
    
    print()
    
    # Create batch/shell files
    create_batch_files()
    
    print()
    
    # Create executable
    if platform.system() == "Windows":
        response = input("Create Windows executable? (y/n): ")
        if response.lower() == 'y':
            create_windows_executable()
    elif platform.system() == "Darwin":
        response = input("Create Mac app? (y/n): ")
        if response.lower() == 'y':
            create_mac_app()
    
    print()
    print("Setup complete!")
    print()
    print("To run the Document Reviewer:")
    if platform.system() == "Windows":
        print("  - Double-click 'RunReviewer.bat'")
        print("  - Or run: python gui_reviewer.py")
    else:
        print("  - Double-click 'RunReviewer.sh'")
        print("  - Or run: python3 gui_reviewer.py")
    print()
    print("First time setup:")
    print("  1. Get your API key from https://console.anthropic.com")
    print("  2. The program will ask for it on first run")
    print()
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
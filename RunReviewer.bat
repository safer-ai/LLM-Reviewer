@echo off
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

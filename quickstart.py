#!/usr/bin/env python3
"""Quick start script for DFL Tool with Bearing Dataset"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """Check if all dependencies are installed"""
    print("Checking dependencies...")
    try:
        import torch
        import pandas
        import sklearn
        import fastapi
        import uvicorn
        print("✓ All dependencies installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("\nInstalling dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True

def download_dataset():
    """Download bearing dataset"""
    print("\nDownloading bearing dataset...")
    try:
        from data_utils import BearingDatasetLoader
        dataset = BearingDatasetLoader()
        print("✓ Dataset downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download dataset: {e}")
        return False

def run_test():
    """Run basic tests"""
    print("\nRunning basic tests...")
    result = subprocess.run([sys.executable, "test_bearing.py"])
    return result.returncode == 0

def start_server():
    """Start the API server"""
    print("\n" + "=" * 60)
    print("Starting DFL Tool API Server...")
    print("=" * 60)
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([sys.executable, "api.py"])
    except KeyboardInterrupt:
        print("\n\nServer stopped.")

def main():
    print("=" * 60)
    print("DFL Tool - Quick Start for Bearing Dataset")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("Please install dependencies and try again")
        return 1
    
    # Download dataset
    if not download_dataset():
        print("Dataset download failed, but you can specify a local CSV file")
    
    # Ask user what to do
    print("\n" + "=" * 60)
    print("What would you like to do?")
    print("=" * 60)
    print("1. Run tests")
    print("2. Start API server")
    print("3. Run example script")
    print("4. All of the above")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        run_test()
    elif choice == "2":
        start_server()
    elif choice == "3":
        print("\nStarting API server in background...")
        server_process = subprocess.Popen([sys.executable, "api.py"])
        time.sleep(3)  # Wait for server to start
        print("Running examples...")
        subprocess.run([sys.executable, "example_bearing.py"])
        print("\nStopping server...")
        server_process.terminate()
    elif choice == "4":
        if run_test():
            print("\n✓ Tests passed! Starting server...")
            start_server()
        else:
            print("\n✗ Tests failed. Please fix errors before starting server.")
    else:
        print("Goodbye!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

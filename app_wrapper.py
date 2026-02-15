#!/usr/bin/env python
"""
Simplified wrapper for Traffic Counter Application
This is used for creating the executable
"""
import sys
import os

# Add the project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

if __name__ == "__main__":
    try:
        from test import main
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")

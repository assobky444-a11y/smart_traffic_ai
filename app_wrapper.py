#!/usr/bin/env python
"""
Launcher for TraffiCount Pro executable.

Starts the bundled Flask server, opens the default web browser and
keeps the process alive.  This script is the entrypoint that Nuitka
will compile into the standalone executable.
"""
import sys
import os
import threading
import time
import webbrowser

# Add the project directory to path so imports like `from app import app`
# work even when the executable is run from another location.
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# optional CUDA check (drivers must be present on end-user machine)
def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            print("[INFO] CUDA detected - GPU acceleration enabled.")
        else:
            print("[WARN] CUDA not available; falling back to CPU.")
    except ImportError:
        print("[WARN] PyTorch not bundled; CUDA check skipped.")


def run_server():
    # import inside function to avoid loading Flask before Nuitka startup
    from app import app
    # disable reloader when running in compiled binary
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    check_cuda()
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    # give Flask a moment to bind
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")
    try:
        # keep the main thread alive while server is running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
 
#!/usr/bin/env python3
"""
Simple script to restart the Streamlit probe analyzer app.
Kills any existing instance on port 8501 and starts a new one.
"""

import subprocess
import time
import signal
import os

PORT = 8501
APP_FILE = "probe_analyzer_app.py"

def find_process_on_port(port):
    """Find process ID using the specified port."""
    try:
        # Use lsof to find process using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            return int(result.stdout.strip().split('\n')[0])
    except Exception as e:
        print(f"Error finding process: {e}")
    return None

def kill_streamlit():
    """Kill existing Streamlit process on port 8501."""
    pid = find_process_on_port(PORT)
    if pid:
        print(f"Found Streamlit process on port {PORT} (PID: {pid})")
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Killed process {pid}")
            time.sleep(2)  # Wait for process to die
            
            # Force kill if still running
            try:
                os.kill(pid, 0)  # Check if still alive
                print(f"Process still alive, force killing...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
            except ProcessLookupError:
                pass  # Process is dead
                
        except ProcessLookupError:
            print(f"Process {pid} already dead")
    else:
        print(f"No process found on port {PORT}")

def start_streamlit():
    """Start Streamlit app."""
    print(f"\nStarting Streamlit on port {PORT}...")
    subprocess.Popen([
        "streamlit", "run", APP_FILE,
        "--server.port", str(PORT),
        "--server.address", "0.0.0.0"
    ])
    print(f"✅ Streamlit started at http://0.0.0.0:{PORT}")

if __name__ == "__main__":
    print("=" * 60)
    print("Streamlit Restart Script")
    print("=" * 60)
    
    kill_streamlit()
    start_streamlit()
    
    print("\n" + "=" * 60)
    print("Done! Streamlit should be running now.")
    print("=" * 60)

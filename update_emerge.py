#!/usr/bin/env python3
"""
A script to download the latest EMerge repository into the current folder.
Requires: Python 3.6+, git installed on PATH.
Usage: Run this script in the folder that contains the "fem" directory.
"""
import os
import sys
import subprocess
import shutil
import stat

def error(msg):
    print(f"Error: {msg}")
    sys.exit(1)

def on_rm_error(func, path, exc_info):
    """Error handler for shutil.rmtree to handle read-only files."""
    # Try to make file writable and retry
    try:
        os.chmod(path, stat.S_IWUSR)
        func(path)
    except Exception:
        print(f"Could not remove {path}")
        raise

def main():
    cwd = os.getcwd()
    fem_path = os.path.join(cwd, "fem")
    if not os.path.isdir(fem_path):
        error("'fem' directory not found in current folder. Please run this script in the folder containing 'fem/'.")

    repo_url = "https://github.com/FennisRobert/EMerge.git"
    tmp_dir = os.path.join(cwd, ".emerge_tmp")

    # Clean up any previous temp directory
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, onerror=on_rm_error)

    # Clone only latest commit for speed
    print(f"Cloning {repo_url}...")
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, tmp_dir])
    except subprocess.CalledProcessError:
        error("Failed to clone repository. Ensure git is installed and you have network access.")

    # Move contents into cwd, overwriting existing files
    for item in os.listdir(tmp_dir):
        src = os.path.join(tmp_dir, item)
        dst = os.path.join(cwd, item)
        # Remove existing
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst, onerror=on_rm_error)
            else:
                try:
                    os.remove(dst)
                except PermissionError:
                    os.chmod(dst, stat.S_IWUSR)
                    os.remove(dst)
        # Move new
        shutil.move(src, dst)

    # Clean up temp directory
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, onerror=on_rm_error)

    print("EMerge repository has been updated successfully.")

if __name__ == "__main__":
    main()

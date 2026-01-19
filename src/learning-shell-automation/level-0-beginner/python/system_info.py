#!/usr/bin/env python3
"""
Level 0 - Beginner: Basic System Information in Python
"""

import os
import platform
from datetime import datetime

def main():
    print("=== System Information ===")
    print(f"Hostname: {platform.node()}")
    print(f"Current User: {os.getenv('USER', os.getenv('USERNAME', 'Unknown'))}")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print("==========================")

if __name__ == "__main__":
    main()

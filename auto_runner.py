#!/usr/bin/env python3
"""Headless 24/7 autonomous cycle runner for GitHub Actions & background daemons."""
import subprocess
import sys
import time

# Feed autonomous exploration, research, language training, and a clean /quit
COMMANDS = "\n".join([
    "/explore autonomous systems architecture and self-optimizing code",
    "practice language patterns and expand reasoning heuristics",
    "/quit"
]) + "\n"

def run_cycle():
    print("[24/7 Runner] Booting synthetic_living_brain.py (loading language_cortex & reasoning_and_research_cortex)...")
    proc = subprocess.Popen(
        [sys.executable, "synthetic_living_brain.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    try:
        stdout, _ = proc.communicate(input=COMMANDS, timeout=240)
        print(stdout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("[24/7 Runner] Cycle timed out; forcing clean exit.")

if __name__ == "__main__":
    run_cycle()

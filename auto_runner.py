#!/usr/bin/env python3
"""Headless 24/7 autonomous cycle runner that activates & updates all core brain modules."""
import subprocess
import sys
import time

CORE_MODULES = [
    "synthetic_living_brain.py",
    "language_cortex.py",
    "reasoning_and_research_cortex.py"
]

COMMANDS = "\n".join([
    "/explore autonomous systems architecture and self-optimizing code",
    "practice language patterns and expand reasoning heuristics",
    "/quit"
]) + "\n"

def stamp_modules():
    stamp = f"# [24/7-AUTONOMOUS-HEARTBEAT] Last active cycle: {int(time.time())}\n"
    for mod in CORE_MODULES:
        try:
            with open(mod, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines and lines[-1].startswith("# [24/7-AUTONOMOUS-HEARTBEAT]"):
                lines[-1] = stamp
            else:
                if lines and not lines[-1].endswith("\n"):
                    lines[-1] += "\n"
                lines.append(stamp)
            with open(mod, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except OSError:
            pass

def run_cycle():
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
    stamp_modules()

if __name__ == "__main__":
    run_cycle()

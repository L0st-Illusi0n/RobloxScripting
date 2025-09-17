# Forsaken Generator Bot

An automation bot that solves the generator puzzle in Forsaken (Roblox).  
It uses computer vision to detect the grid and endpoints, then simulates human-like mouse movements to complete the puzzle automatically.

## Features
- Detects puzzle grid and endpoints
- Solves puzzles with strict or flexible pathing
- Smooth, human-like cursor movement
- GUI with status indicator, debug window, and settings (speed, jitter, smoothness)
- Supports single-puzzle and continuous modes

## Installation
1. Clone this repository or download the source.
2. Install dependencies:
   pip install -r requirements.txt
   Intercepter driver by Oblitum at: https://github.com/oblitum/Interception/releases
3. Install the Intercepter driver by downloading the latest release and running the installer inside the "command line installer" folder using the following command:
   install-interception.exe /install
4. Run Build.bat and wait for the exe to finish building then can be accessed inside the "dist" folder it will create inside of the src folder.
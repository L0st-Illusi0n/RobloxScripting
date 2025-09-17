# Forsaken 1x Entanglement Autoclicker

A simple but optimized bot that automatically detects and clicks popups on the screen.  
Built for speed, with multithreading, region-based detection, and a Tkinter-based GUI.

## Features
- Real-time popup detection using OpenCV template matching
- Click automation via Interception driver
- Customizable settings:
  - Confidence (image match strictness)
  - Click interval
  - Screen reader resolution scaling (speed vs precision)
- GUI controls:
  - Start/Stop toggle
  - Status indicator (Running / Idle / Disabled)
  - Click counter with reset button
  - Debug console (logs every click)

## Installation
1. Clone this repository or download the source.
2. Install dependencies:
   pip install -r requirements.txt
   Intercepter driver by Oblitum at: https://github.com/oblitum/Interception/releases
3. Install the Intercepter driver by downloading the latest release and running the installer inside the "command line installer" folder using the following command:
   install-interception.exe /install
4. Run Build.bat and wait for the exe to finish building then can be accessed inside the "dist" folder it will create inside of the src folder.
# Doom By Fate Generator Bot

Automation bot for the generator minigame in Doom By Fate (Roblox). It detects the colored wire puzzle, connects the correct endpoints, then performs the follow-up cord pulls automatically.

## Features
- Detects top and bottom wire endpoints by color.
- Solves the wire stage and then handles the cord pull stage.
- Supports a manual start hotkey and an auto mode for repeated generators.
- Keeps `DBFGenBot.timing.json` external and editable next to the built executable.
- Includes a one-file Windows build script using Nuitka.

## Requirements
- Windows
- Python 3.12
- The Interception driver installed
- A game layout that matches the capture regions currently hardcoded in `src/DBFGenBot.py`

## Installation
1. Clone this repository or download the source.
2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3. Install the Interception driver from:
   `https://github.com/oblitum/Interception/releases`
4. Open the extracted Interception package, go to the `command line installer` folder, then run:

```powershell
install-interception.exe /install
```

5. Change into `src` and run `Build.bat`.

## Usage
- Keep `DBFGenBot.exe` and `DBFGenBot.timing.json` in the same folder.
- Launch `DBFGenBot.exe`.
- Press `Insert` to start solving.
- Press `Home` to toggle auto mode.
- Press `Delete` to immediately kill the current run.

When auto mode is enabled, the bot waits for a valid generator puzzle and starts automatically when one is detected.

## Timing Config
`DBFGenBot.timing.json` is intentionally kept outside the EXE so you can tune movement without rebuilding.

Available settings:
- `wire_speed`: multiplier for wire movement speed
- `cord_speed`: multiplier for cord pull speed
- `wire_grab_delay`: delay before dragging a wire
- `cord_grab_delay`: delay before pulling the cord

Compiled builds read the timing file from the same folder as `DBFGenBot.exe`. Running the Python source directly reads the file from `src`.

## Build
Run the build script from the `src` folder:

```powershell
Build.bat
```

This produces:
- `src/dist/DBFGenBot.exe`
- `src/dist/DBFGenBot.timing.json`

The build script also cleans the temporary Nuitka build folders after a successful build.

## Notes
- This bot is Windows-only because it depends on the Interception driver.
- The screen capture boxes are hardcoded for the current puzzle position. If the puzzle appears in a different location on your setup, adjust the values near `TOP_BOX`, `BOT_BOX`, and `CORD_BOX` in `src/DBFGenBot.py`.

# type: ignore[import]
import cv2
import numpy as np
import ctypes
import time
import os
import sys
import mss
import threading
import queue
import interception

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "Images")
CONFIDENCE = 0.85
CHECK_INTERVAL = 0.01
CLICK_INTERVAL = 0.02
DEBUG_MODE = True
stop_requested = False
SCALE = 0.5
DEDUP_RADIUS = 20
DEDUP_TIMEOUT = 0.5
REGION = {
    "top": 185,
    "left": 460,
    "width": 1460 - 460,
    "height": 900 - 185
}
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = sys._MEIPASS
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "Images")
interception.auto_capture_devices(keyboard=True, mouse=True)

def move_to(x, y):
    interception.move_to(int(x), int(y))

def click(x=None, y=None, button="left", delay=0.05):
    if x is None or y is None:
        x, y = get_cursor_pos()
    interception.click(int(x), int(y), button=button, delay=delay)

def get_cursor_pos():
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

templates = []
for f in os.listdir(IMAGE_FOLDER):
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(IMAGE_FOLDER, f)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            if SCALE != 1.0:
                img = cv2.resize(img, (0,0), fx=SCALE, fy=SCALE)
            templates.append((img, f))
        else:
            print(f"[!] Failed to load {path}")

if not templates:
    print(f"[!] No valid images found in '{IMAGE_FOLDER}'")
    exit()

match_queue = queue.Queue()
recent_clicks = []

def is_duplicate(x, y):
    now = time.time()
    global recent_clicks
    recent_clicks = [(cx, cy, t) for (cx, cy, t) in recent_clicks if now - t < DEDUP_TIMEOUT]
    for (cx, cy, t) in recent_clicks:
        if abs(cx - x) <= DEDUP_RADIUS and abs(cy - y) <= DEDUP_RADIUS:
            return True
    recent_clicks.append((x, y, now))
    return False

def detector_thread(log_fn, on_click, running_flag):
    global stop_requested
    sct = mss.mss()
    while not stop_requested and running_flag():
        frame = np.array(sct.grab(REGION))[:, :, :3]
        if SCALE != 1.0:
            frame = cv2.resize(frame, (0,0), fx=SCALE, fy=SCALE)
        for template, name in templates:
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val >= CONFIDENCE:
                h, w, _ = template.shape
                center_x = int((max_loc[0] + w // 2) / SCALE + REGION["left"])
                center_y = int((max_loc[1] + h // 2) / SCALE + REGION["top"])
                if not is_duplicate(center_x, center_y):
                    match_queue.put((center_x, center_y, name))

        time.sleep(CHECK_INTERVAL)

def clicker_thread(log_fn, on_click, running_flag):
    global stop_requested
    while not stop_requested and running_flag():
        try:
            x, y, name = match_queue.get(timeout=1)
            move_to(x, y)
            click()
            if on_click:
                on_click(name, x, y)
            if log_fn:
                log_fn(f"[✓] Clicked {name} at ({x}, {y})\n")
            elif DEBUG_MODE:
                print(f"[✓] Clicked {name} at ({x}, {y})")
            time.sleep(CLICK_INTERVAL)
        except queue.Empty:
            pass

def run_with_callbacks(on_click=None, log_fn=None, running_flag=lambda: True):
    global stop_requested
    stop_requested = False
    t1 = threading.Thread(target=detector_thread, args=(log_fn, on_click, running_flag), daemon=True)
    t2 = threading.Thread(target=clicker_thread, args=(log_fn, on_click, running_flag), daemon=True)
    t1.start()
    t2.start()
    while not stop_requested and running_flag():
        time.sleep(0.1)

def main():
    print("Bot started. Watching for images...")
    run_with_callbacks()

if __name__ == "__main__":
    main()

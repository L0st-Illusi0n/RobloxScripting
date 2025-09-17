# type: ignore[import]
import cv2
import numpy as np
import ctypes
import time
import os
import mss
import threading
import queue
import interception

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "Images")
CONFIDENCE = 0.72
CHECK_INTERVAL = 0.05
CLICK_INTERVAL = 0.02
DEBUG_MODE = True
REGION = {
    "top": 185,
    "left": 460,
    "width": 1460 - 460,
    "height": 900 - 185
}
SCALE = 0.67
DEDUP_RADIUS = 20
DEDUP_TIMEOUT = 1
stop_requested = False
templates = []
match_queue = queue.Queue()
recent_clicks = []
_template_scale = 1.0
_scale_warning_emitted = False
state_lock = threading.Lock()
templates_lock = threading.Lock()
interception.auto_capture_devices(keyboard=True, mouse=True)

def _emit(log_fn, message):
    if log_fn:
        if not message.endswith("\n"):
            message += "\n"
        log_fn(message)
    else:
        print(message)

def get_effective_scale():
    return SCALE if SCALE > 0 else 1.0

def _should_resize(scale):
    return abs(scale - 1.0) > 1e-6

def load_templates(log_fn=None):
    global _template_scale, _scale_warning_emitted
    with templates_lock:
        templates.clear()
        scale = SCALE
        if scale <= 0:
            if not _scale_warning_emitted:
                _emit(log_fn, "[!] SCALE must be greater than 0. Defaulting to 1.0.")
            _scale_warning_emitted = True
            scale = 1.0
        else:
            _scale_warning_emitted = False
        _template_scale = scale
        resize_needed = _should_resize(scale)
        if not os.path.isdir(IMAGE_FOLDER):
            _emit(log_fn, f"[!] Image folder not found: '{IMAGE_FOLDER}'")
            return False
        for filename in sorted(os.listdir(IMAGE_FOLDER)):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(IMAGE_FOLDER, filename)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                _emit(log_fn, f"[!] Failed to load {path}")
                continue
            if resize_needed:
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            templates.append((img, filename))
    if not templates:
        _emit(log_fn, f"[!] No valid images found in '{IMAGE_FOLDER}'")
        return False
    _emit(log_fn, f"Loaded {len(templates)} templates from '{IMAGE_FOLDER}'")
    return True

def _reset_runtime_state():
    global match_queue, recent_clicks
    with state_lock:
        recent_clicks.clear()
    while True:
        try:
            match_queue.get_nowait()
        except queue.Empty:
            break

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

def is_duplicate(x, y):
    now = time.time()
    with state_lock:
        active = []
        for cx, cy, t in recent_clicks:
            if now - t < DEDUP_TIMEOUT:
                active.append((cx, cy, t))
        recent_clicks[:] = active
        for cx, cy, _ in recent_clicks:
            if abs(cx - x) <= DEDUP_RADIUS and abs(cy - y) <= DEDUP_RADIUS:
                return True
        recent_clicks.append((x, y, now))
        return False

def _should_stop(stop_event=None, running_flag=None):
    if stop_requested:
        return True
    if stop_event is not None and stop_event.is_set():
        return True
    if running_flag is not None:
        try:
            if not running_flag():
                return True
        except Exception:
            return True
    return False

def detector_thread(stop_event=None, running_flag=None, log_fn=None):
    sct = mss.mss()
    while True:
        if _should_stop(stop_event, running_flag):
            break
        with templates_lock:
            current_templates = list(templates)
            template_scale = _template_scale
        if not current_templates:
            time.sleep(0.1)
            continue
        frame = np.array(sct.grab(REGION))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if _should_resize(template_scale):
            frame_bgr = cv2.resize(frame_bgr, (0, 0), fx=template_scale, fy=template_scale)
        for template, name in current_templates:
            result = cv2.matchTemplate(frame_bgr, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val >= CONFIDENCE:
                h, w, _ = template.shape
                center_x = int((max_loc[0] + w // 2) / template_scale + REGION["left"])
                center_y = int((max_loc[1] + h // 2) / template_scale + REGION["top"])
                if not is_duplicate(center_x, center_y):
                    match_queue.put((center_x, center_y, name))
        time.sleep(CHECK_INTERVAL)

def clicker_thread(stop_event=None, running_flag=None, on_click=None, log_fn=None):
    while True:
        if _should_stop(stop_event, running_flag):
            break
        try:
            x, y, name = match_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if _should_stop(stop_event, running_flag):
            break
        move_to(x, y)
        time.sleep(0.01)
        click()
        if on_click is not None:
            on_click(name, x, y)
        elif log_fn is not None:
            _emit(log_fn, f"[+] Clicked {name} at ({x}, {y})")
        elif DEBUG_MODE:
            print(f"[+] Clicked {name} at ({x}, {y})")
        time.sleep(CLICK_INTERVAL)

def run_with_callbacks(on_click=None, log_fn=None, running_flag=None):
    global stop_requested
    stop_requested = False
    _reset_runtime_state()
    if not load_templates(log_fn):
        stop_requested = True
        return
    stop_event = threading.Event()
    detector = threading.Thread(
        target=detector_thread,
        kwargs={"stop_event": stop_event, "running_flag": running_flag, "log_fn": log_fn},
        daemon=True,
    )
    clicker = threading.Thread(
        target=clicker_thread,
        kwargs={"stop_event": stop_event, "running_flag": running_flag, "on_click": on_click, "log_fn": log_fn},
        daemon=True,
    )
    detector.start()
    clicker.start()
    try:
        while not _should_stop(stop_event, running_flag):
            time.sleep(0.1)
    finally:
        stop_event.set()
        stop_requested = True
        detector.join(timeout=1.0)
        clicker.join(timeout=1.0)

def main():
    if not load_templates():
        return
    print("Bot started. Watching for images...")
    threading.Thread(target=detector_thread, daemon=True).start()
    threading.Thread(target=clicker_thread, daemon=True).start()
    while True:
        time.sleep(1)  # keep main alive

if __name__ == "__main__":
    main()

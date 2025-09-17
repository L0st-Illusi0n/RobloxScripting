import ctypes
import cv2
import numpy as np
import pyautogui
import keyboard
from collections import deque
import interception
import time, random

interception.auto_capture_devices(keyboard=True, mouse=True)

SPEED = 1.25
JITTER_SCALE = 1.2
SMOOTHNESS = 0.8

def set_speed(value):
    global SPEED
    try:
        SPEED = max(0.1, float(value))
    except (TypeError, ValueError):
        pass

def set_jitter_scale(value):
    global JITTER_SCALE
    try:
        JITTER_SCALE = max(0.0, float(value))
    except (TypeError, ValueError):
        pass

def set_smoothness(value):
    global SMOOTHNESS
    try:
        SMOOTHNESS = max(0.2, float(value))
    except (TypeError, ValueError):
        pass

def get_settings():
    return {
        'speed': SPEED,
        'jitter_scale': JITTER_SCALE,
        'smoothness': SMOOTHNESS,
    }

def scaled_sleep(base_time):
    time.sleep(base_time * SPEED)

def move_to(x, y):
    interception.move_to(int(x), int(y))

def mouse_down():
    x, y = get_cursor_pos()
    interception.mouse_down(button="left")

def mouse_up():
    x, y = get_cursor_pos()
    interception.mouse_up(button="left")

def click(x=None, y=None, button="left", delay=0.05):
    if x is None or y is None:
        x, y = get_cursor_pos()
    interception.click(int(x), int(y), button=button, delay=delay)

def get_cursor_pos():
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

SEARCH_REGION = {
    "top": 200,
    "left": 400,
    "width": 1100,
    "height": 800
}


# Tunables for dynamic color detection
COLOR_DISTANCE_THRESHOLD = 45
SATURATION_THRESHOLD = 60
VALUE_THRESHOLD = 50
BRIGHT_VALUE_THRESHOLD = 200
MIN_POINT_PIXELS = 20
CORE_CROP_RATIO = 0.25

SW_RESTORE = 9
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
EnumWindows = user32.EnumWindows
IsWindowVisible = user32.IsWindowVisible
GetWindowTextW = user32.GetWindowTextW
GetWindowTextLengthW = user32.GetWindowTextLengthW
GetForegroundWindow = user32.GetForegroundWindow
SetForegroundWindow = user32.SetForegroundWindow
ShowWindow = user32.ShowWindow
BringWindowToTop = user32.BringWindowToTop
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)

def _enum_windows():
    results = []
    @EnumWindowsProc
    def _proc(hwnd, lParam):
        try:
            if IsWindowVisible(hwnd):
                length = GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value
                    results.append((hwnd, title))
        except Exception:
            pass
        return True
    EnumWindows(_proc, 0)
    return results

def find_window_by_substring(substring):
    substring = substring.lower()
    for hwnd, title in _enum_windows():
        if substring in title.lower():
            return hwnd, title
    return None, None

def focus_window_by_name(substring="Roblox", timeout=1.5, require_restore=True):
    hwnd = find_window_by_substring(substring)
    if not hwnd:
        return False
    try:
        if require_restore:
            ShowWindow(hwnd, SW_RESTORE)
        BringWindowToTop(hwnd)
        SetForegroundWindow(hwnd)
    except Exception:
        pass
    deadline = time.time() + timeout
    while time.time() < deadline:
        if GetForegroundWindow() == hwnd:
            return True
        scaled_sleep(0.03)
    try:
        ShowWindow(hwnd, SW_RESTORE)
        BringWindowToTop(hwnd)
        SetForegroundWindow(hwnd)
    except Exception:
        pass
    return GetForegroundWindow() == hwnd

def ensure_focused_and_pause(window_substring="Roblox", timeout=1.5, pause_after=0.12):
    ok = focus_window_by_name(window_substring, timeout=timeout)
    if ok:
        scaled_sleep(pause_after)
    return ok

def smoothstep(t):
    return t * t * (3 - 2 * t)

def smooth_move_to(target, duration=0.2, jitter=False):
    start_x, start_y = get_cursor_pos()
    dx = target[0] - start_x
    dy = target[1] - start_y
    distance = max(1.0, (dx**2 + dy**2) ** 0.5)
    smoothness = max(0.2, SMOOTHNESS)
    steps = max(4, int((distance / 12) * smoothness))
    delay = max(0.004, duration / max(1, steps))
    jitter_amount = max(0.0, JITTER_SCALE) * 0.7
    for step in range(1, steps + 1):
        t = step / steps
        eased = smoothstep(t)
        x = start_x + dx * eased
        y = start_y + dy * eased
        if jitter and step < steps and jitter_amount > 0:
            x += random.uniform(-jitter_amount, jitter_amount)
            y += random.uniform(-jitter_amount, jitter_amount)
        interception.move_to(int(x), int(y))
        scaled_sleep(delay)
    interception.move_to(int(target[0]), int(target[1]))

TOLERANCE = 5

def color_match(pixel, target, tol=TOLERANCE):
    return all(abs(int(pixel[i]) - target[i]) <= tol for i in range(3))

def capture_region():
    screenshot = pyautogui.screenshot(region=(
        SEARCH_REGION["left"], SEARCH_REGION["top"],
        SEARCH_REGION["width"], SEARCH_REGION["height"]
    ))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def find_puzzle_bbox(img):
    mask_black = cv2.inRange(img, (0, 0, 0), (0, 0, 0))
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)

def filter_lines(lines, tolerance=5):
    if len(lines) < 2:
        return lines
    spacings = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
    spacing_counts = {}
    for s in spacings:
        rounded = round(s / tolerance) * tolerance
        spacing_counts[rounded] = spacing_counts.get(rounded, 0) + 1
    expected_spacing = max(spacing_counts, key=spacing_counts.get)
    filtered = [lines[0]]
    for i in range(1, len(lines)):
        if abs(lines[i] - filtered[-1] - expected_spacing) <= tolerance:
            filtered.append(lines[i])
    return filtered

def detect_grid(img):
    debug = img.copy()
    mask_div = cv2.inRange(img, (20, 20, 20), (20, 20, 20))
    horizontal_sum = np.sum(mask_div, axis=1)
    horizontal_lines = [i for i, v in enumerate(horizontal_sum) if v > 0.9 * mask_div.shape[1] * 255]
    horizontal_lines = filter_lines(horizontal_lines, tolerance=5)
    vertical_sum = np.sum(mask_div, axis=0)
    vertical_lines = [i for i, v in enumerate(vertical_sum) if v > 0.9 * mask_div.shape[0] * 255]
    vertical_lines = filter_lines(vertical_lines, tolerance=5)
    rows = len(horizontal_lines) - 1
    cols = len(vertical_lines) - 1
    print(f"Detected grid: {rows}x{cols}")
    for r in horizontal_lines:
        cv2.line(debug, (0, r), (img.shape[1], r), (0, 255, 0), 1)
    for c in vertical_lines:
        cv2.line(debug, (c, 0), (c, img.shape[0]), (0, 255, 0), 1)
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            x1, x2 = vertical_lines[j], vertical_lines[j + 1]
            y1, y2 = horizontal_lines[i], horizontal_lines[i + 1]
            cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return rows, cols, horizontal_lines, vertical_lines, debug

def detect_points(img, rows, cols, h_lines, v_lines, debug):
    detected = []
    color_clusters = []

    def assign_label(mean_color):
        for cluster in color_clusters:
            if np.linalg.norm(mean_color - cluster['mean']) < COLOR_DISTANCE_THRESHOLD:
                total = cluster['count'] + 1
                cluster['mean'] = (cluster['mean'] * cluster['count'] + mean_color) / total
                cluster['count'] = total
                return cluster['label']
        label = f"Flow{len(color_clusters) + 1}"
        color_clusters.append({
            'label': label,
            'mean': mean_color.copy(),
            'count': 1,
        })
        return label

    for i in range(rows):
        for j in range(cols):
            x1, x2 = v_lines[j], v_lines[j + 1]
            y1, y2 = h_lines[i], h_lines[i + 1]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            h, w = roi.shape[:2]
            pad = int(round(min(h, w) * CORE_CROP_RATIO))
            if pad > 0 and h > 2 * pad and w > 2 * pad:
                core = roi[pad:h - pad, pad:w - pad]
            else:
                core = roi
            if core.size == 0:
                core = roi
            hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            mask = ((sat > SATURATION_THRESHOLD) & (val > VALUE_THRESHOLD)) | (val > BRIGHT_VALUE_THRESHOLD)
            count = int(np.count_nonzero(mask))
            if count < MIN_POINT_PIXELS:
                continue
            pixels = core[mask]
            if pixels.size == 0:
                continue
            mean_color = np.asarray(pixels.mean(axis=0), dtype=np.float32)
            label = assign_label(mean_color)
            detected.append((i, j, label))
            color_bgr = tuple(int(round(c)) for c in mean_color)
            cv2.circle(debug, (cx, cy), 10, color_bgr, 2)
            cv2.putText(
                debug, label, (cx - 20, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    color_map = {cluster['label']: tuple(int(round(c)) for c in cluster['mean']) for cluster in color_clusters}
    return detected, color_map

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def solve_paths(rows, cols, detected):
    endpoints = {}
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for r, c, color in detected:
        endpoints.setdefault(color, []).append((r, c))
        grid[r][c] = color
    if not endpoints:
        return None
    
    def neighbors(cell):
        r, c = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield (nr, nc)

    def free_neighbor_count(cell, partner):
        count = 0
        for nb in neighbors(cell):
            if nb == partner or grid[nb[0]][nb[1]] is None:
                count += 1
        return count
    color_pairs = {}
    active_colors = []
    for color, pts in endpoints.items():
        if len(pts) < 2:
            print(f"Skipping {color}: only {len(pts)} endpoint detected.")
            continue
        if len(pts) > 2:
            print(f"Warning: expected two endpoints for {color}, using the first two.")
            pts = sorted(pts)[:2]
        a, b = pts
        if free_neighbor_count(a, b) <= free_neighbor_count(b, a):
            start, end = a, b
        else:
            start, end = b, a
        color_pairs[color] = (start, end)
        active_colors.append(color)
    if not active_colors:
        return None

    active_colors.sort(key=lambda c: (-manhattan(color_pairs[c][0], color_pairs[c][1]), c))
    progress = {color: [color_pairs[color][0]] for color in active_colors}
    completed = set()
    solution_paths = {}
    def path_possible(color):
        current = progress[color][-1]
        target = color_pairs[color][1]
        queue = deque([current])
        visited = {current}
        while queue:
            cell = queue.popleft()
            for nb in neighbors(cell):
                if nb == target:
                    return True
                if nb in visited:
                    continue
                if grid[nb[0]][nb[1]] is None:
                    visited.add(nb)
                    queue.append(nb)
        return False

    def choose_color():
        chosen = None
        chosen_moves = None
        for color in active_colors:
            if color in completed:
                continue
            current = progress[color][-1]
            target = color_pairs[color][1]
            moves = []
            for nb in neighbors(current):
                if nb == target:
                    moves.append(nb)
                elif grid[nb[0]][nb[1]] is None and nb not in progress[color]:
                    moves.append(nb)
            if not moves:
                return color, []
            moves.sort(key=lambda nb: manhattan(nb, target))
            if chosen is None or len(moves) < len(chosen_moves):
                chosen = color
                chosen_moves = moves
        return chosen, chosen_moves

    def search():
        if len(completed) == len(active_colors):
            for color in active_colors:
                solution_paths[color] = list(progress[color])
            return True
        color, moves = choose_color()
        if color is None or not moves:
            return False
        target = color_pairs[color][1]
        for move in moves:
            progress[color].append(move)
            reached_goal = move == target
            if not reached_goal:
                grid[move[0]][move[1]] = color
            else:
                completed.add(color)
            constraints_ok = True
            if reached_goal:
                pending = [c for c in active_colors if c not in completed]
            else:
                pending = [color] + [c for c in active_colors if c not in completed and c != color]
            for pending_color in pending:
                if not path_possible(pending_color):
                    constraints_ok = False
                    break
            if constraints_ok and search():
                return True
            if reached_goal:
                completed.discard(color)
            else:
                grid[move[0]][move[1]] = None
            progress[color].pop()
        return False
    success = search()
    if success:
        return solution_paths
    return None

def solve_paths_flexible(rows, cols, detected):
    strict_solution = solve_paths(rows, cols, detected)
    if strict_solution:
        return strict_solution
    print("Strict fill failed. Trying relaxed shortest-path mode...")
    endpoints = {}
    for r, c, color in detected:
        endpoints.setdefault(color, []).append((r, c))
    relaxed_solution = {}
    for color, pts in endpoints.items():
        if len(pts) != 2:
            continue
        start, end = pts
        queue = deque([[start]])
        visited = {start}
        found_path = None
        while queue:
            path = queue.popleft()
            cell = path[-1]
            if cell == end:
                found_path = path
                break
            r, c = cell
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(path + [(nr, nc)])
        if found_path:
            relaxed_solution[color] = found_path
    return relaxed_solution if relaxed_solution else None

def draw_paths(debug_img, paths, h_lines, v_lines, color_map):
    if not paths:
        return
    for color, cells in paths.items():
        if len(cells) < 2:
            continue
        color_bgr = color_map.get(color, (255, 255, 255))
        centers = []
        for r, c in cells:
            x1, x2 = v_lines[c], v_lines[c + 1]
            y1, y2 = h_lines[r], h_lines[r + 1]
            px = (x1 + x2) // 2
            py = (y1 + y2) // 2
            centers.append((px, py))
        for i in range(len(centers) - 1):
            cv2.line(debug_img, centers[i], centers[i + 1], color_bgr, 5)
            cv2.circle(debug_img, centers[i], 6, color_bgr, -1)
        cv2.circle(debug_img, centers[-1], 6, color_bgr, -1)

def cell_center(row, col, h_lines, v_lines, origin_x, origin_y):
    x1 = origin_x + v_lines[col]
    x2 = origin_x + v_lines[col + 1]
    y1 = origin_y + h_lines[row]
    y2 = origin_y + h_lines[row + 1]
    return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))

def cell_random_point(row, col, h_lines, v_lines, origin_x, origin_y, jitter=True):
    if not jitter:
        return cell_center(row, col, h_lines, v_lines, origin_x, origin_y)
    jitter_scale = max(0.0, JITTER_SCALE)
    if jitter_scale <= 0:
        return cell_center(row, col, h_lines, v_lines, origin_x, origin_y)
    x1 = origin_x + v_lines[col]
    x2 = origin_x + v_lines[col + 1]
    y1 = origin_y + h_lines[row]
    y2 = origin_y + h_lines[row + 1]
    span_x = max(1, x2 - x1)
    span_y = max(1, y2 - y1)
    base_pad_x = max(2, min(12, int(span_x * 0.3)))
    base_pad_y = max(2, min(12, int(span_y * 0.3)))
    pad_scale = max(0.1, min(3.0, jitter_scale))
    pad_x = max(2, min(span_x // 2, int(base_pad_x * pad_scale)))
    pad_y = max(2, min(span_y // 2, int(base_pad_y * pad_scale)))
    if span_x <= pad_x * 2:
        px = (x1 + x2) / 2
    else:
        px = random.uniform(x1 + pad_x, x2 - pad_x)
    if span_y <= pad_y * 2:
        py = (y1 + y2) / 2
    else:
        py = random.uniform(y1 + pad_y, y2 - pad_y)
    return int(round(px)), int(round(py))

def smoothstep(t):
    return t * t * (3 - 2 * t)

def order_paths_nearest_mouse(solutions, h_lines, v_lines, origin_x, origin_y):
    mx, my = get_cursor_pos()
    paths_with_dist = []
    for color, cells in solutions.items():
        if not cells:
            continue
        first_row, first_col = cells[0]
        last_row, last_col = cells[-1]
        first_px, first_py = cell_center(first_row, first_col, h_lines, v_lines, origin_x, origin_y)
        last_px, last_py = cell_center(last_row, last_col, h_lines, v_lines, origin_x, origin_y)
        dist_first = (first_px - mx) ** 2 + (first_py - my) ** 2
        dist_last = (last_px - mx) ** 2 + (last_py - my) ** 2
        if dist_first <= dist_last:
            dist, entry_index = dist_first, 0
        else:
            dist, entry_index = dist_last, -1
        paths_with_dist.append((dist, color, cells, entry_index))
    paths_with_dist.sort(key=lambda t: t[0])
    return [(color, cells, entry_index) for dist, color, cells, entry_index in paths_with_dist]

def path_pixel_centers(cells, h_lines, v_lines, origin_x, origin_y):
    centers = []
    for r, c in cells:
        x1, x2 = v_lines[c], v_lines[c+1]
        y1, y2 = h_lines[r], h_lines[r+1]
        centers.append((
            origin_x + (x1 + x2) // 2,
            origin_y + (y1 + y2) // 2
        ))
    return centers

def interpolate_segment(p1, p2, steps=12, curve=True):
    x1, y1 = p1
    x2, y2 = p2
    points = []
    for i in range(steps+1):
        t = i / steps
        if curve:
            t = t * t * (3 - 2 * t)
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        points.append((int(x), int(y)))
    return points

def drag_path(centers, hold=True):
    if not centers or len(centers) < 2:
        return
    smooth_move_to(centers[0], duration=0.15 * SPEED, jitter=False)
    scaled_sleep(0.01)
    if hold:
        mouse_down()
        scaled_sleep(0.005)
    for i in range(len(centers) - 1):
        segment = interpolate_segment(centers[i], centers[i+1], steps=10)
        for j, (px, py) in enumerate(segment):
            interception.move_to(px, py)
            scaled_sleep(0.0025 + random.uniform(-0.0005, 0.0005))
        if i < len(centers) - 2:
            dx1 = centers[i+1][0] - centers[i][0]
            dy1 = centers[i+1][1] - centers[i][1]
            dx2 = centers[i+2][0] - centers[i+1][0]
            dy2 = centers[i+2][1] - centers[i+1][1]
            if (dx1, dy1) != (dx2, dy2):
                scaled_sleep(random.uniform(0.02, 0.04))
        if random.random() < 0.15:
            scaled_sleep(random.uniform(0.01, 0.03))
    if hold:
        scaled_sleep(0.01)
        mouse_up()

def execute_single_path(cells, h_lines, v_lines, origin_x, origin_y):
    if len(cells) < 2:
        print("Path too short, skipping.")
        return
    centers = path_pixel_centers(cells, h_lines, v_lines, origin_x, origin_y)
    drag_path(centers)

def execute_all_paths(solutions, h_lines, v_lines, origin_x, origin_y):
    remaining = dict(solutions)
    while remaining:
        ordered = order_paths_nearest_mouse(remaining, h_lines, v_lines, origin_x, origin_y)
        if not ordered:
            break
        color, path, entry_index = ordered[0]
        if entry_index == -1:
            path = list(reversed(path))
        print(f"Executing path for {color}...")
        execute_single_path(path, h_lines, v_lines, origin_x, origin_y)
        del remaining[color]

def run_one_puzzle(return_debug=False, progress_callback=None):
    def _result(success, debug_image=None):
        if return_debug:
            return success, debug_image
        return success

    def _emit(stage, image):
        if progress_callback is None:
            return
        try:
            progress_callback(stage, image)
        except Exception:
            pass

    screenshot = capture_region()
    bbox = find_puzzle_bbox(screenshot)
    if not bbox:
        print("Puzzle not found.")
        _emit("no_puzzle", None)
        return _result(False, None)
    x, y, w, h = bbox
    puzzle = screenshot[y:y + h, x:x + w]
    _emit("puzzle_captured", puzzle.copy())
    origin_x = SEARCH_REGION["left"] + x
    origin_y = SEARCH_REGION["top"] + y
    rows, cols, h_lines, v_lines, debug = detect_grid(puzzle)
    debug_image = debug
    _emit("grid_detected", debug.copy())
    if rows <= 0 or cols <= 0:
        print("Grid detection failed.")
        _emit("grid_failed", debug.copy())
        return _result(False, debug_image)
    detected, color_map = detect_points(puzzle, rows, cols, h_lines, v_lines, debug)
    debug_image = debug
    _emit("points_detected", debug.copy())
    print("\nDetected points:")
    for d in detected:
        print(f"Row {d[0]}, Col {d[1]} -> {d[2]}")
    print(f"Total unique paths needed: {len(color_map)}")
    solutions = solve_paths_flexible(rows, cols, detected)
    if solutions:
        preview = debug.copy()
        draw_paths(preview, solutions, h_lines, v_lines, color_map)
        debug_image = preview.copy()
        _emit("paths_ready", preview.copy())
        for color, path in solutions.items():
            print(f"Path for {color}: {path}")
        execute_all_paths(solutions, h_lines, v_lines, origin_x, origin_y)
    else:
        print("Solver could not find a valid set of paths.")
        _emit("no_solution", debug.copy())
    print("Puzzle Complete/Canceled!")
    return _result(True, debug_image)

def main():
    print("Ready. Press PgDn for one puzzle, PgUp for continuous mode. Press end to quit.")
    while True:
        key = keyboard.read_event(suppress=False).name
        if key == "page down":
            print("Running single puzzle...")
            run_one_puzzle()
        elif key == "page up":
            print("Running continuous mode...")
            while True:
                success = run_one_puzzle()
                if not success:
                    print("No puzzle detected, stopping continuous mode.")
                    print("Generator Complete/Canceled!")
                    break
                scaled_sleep(0.35)
                if keyboard.is_pressed("end"):
                    print("end pressed, stopping continuous mode.")
                    return
        elif key == "end":
            print("Exiting script.")
            break

if __name__ == "__main__":
    main()

import ctypes
import json
import math
import threading
import time
from pathlib import Path
import interception
import keyboard
import mss
import numpy as np
interception.auto_capture_devices(keyboard=True, mouse=True)
_mss_local = threading.local()

START_KEY = "insert"
KILL_KEY = "delete"
AUTO_KEY = "home"
CLOCK = time.perf_counter
SLEEP_POLL_INTERVAL = 0.01
try:
    BASE_DIR = Path(__compiled__.containing_dir) # type: ignore
except NameError:
    BASE_DIR = Path(__file__).resolve().parent
TIMING_CONFIG_PATH = BASE_DIR / "DBFGenBot.timing.json"
DEFAULT_TIMING = {
    "wire_speed": 4.0,
    "cord_speed": 1.1,
    "wire_grab_delay": 0.1,
    "cord_grab_delay": 0.1,
}

def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

def _load_timing_settings(path: Path):
    settings = dict(DEFAULT_TIMING)
    if not path.exists():
        return settings
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in DEFAULT_TIMING:
                if key in payload:
                    settings[key] = payload[key]
        else:
            print(f"[WARN] Timing config {path.name} is not a JSON object. Using defaults.", flush=True)
    except Exception as exc:
        print(f"[WARN] Failed to load timing config {path.name}: {exc!r}. Using defaults.", flush=True)
    return settings

TIMING = _load_timing_settings(TIMING_CONFIG_PATH)
WIRE_SPEED = max(0.1, _safe_float(TIMING.get("wire_speed"), DEFAULT_TIMING["wire_speed"]))
CORD_SPEED = max(0.1, _safe_float(TIMING.get("cord_speed"), DEFAULT_TIMING["cord_speed"]))
WIRE_GRAB_DELAY = max(0.0, _safe_float(TIMING.get("wire_grab_delay"), DEFAULT_TIMING["wire_grab_delay"]))
CORD_GRAB_DELAY = max(0.0, _safe_float(TIMING.get("cord_grab_delay"), DEFAULT_TIMING["cord_grab_delay"]))

def _speed_factor(value) -> float:
    return 1.0 / max(0.1, float(value))

def _wire_scale(value: float) -> float:
    return float(value) * _speed_factor(WIRE_SPEED)

def _cord_scale(value: float) -> float:
    return float(value) * _speed_factor(CORD_SPEED)

COLOR_TOL = 1
ALLOWED_COLORS = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]
COLOR_MATCH_MAX_DIST = 170
COLOR_MATCH_MAX_DIST_SQ = COLOR_MATCH_MAX_DIST * COLOR_MATCH_MAX_DIST

WIRE_BOX_Y_OFFSET = 10
TOP_BOX = (507, 284 + WIRE_BOX_Y_OFFSET, 1412, 378 + WIRE_BOX_Y_OFFSET)
BOT_BOX = (507, 699 + WIRE_BOX_Y_OFFSET, 1412, 794 + WIRE_BOX_Y_OFFSET)
CORD_BOX = (1459, 565, 1536, 583)

SCAN_STRIP_H = 5
VALID_SUM_MIN = 30
MIN_SEG_W = 10
MERGE_GAP_PX = 12
CONSISTENCY_ATTEMPTS = 4
CONSISTENCY_DELAY = 0.02
CORD_CONSISTENCY_ATTEMPTS = 3
CORD_CONSISTENCY_DELAY = 0.005
CORD_CHECK_MIN_INTERVAL = 0.08
REPEAT_SIG_COOLDOWN = 0.75
IDLE_RESCAN_DELAY = 0.08
IDLE_STATUS_INTERVAL = 2.0
AUTO_MIN_WIRES = 4
AUTO_CONSISTENCY_ATTEMPTS = 3
AUTO_CONSISTENCY_DELAY = 0.03
AUTO_SCAN_INTERVAL = 0.06
WIRE_INITIAL_DETECT_RETRY_DELAY = 0.03
WIRE_RELEASE_OFFSET_Y = 26
WIRE_APPROACH_STEPS = 18
WIRE_DRAG_STEPS = 22
WIRE_RELEASE_STEPS = 6
WIRE_HOVER_DELAY = 0.01
POST_WIRE_DELAY = 0.06
WIRE_VERIFY_RETRIES = 3
WIRE_VERIFY_POST_DRAG_DELAY = 0.03
WIRE_VERIFY_RETRY_DELAY = 0.04
WIRE_VERIFY_SCAN_ATTEMPTS = 2
WIRE_VERIFY_SCAN_DELAY = 0.02
WIRE_VERIFY_SAMPLE_SPACING = 12.0
WIRE_VERIFY_SEARCH_RADIUS = 4
WIRE_VERIFY_MARGIN = 10
WIRE_VERIFY_PROBE_NORMAL_OFFSET_INNER = 5.0
WIRE_VERIFY_PROBE_NORMAL_OFFSET_OUTER = 9.0
WIRE_VERIFY_PROBE_ALONG_OFFSET = 4.0
WIRE_VERIFY_START_RANGE = (0.14, 0.38)
WIRE_VERIFY_END_RANGE = (0.62, 0.86)
WIRE_VERIFY_MID_SKIP_RANGE = (0.44, 0.56)
WIRE_VERIFY_ZONE_MATCH_RATIO = 0.55
WIRE_VERIFY_BODY_MATCH_RATIO = 0.48
WIRE_VERIFY_ZONE_MIN_STREAK = 3
CORD_GRAB_OFFSET_Y = 6
CORD_APPROACH_STEPS = 20
CORD_DRAG_STEPS = 16
CORD_PULL_COUNT = 7
CORD_HOLD_DELAY = 0.08
CORD_NUDGE_STEPS = 6
CORD_BETWEEN_PULLS = 0.1
POST_COMPLETE_RESCAN_DELAY = 0.25
PRINT_DETECTED_WIRES = False

kill_switch_event = threading.Event()
run_lock = threading.Lock()
run_in_progress = False
auto_mode_event = threading.Event()

class KillSwitchActivated(Exception):
    pass

class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def get_sct():
    sct = getattr(_mss_local, "sct", None)
    if sct is None:
        _mss_local.sct = mss.mss()
        sct = _mss_local.sct
    return sct

def get_cursor_pos():
    pt = _POINT()
    try:
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return int(pt.x), int(pt.y)
    except Exception:
        pass
    return None

def abort_if_killed():
    if kill_switch_event.is_set():
        raise KillSwitchActivated

def sleep_until(deadline: float, *, abortable: bool = True):
    while True:
        if abortable:
            abort_if_killed()
        remaining = deadline - CLOCK()
        if remaining <= 0:
            return
        time.sleep(min(SLEEP_POLL_INTERVAL, remaining))

def sleep_seconds(duration: float, *, abortable: bool = True):
    duration = max(0.0, _safe_float(duration, 0.0))
    if duration <= 0:
        if abortable:
            abort_if_killed()
        return
    sleep_until(CLOCK() + duration, abortable=abortable)

def sleep_with_abort(duration: float):
    sleep_seconds(duration, abortable=True)

def engage_kill_switch():
    if kill_switch_event.is_set():
        return
    kill_switch_event.set()
    try:
        interception.mouse_up(button="left")
        interception.mouse_up(button="right")
    except Exception:
        pass
    print("[KILL] Kill switch activated. Stopping now.", flush=True)

def reset_kill_switch():
    if kill_switch_event.is_set():
        kill_switch_event.clear()

def toggle_auto_mode():
    if auto_mode_event.is_set():
        auto_mode_event.clear()
        print("[AUTO] Auto mode OFF.", flush=True)
    else:
        auto_mode_event.set()
        print("[AUTO] Auto mode ON.", flush=True)

def color_close(a, b, tol=COLOR_TOL):
    return (
        abs(a[0] - b[0]) <= tol
        and abs(a[1] - b[1]) <= tol
        and abs(a[2] - b[2]) <= tol
    )

def _nearest_allowed_color(rgb):
    r = int(rgb[0])
    g = int(rgb[1])
    b = int(rgb[2])
    best_idx = None
    best_dist2 = None
    for i, c in enumerate(ALLOWED_COLORS):
        dr = r - c[0]
        dg = g - c[1]
        db = b - c[2]
        dist2 = (dr * dr) + (dg * dg) + (db * db)
        if best_dist2 is None or dist2 < best_dist2:
            best_dist2 = dist2
            best_idx = i
    if best_dist2 is None or best_dist2 > COLOR_MATCH_MAX_DIST_SQ:
        return None
    return best_idx

def _ease(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)

def _bool_ratio(values) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v) / float(len(values))

def _max_true_streak(values) -> int:
    best = 0
    streak = 0
    for value in values:
        if value:
            streak += 1
            if streak > best:
                best = streak
        else:
            streak = 0
    return best

def _line_sample_matches_color(rgb, probe_points, target_idx: int, search_radius: int) -> bool:
    h, w = rgb.shape[:2]
    for probe_x, probe_y in probe_points:
        x0 = max(0, probe_x - search_radius)
        x1 = min(w, probe_x + search_radius + 1)
        y0 = max(0, probe_y - search_radius)
        y1 = min(h, probe_y + search_radius + 1)
        for y in range(y0, y1):
            for x in range(x0, x1):
                pixel = rgb[y, x]
                if int(pixel[0]) + int(pixel[1]) + int(pixel[2]) < VALID_SUM_MIN:
                    continue
                if _nearest_allowed_color(pixel) == target_idx:
                    return True
    return False

def _verify_wire_connection_once(start, end, target_color):
    try:
        target_idx = ALLOWED_COLORS.index(target_color)
    except ValueError:
        target_idx = _nearest_allowed_color(target_color)
    if target_idx is None:
        return False, {"start": (0, 0), "body": (0, 0), "end": (0, 0)}
    sx = float(start[0])
    sy = float(start[1])
    ex = float(end[0])
    ey = float(end[1])
    dist = math.hypot(ex - sx, ey - sy)
    if dist <= 0:
        return False, {"start": (0, 0), "body": (0, 0), "end": (0, 0)}
    dir_x = (ex - sx) / dist
    dir_y = (ey - sy) / dist
    normal_x = -dir_y
    normal_y = dir_x
    sample_count = max(18, int(dist / WIRE_VERIFY_SAMPLE_SPACING))
    pad = int(math.ceil(
        WIRE_VERIFY_MARGIN
        + WIRE_VERIFY_SEARCH_RADIUS
        + WIRE_VERIFY_PROBE_NORMAL_OFFSET_OUTER
        + WIRE_VERIFY_PROBE_ALONG_OFFSET
    ))
    left = int(math.floor(min(sx, ex) - pad))
    top = int(math.floor(min(sy, ey) - pad))
    right = int(math.ceil(max(sx, ex) + pad))
    bottom = int(math.ceil(max(sy, ey) + pad))
    width = max(1, right - left)
    height = max(1, bottom - top)
    img = get_sct().grab({"left": left, "top": top, "width": width, "height": height})
    arr = np.asarray(img)
    rgb = arr[:, :, :3][:, :, ::-1].astype(np.int16)
    start_flags = []
    body_flags = []
    end_flags = []
    for i in range(sample_count + 1):
        abort_if_killed()
        t = i / sample_count
        x = int(round(sx + ((ex - sx) * t))) - left
        y = int(round(sy + ((ey - sy) * t))) - top
        probe_points = []
        probe_offsets = (
            (0.0, 0.0),
            (0.0, WIRE_VERIFY_PROBE_NORMAL_OFFSET_INNER),
            (0.0, -WIRE_VERIFY_PROBE_NORMAL_OFFSET_INNER),
            (0.0, WIRE_VERIFY_PROBE_NORMAL_OFFSET_OUTER),
            (0.0, -WIRE_VERIFY_PROBE_NORMAL_OFFSET_OUTER),
            (WIRE_VERIFY_PROBE_ALONG_OFFSET, 0.0),
            (-WIRE_VERIFY_PROBE_ALONG_OFFSET, 0.0),
        )
        for along_offset, normal_offset in probe_offsets:
            probe_x = int(round(
                x
                + (dir_x * along_offset)
                + (normal_x * normal_offset)
            ))
            probe_y = int(round(
                y
                + (dir_y * along_offset)
                + (normal_y * normal_offset)
            ))
            probe_point = (probe_x, probe_y)
            if probe_point not in probe_points:
                probe_points.append(probe_point)
        matched = _line_sample_matches_color(rgb, probe_points, target_idx, WIRE_VERIFY_SEARCH_RADIUS)
        if WIRE_VERIFY_START_RANGE[0] <= t <= WIRE_VERIFY_START_RANGE[1]:
            start_flags.append(matched)
        if WIRE_VERIFY_END_RANGE[0] <= t <= WIRE_VERIFY_END_RANGE[1]:
            end_flags.append(matched)
        if WIRE_VERIFY_START_RANGE[0] <= t <= WIRE_VERIFY_END_RANGE[1]:
            if not (WIRE_VERIFY_MID_SKIP_RANGE[0] <= t <= WIRE_VERIFY_MID_SKIP_RANGE[1]):
                body_flags.append(matched)
    start_ratio = _bool_ratio(start_flags)
    body_ratio = _bool_ratio(body_flags)
    end_ratio = _bool_ratio(end_flags)
    start_streak = _max_true_streak(start_flags)
    end_streak = _max_true_streak(end_flags)
    min_start_streak = min(WIRE_VERIFY_ZONE_MIN_STREAK, len(start_flags))
    min_end_streak = min(WIRE_VERIFY_ZONE_MIN_STREAK, len(end_flags))
    verified = (
        start_ratio >= WIRE_VERIFY_ZONE_MATCH_RATIO
        and end_ratio >= WIRE_VERIFY_ZONE_MATCH_RATIO
        and body_ratio >= WIRE_VERIFY_BODY_MATCH_RATIO
        and start_streak >= min_start_streak
        and end_streak >= min_end_streak
    )
    stats = {
        "start": (sum(1 for v in start_flags if v), len(start_flags)),
        "body": (sum(1 for v in body_flags if v), len(body_flags)),
        "end": (sum(1 for v in end_flags if v), len(end_flags)),
    }
    return verified, stats

def verify_wire_connection(start, end, target_color, attempts=WIRE_VERIFY_SCAN_ATTEMPTS, delay=WIRE_VERIFY_SCAN_DELAY):
    last_stats = {"start": (0, 0), "body": (0, 0), "end": (0, 0)}
    total_attempts = max(1, int(attempts))
    for attempt_idx in range(total_attempts):
        verified, stats = _verify_wire_connection_once(start, end, target_color)
        last_stats = stats
        if verified:
            return True, stats
        if attempt_idx + 1 < total_attempts:
            sleep_with_abort(delay)
    return False, last_stats

def smooth_move_from_to(start, end, steps=20, delay=0.003):
    steps = max(1, int(steps))
    delay = max(0.0, float(delay))
    sx, sy = start
    ex, ey = end
    next_tick = CLOCK()
    for i in range(1, steps + 1):
        abort_if_killed()
        t = _ease(i / steps)
        x = int(sx + (ex - sx) * t)
        y = int(sy + (ey - sy) * t)
        interception.move_to(x, y)
        if delay > 0:
            next_tick += delay
            sleep_until(next_tick, abortable=True)

def smooth_move_to(target, steps=20, delay=0.003):
    abort_if_killed()
    cur = get_cursor_pos()
    if cur is None:
        interception.move_to(int(target[0]), int(target[1]))
        return
    smooth_move_from_to(cur, target, steps=steps, delay=delay)

def move_to_and_wait(target, steps, delay, settle_delay=0.0):
    smooth_move_to(target, steps=steps, delay=delay)
    if settle_delay > 0:
        sleep_with_abort(settle_delay)

def smooth_drag(start, end):
    abort_if_killed()
    move_to_and_wait(
        start,
        steps=WIRE_APPROACH_STEPS,
        delay=_wire_scale(0.0025),
        settle_delay=WIRE_GRAB_DELAY,
    )
    abort_if_killed()
    interception.mouse_down(button="left")
    sleep_with_abort(_wire_scale(0.015))
    release_y = max(int(end[1]) + WIRE_RELEASE_OFFSET_Y, BOT_BOX[3] + 6)
    release_target = (int(end[0]), int(release_y))
    try:
        smooth_move_from_to(start, end, steps=WIRE_DRAG_STEPS, delay=_wire_scale(0.0035))
        sleep_with_abort(_wire_scale(WIRE_HOVER_DELAY))
        # The generator now auto-connects on hover, so release below the target
        # instead of dropping directly on the bottom socket.
        smooth_move_from_to(end, release_target, steps=WIRE_RELEASE_STEPS, delay=_wire_scale(0.0022))
    finally:
        interception.mouse_up(button="left")
        sleep_with_abort(_wire_scale(0.02))

def _quant_pos(p, q=8):
    return (int(p[0] // q), int(p[1] // q))

def _color_key(c, step=32):
    return (int(c[0] // step), int(c[1] // step), int(c[2] // step))

def _color_signature(wires):
    counts = [0] * len(ALLOWED_COLORS)
    for w in wires:
        idx = _nearest_allowed_color(w["color"])
        if idx is not None:
            counts[idx] += 1
    return tuple(counts)

def _puzzle_signature(top_wires, bot_wires):
    return (_color_signature(top_wires), _color_signature(bot_wires))

def _match_count(top_wires, bot_wires):
    top_counts = _color_signature(top_wires)
    bot_counts = _color_signature(bot_wires)
    total = 0
    for i in range(len(ALLOWED_COLORS)):
        total += min(top_counts[i], bot_counts[i])
    return total

def _has_duplicate_colors(wires):
    counts = _color_signature(wires)
    return any(count > 1 for count in counts)

def _is_valid_auto_puzzle(top_wires, bot_wires, min_wires=AUTO_MIN_WIRES):
    if not top_wires or not bot_wires:
        return False
    if len(top_wires) < min_wires or len(bot_wires) < min_wires:
        return False
    if len(top_wires) != len(bot_wires):
        return False
    if _has_duplicate_colors(top_wires) or _has_duplicate_colors(bot_wires):
        return False
    return _color_signature(top_wires) == _color_signature(bot_wires)

def _signature(top_wires, bot_wires):
    top_sig = tuple(sorted((w["pos"][0] // 8, _color_key(w["color"])) for w in top_wires))
    bot_sig = tuple(sorted((w["pos"][0] // 8, _color_key(w["color"])) for w in bot_wires))
    return (top_sig, bot_sig)

def scan_wire_band_fast(box, strip_h=SCAN_STRIP_H):
    abort_if_killed()
    x1, y1, x2, y2 = box
    w = x2 - x1
    y = (y1 + y2) // 2
    top = y - (strip_h // 2)
    img = get_sct().grab({"left": x1, "top": top, "width": w, "height": strip_h})
    arr = np.asarray(img)
    rgb = arr[:, :, :3][:, :, ::-1].astype(np.int16)
    min_hits = max(1, strip_h // 2)
    color_idx = np.full(w, -1, dtype=np.int16)
    for x in range(w):
        counts = [0] * len(ALLOWED_COLORS)
        for yi in range(strip_h):
            pixel = rgb[yi, x]
            if int(pixel[0]) + int(pixel[1]) + int(pixel[2]) < VALID_SUM_MIN:
                continue
            idx = _nearest_allowed_color(pixel)
            if idx is not None:
                counts[idx] += 1
        best_idx, best_count = max(enumerate(counts), key=lambda item: item[1])
        if best_count >= min_hits:
            color_idx[x] = best_idx
    segments = []
    x = 0
    while x < w:
        idx = int(color_idx[x])
        if idx < 0:
            x += 1
            continue
        start = x
        x += 1
        while x < w and color_idx[x] == idx:
            x += 1
        end = x
        segments.append([start, end, idx])
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        gap = seg[0] - prev[1]
        if seg[2] == prev[2] and gap <= MERGE_GAP_PX:
            prev[1] = seg[1]
        else:
            merged.append(seg)
    wires = []
    for start, end, idx in merged:
        if end - start < MIN_SEG_W:
            continue
        cx = x1 + (start + end - 1) // 2
        wires.append({"pos": (int(cx), int(y)), "color": ALLOWED_COLORS[idx]})
    return wires

def scan_puzzle_consistent(expected_sig, attempts=CONSISTENCY_ATTEMPTS, delay=CONSISTENCY_DELAY):
    last_top = None
    last_bot = None
    total_attempts = max(1, int(attempts))
    for attempt_idx in range(total_attempts):
        top = scan_wire_band_fast(TOP_BOX)
        bot = scan_wire_band_fast(BOT_BOX)
        if top and bot:
            last_top = top
            last_bot = bot
            if expected_sig is None:
                return top, bot, True
            if _puzzle_signature(top, bot) == expected_sig:
                return top, bot, True
        if attempt_idx + 1 < total_attempts:
            sleep_with_abort(delay)
    return last_top, last_bot, False

def detect_puzzle_stable(min_wires=AUTO_MIN_WIRES, attempts=AUTO_CONSISTENCY_ATTEMPTS, delay=AUTO_CONSISTENCY_DELAY):
    expected_sig = None
    stable = 0
    total_attempts = max(1, int(attempts))
    for attempt_idx in range(total_attempts):
        top = scan_wire_band_fast(TOP_BOX)
        bot = scan_wire_band_fast(BOT_BOX)
        if _is_valid_auto_puzzle(top, bot, min_wires=min_wires):
            sig = _puzzle_signature(top, bot)
            if expected_sig is None or sig == expected_sig:
                expected_sig = sig if expected_sig is None else expected_sig
                stable += 1
            else:
                expected_sig = sig
                stable = 1
            if stable >= total_attempts:
                return True
        else:
            expected_sig = None
            stable = 0
        if attempt_idx + 1 < total_attempts:
            sleep_with_abort(delay)
    return False

def detect_puzzle():
    try:
        top = scan_wire_band_fast(TOP_BOX)
        bot = scan_wire_band_fast(BOT_BOX)
    except KillSwitchActivated:
        raise
    if not top or not bot:
        return None, None, None
    sig = _signature(top, bot)
    return top, bot, sig

def solve_wires():
    top_init = None
    bot_init = None
    for attempt_idx in range(12):
        abort_if_killed()
        top_init = scan_wire_band_fast(TOP_BOX)
        bot_init = scan_wire_band_fast(BOT_BOX)
        if top_init and bot_init:
            break
        if attempt_idx < 11:
            sleep_with_abort(WIRE_INITIAL_DETECT_RETRY_DELAY)
    if not top_init or not bot_init:
        print("No wires detected (puzzle not open?)", flush=True)
        return None
    expected_sig = _puzzle_signature(top_init, bot_init)
    targets = top_init[:]
    solved_targets = set()
    used_bottom = set()
    used_top = set()
    if PRINT_DETECTED_WIRES:
        print("TOP WIRES:", flush=True)
        for w in top_init:
            print(w, flush=True)
        print("BOTTOM WIRES:", flush=True)
        for w in bot_init:
            print(w, flush=True)
    print(f"Detected {len(targets)} wires.", flush=True)
    max_loops = 250
    loops = 0
    while len(solved_targets) < len(targets) and loops < max_loops:
        abort_if_killed()
        loops += 1
        top_now, bot_now, ok = scan_puzzle_consistent(expected_sig)
        if not ok or not top_now or not bot_now:
            print("Puzzle lost or moved away. Stopping.", flush=True)
            return None
        progress = False
        for ti, tw in enumerate(targets):
            if ti in solved_targets:
                continue
            target_color = tw["color"]
            target_x = tw["pos"][0]
            top_candidates = []
            for cur in top_now:
                if color_close(cur["color"], target_color):
                    tq = _quant_pos(cur["pos"], q=8)
                    if tq in used_top:
                        continue
                    top_candidates.append(cur)
            if not top_candidates:
                continue
            top_pick = min(top_candidates, key=lambda w: abs(w["pos"][0] - target_x))
            tp = top_pick["pos"]
            bot_candidates = []
            for cur in bot_now:
                if color_close(cur["color"], target_color):
                    bq = _quant_pos(cur["pos"], q=8)
                    if bq in used_bottom:
                        continue
                    bot_candidates.append(cur)
            if not bot_candidates:
                continue
            bot_pick = min(bot_candidates, key=lambda w: abs(w["pos"][0] - tp[0]))
            bp = bot_pick["pos"]
            print(f"Wire match {target_color}: {tp} -> {bp}", flush=True)
            verified = False
            verify_stats = {"start": (0, 0), "body": (0, 0), "end": (0, 0)}
            for drag_attempt in range(1, WIRE_VERIFY_RETRIES + 1):
                smooth_drag(tp, bp)
                sleep_with_abort(WIRE_VERIFY_POST_DRAG_DELAY)
                verified, verify_stats = verify_wire_connection(tp, bp, target_color)
                if verified:
                    break
                start_hits, start_total = verify_stats["start"]
                body_hits, body_total = verify_stats["body"]
                end_hits, end_total = verify_stats["end"]
                print(
                    f"[VERIFY] Wire not confirmed after attempt {drag_attempt}/{WIRE_VERIFY_RETRIES} "
                    f"(start {start_hits}/{start_total}, body {body_hits}/{body_total}, end {end_hits}/{end_total}).",
                    flush=True,
                )
                if drag_attempt < WIRE_VERIFY_RETRIES:
                    sleep_with_abort(WIRE_VERIFY_RETRY_DELAY)
            if not verified:
                print("Wire verification failed. Stopping.", flush=True)
                return None
            used_top.add(_quant_pos(tp, q=8))
            used_bottom.add(_quant_pos(bp, q=8))
            solved_targets.add(ti)
            sleep_with_abort(_wire_scale(POST_WIRE_DELAY))
            progress = True
            break
        if not progress:
            print(f"No more matches found. Solved {len(solved_targets)}/{len(targets)}.", flush=True)
            break
    solved_ok = len(solved_targets) == len(targets)
    print(f"Wires solved: {len(solved_targets)}/{len(targets)}", flush=True)
    if not solved_ok:
        return None
    return expected_sig

def pull_cord(expected_sig=None):
    abort_if_killed()
    cx = (CORD_BOX[0] + CORD_BOX[2]) // 2
    cy = (CORD_BOX[1] + CORD_BOX[3]) // 2
    grab_y = cy + CORD_GRAB_OFFSET_Y
    grab = (cx, grab_y)
    up = (cx, grab_y - 300)
    down = (cx, grab_y + 300)
    last_check = 0.0

    def puzzle_ok():
        nonlocal last_check
        if expected_sig is None:
            return True
        now = CLOCK()
        if (now - last_check) < CORD_CHECK_MIN_INTERVAL:
            return True
        last_check = now
        _, _, ok = scan_puzzle_consistent(
            expected_sig,
            attempts=CORD_CONSISTENCY_ATTEMPTS,
            delay=CORD_CONSISTENCY_DELAY,
        )
        if not ok:
            print("Puzzle lost or moved away. Stopping.", flush=True)
        return ok

    if not puzzle_ok():
        return False
    move_to_and_wait(
        grab,
        steps=CORD_APPROACH_STEPS,
        delay=_cord_scale(0.0025),
        settle_delay=CORD_GRAB_DELAY,
    )
    abort_if_killed()
    interception.mouse_down(button="left")
    sleep_with_abort(_cord_scale(CORD_HOLD_DELAY))
    try:
        smooth_move_from_to(grab, (cx, grab_y + 12), steps=CORD_NUDGE_STEPS, delay=_cord_scale(0.0025))
        sleep_with_abort(_cord_scale(0.02))
        for i in range(CORD_PULL_COUNT):
            abort_if_killed()
            if not puzzle_ok():
                return False
            print(f"Pulling cord {i + 1}/{CORD_PULL_COUNT}", flush=True)
            cur = get_cursor_pos() or grab
            smooth_move_from_to(cur, down, steps=CORD_DRAG_STEPS, delay=_cord_scale(0.0030))
            sleep_with_abort(_cord_scale(CORD_BETWEEN_PULLS))
            if not puzzle_ok():
                return False
            smooth_move_from_to(down, up, steps=CORD_DRAG_STEPS, delay=_cord_scale(0.0030))
            sleep_with_abort(_cord_scale(CORD_BETWEEN_PULLS))
        return True
    finally:
        try:
            interception.mouse_up(button="left")
        except Exception:
            pass
        sleep_with_abort(_cord_scale(0.05))

def run_generators_in_a_row():
    last_sig = None
    last_complete_time = 0.0
    last_idle_note = 0.0
    while True:
        abort_if_killed()
        top, bot, sig = detect_puzzle()
        if sig is None:
            now = CLOCK()
            if (now - last_idle_note) >= IDLE_STATUS_INTERVAL:
                print("Waiting for puzzle...", flush=True)
                last_idle_note = now
            sleep_with_abort(IDLE_RESCAN_DELAY)
            continue
        if last_sig is not None and sig == last_sig:
            if (CLOCK() - last_complete_time) < REPEAT_SIG_COOLDOWN:
                sleep_with_abort(0.05)
                continue
        print("Starting generator solve...", flush=True)
        expected_sig = solve_wires()
        if expected_sig is None:
            print("Wire stage incomplete (stopping).", flush=True)
            return
        cord_ok = pull_cord(expected_sig)
        if not cord_ok:
            print("Cord stage incomplete (stopping).", flush=True)
            return
        print("Generator completed.", flush=True)
        last_sig = sig
        last_complete_time = CLOCK()
        sleep_with_abort(_cord_scale(POST_COMPLETE_RESCAN_DELAY))

def auto_loop():
    while True:
        time.sleep(AUTO_SCAN_INTERVAL)
        if not auto_mode_event.is_set():
            continue
        if kill_switch_event.is_set():
            continue
        with run_lock:
            running = run_in_progress
        if running:
            continue
        try:
            if detect_puzzle_stable():
                start_hotkey()
        except KillSwitchActivated:
            continue

def start_hotkey():
    global run_in_progress
    with run_lock:
        if run_in_progress:
            print("[INFO] Already running.", flush=True)
            return
        run_in_progress = True
    reset_kill_switch()

    def worker():
        global run_in_progress
        try:
            run_generators_in_a_row()
        except KillSwitchActivated:
            print("[INFO] Run aborted.", flush=True)
        except Exception as e:
            print(f"[ERROR] {e!r}", flush=True)
        finally:
            sct = getattr(_mss_local, "sct", None)
            if sct is not None:
                try:
                    sct.close()
                except Exception:
                    pass
                try:
                    delattr(_mss_local, "sct")
                except Exception:
                    pass
            try:
                interception.mouse_up(button="left")
                interception.mouse_up(button="right")
            except Exception:
                pass
            with run_lock:
                run_in_progress = False
            print("[INFO] Stopped.", flush=True)

    threading.Thread(target=worker, daemon=True).start()

def main():
    print("Ready.", flush=True)
    if TIMING_CONFIG_PATH.exists():
        print(f"Timing config: {TIMING_CONFIG_PATH.name}", flush=True)
    print(
        f"Press {START_KEY.upper()} to start. "
        f"Press {AUTO_KEY.upper()} to toggle auto. "
        f"Press {KILL_KEY.upper()} to kill.",
        flush=True,
    )
    keyboard.add_hotkey(START_KEY, start_hotkey, suppress=False)
    keyboard.add_hotkey(AUTO_KEY, toggle_auto_mode, suppress=False)
    keyboard.add_hotkey(KILL_KEY, engage_kill_switch, suppress=False)
    threading.Thread(target=auto_loop, daemon=True).start()
    while True:
        time.sleep(0.25)

if __name__ == "__main__":
    main()

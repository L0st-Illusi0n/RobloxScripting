# type: ignore[import]
import ctypes
import cv2
import numpy as np
import pyautogui
import keyboard
from collections import deque
from contextlib import contextmanager, nullcontext
from threading import Event, Thread, RLock
from typing import Optional
import interception
import time, random
interception.auto_capture_devices(keyboard=True, mouse=True)
KILL_SWITCH_KEY = "delete"
class KillSwitchActivated(Exception):
    pass

class PuzzleNotAvailable(Exception):
    pass

class MouseBlockedAtStart(Exception):
    def __init__(self, position):
        normalized = (int(position[0]), int(position[1]))
        self.position = normalized
        super().__init__(f"Mouse at blocked start position {normalized}.")

class MouseMovementStalled(Exception):
    pass

kill_switch_event = Event()
_kill_switch_handle = None
_puzzle_monitor = None
_mouse_monitor = None

def engage_kill_switch():
    if kill_switch_event.is_set():
        return
    kill_switch_event.set()
    try:
        interception.mouse_up(button="left")
        interception.mouse_up(button="right")
    except Exception:
        pass
    print(f"Kill switch '{KILL_SWITCH_KEY}' engaged. Stopping mouse input.")

def reset_kill_switch():
    if kill_switch_event.is_set():
        kill_switch_event.clear()

def set_puzzle_monitor(monitor):
    global _puzzle_monitor
    _puzzle_monitor = monitor

def clear_puzzle_monitor():
    global _puzzle_monitor
    _puzzle_monitor = None

def set_mouse_monitor(monitor):
    global _mouse_monitor
    _mouse_monitor = monitor

def clear_mouse_monitor():
    global _mouse_monitor
    _mouse_monitor = None

def get_mouse_monitor():
    return _mouse_monitor

def get_puzzle_monitor():
    return _puzzle_monitor

def abort_if_killed():
    if kill_switch_event.is_set():
        raise KillSwitchActivated
    monitor = _puzzle_monitor
    if monitor is not None:
        monitor.ensure_present()
    movement_monitor = _mouse_monitor
    if movement_monitor is not None:
        movement_monitor.ensure_active()

def _register_kill_switch_hotkey():
    global _kill_switch_handle
    if _kill_switch_handle is not None:
        try:
            keyboard.remove_hotkey(_kill_switch_handle)
        except KeyError:
            pass
        _kill_switch_handle = None
    try:
        _kill_switch_handle = keyboard.add_hotkey(
            KILL_SWITCH_KEY,
            engage_kill_switch,
            suppress=False,
        )
    except Exception as exc:
        _kill_switch_handle = None
        print(f"Failed to register kill switch hotkey '{KILL_SWITCH_KEY}': {exc}")

def set_kill_switch_key(hotkey):
    global KILL_SWITCH_KEY
    normalized = hotkey.strip().lower()
    if not normalized:
        raise ValueError("Kill switch hotkey cannot be empty.")
    KILL_SWITCH_KEY = normalized
    _register_kill_switch_hotkey()

def get_kill_switch_key():
    return KILL_SWITCH_KEY

_register_kill_switch_hotkey()
SPEED = 1.35
JITTER_SCALE = 0.8
SMOOTHNESS = 0.75
STEPS = 5
AUTO_IDLE_DELAY = 0.5
MOUSE_STALL_TIMEOUT = 10
BLOCKED_CURSOR_POSITIONS = {
    (960, 540),
    (960, 527),
}
_blocked_cursor_lock = RLock()
_blocked_cursor_whitelist_stack = []
@contextmanager
def _whitelist_blocked_positions(positions):
    normalized = {
        (int(pos[0]), int(pos[1]))
        for pos in positions
        if isinstance(pos, (tuple, list)) and len(pos) == 2
    }
    if not normalized:
        yield
        return
    with _blocked_cursor_lock:
        _blocked_cursor_whitelist_stack.append(normalized)
    try:
        yield
    finally:
        with _blocked_cursor_lock:
            if _blocked_cursor_whitelist_stack and _blocked_cursor_whitelist_stack[-1] is normalized:
                _blocked_cursor_whitelist_stack.pop()
            else:
                try:
                    _blocked_cursor_whitelist_stack.remove(normalized)
                except ValueError:
                    pass

def _is_blocked_position_whitelisted(position):
    normalized = (int(position[0]), int(position[1]))
    with _blocked_cursor_lock:
        for allowed in reversed(_blocked_cursor_whitelist_stack):
            if normalized in allowed:
                return True
    return False

HUMAN_CURVE = 0.65
STEP_NOISE_PX = 0.6
OVERSHOOT_PROB = 0.30
OVERSHOOT_MAX_PX = 12
SETTLE_MICRO_PX = 6
MIN_JERK = True
PATH_STEP_PX = 14.0
TIME_PER_PX  = 0.00055
MIN_PER_STEP = 0.0010 

def set_speed(value):
    global SPEED
    try:
        SPEED = max(0.1, float(value))
    except (TypeError, ValueError):
        pass

def _speed_factor():
    return 1.0 / max(0.1, float(SPEED))

def scaled_sleep(base_time):
    total = max(0.0, float(base_time) * _speed_factor())
    if total <= 0:
        abort_if_killed()
        return
    deadline = time.time() + total
    while True:
        abort_if_killed()
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(0.05, remaining))
    abort_if_killed()

def time_for_distance(px, base_time_per_px=None, *, clamp=(0.04, 2.2)):
    if base_time_per_px is None:
        base_time_per_px = float(globals().get("TIME_PER_PX", 0.0011))
    px = max(0.0, float(px))
    t = px * base_time_per_px * _speed_factor()
    lo, hi = clamp
    return max(lo, min(hi, t))

def steps_for_distance(px, *, density_px=None, base_steps=None):
    if density_px is None:
        density_px = float(globals().get("PATH_STEP_PX", 14.0))
    if base_steps is None:
        base_steps = max(4, int(globals().get("STEPS", 7)))
    px = max(0.0, float(px))
    return max(base_steps, int(px / max(2.0, density_px)))

def per_step_delay(total_time, steps, *, min_per_step=None):
    if min_per_step is None:
        min_per_step = float(globals().get("MIN_PER_STEP", 0.0020))
    steps = max(1, int(steps))
    return max(min_per_step * _speed_factor(), float(total_time) / steps)

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

def move_to(x, y):
    ensure_cursor_not_blocked()
    abort_if_killed()
    interception.move_to(int(x), int(y))

def mouse_down():
    abort_if_killed()
    x, y = get_cursor_pos()
    interception.mouse_down(button="left")

def mouse_up():
    abort_if_killed()
    x, y = get_cursor_pos()
    interception.mouse_up(button="left")

def click(x=None, y=None, button="left", delay=0.05):
    abort_if_killed()
    if x is None or y is None:
        x, y = get_cursor_pos()
    interception.click(int(x), int(y), button=button, delay=max(0.0, float(delay) * _speed_factor()))

def get_cursor_pos():
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

def handle_blocked_cursor(position):
    normalized = tuple(int(v) for v in position)
    if get_puzzle_monitor() is not None and _is_blocked_position_whitelisted(normalized):
        print(f"Blocked cursor at {normalized} is whitelisted during puzzle execution; ignoring auto-close.")
        return True
    print(f"Blocked cursor detected at {normalized}. Custom handler invoked.")
    interception.key_down("w")
    time.sleep(0.2)
    interception.key_up("w")
    return False

def is_blocked_cursor_position(position):
    try:
        x, y = position
    except (TypeError, ValueError):
        return False
    return (int(x), int(y)) in BLOCKED_CURSOR_POSITIONS

def ensure_cursor_not_blocked():
    position = get_cursor_pos()
    if is_blocked_cursor_position(position):
        if handle_blocked_cursor(position):
            return position
        raise MouseBlockedAtStart(position)
    return position

def cell_centers_from_path(cells, h_lines, v_lines):
    points = []
    for (r, c) in cells:
        x1, x2 = v_lines[c], v_lines[c + 1]
        y1, y2 = h_lines[r], h_lines[r + 1]
        points.append(((x1 + x2) // 2, (y1 + y2) // 2))
    return points

def _sample_bezier(p0, c1, c2, p3, t):
    mt = 1.0 - t
    x = (mt*mt*mt)*p0[0] + 3*mt*mt*t*c1[0] + 3*mt*t*t*c2[0] + (t*t*t)*p3[0]
    y = (mt*mt*mt)*p0[1] + 3*mt*mt*t*c1[1] + 3*mt*t*t*c2[1] + (t*t*t)*p3[1]
    return (x, y)

def _min_jerk(t):
    return (10*t*t*t - 15*t*t*t*t + 6*t*t*t*t*t)

def _ease(t):
    try:
        return _min_jerk(t) if MIN_JERK else (t*t*(3 - 2*t))
    except NameError:
        return (t*t*(3 - 2*t))

def _path_length(points):
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        total += (dx*dx + dy*dy) ** 0.5
    return total

def _resample_bezier_segments(beziers, step_px=6.0):
    if not beziers:
        return []
    dense = []
    for (p0, c1, c2, p3) in beziers:
        approx = (
            ((p0[0]-p3[0])**2 + (p0[1]-p3[1])**2) ** 0.5 +
            ((p0[0]-c1[0])**2 + (p0[1]-c1[1])**2) ** 0.5 +
            ((c2[0]-p3[0])**2 + (c2[1]-p3[1])**2) ** 0.5
        )
        samples = max(6, int(approx / max(2.0, step_px)))
        for i in range(samples):
            t = i / float(samples)
            dense.append(_sample_bezier(p0, c1, c2, p3, t))
    dense.append(beziers[-1][3])
    return dense

def _cm_bezier_segments(points, tau=0.5):
    if not points or len(points) == 1:
        return []
    if len(points) == 2:
        p0, p3 = points[0], points[1]
        c1 = (p0[0] + (p3[0]-p0[0]) * 0.33, p0[1] + (p3[1]-p0[1]) * 0.33)
        c2 = (p0[0] + (p3[0]-p0[0]) * 0.66, p0[1] + (p3[1]-p0[1]) * 0.66)
        return [(p0, c1, c2, p3)]

    segs = []
    pts = [points[0]] + points + [points[-1]]
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i-1], pts[i], pts[i+1], pts[i+2]
        t1 = ((p2[0] - p0[0]) * tau, (p2[1] - p0[1]) * tau)
        t2 = ((p3[0] - p1[0]) * tau, (p3[1] - p1[1]) * tau)
        c1 = (p1[0] + t1[0] / 3.0, p1[1] + t1[1] / 3.0)
        c2 = (p2[0] - t2[0] / 3.0, p2[1] - t2[1] / 3.0)
        segs.append((p1, c1, c2, p2))
    return segs

def _centers_from_cells_absolute(cells, h_lines, v_lines, ox, oy):
    pts = []
    for (r, c) in cells:
        x1, x2 = v_lines[c], v_lines[c + 1]
        y1, y2 = h_lines[r], h_lines[r + 1]
        pts.append((ox + (x1 + x2)//2, oy + (y1 + y2)//2))
    return pts

def human_drag_cells(
    cells,
    h_lines,
    v_lines,
    *,
    jitter=True,
    position_callback=None,
    progress_callback=None,
    progress_visualizer=None,
    color=None,
    trail_monitor=None
):
    ensure_cursor_not_blocked()
    abort_if_killed()

    if progress_visualizer is not None:
        ox = getattr(progress_visualizer, "origin_x", 0)
        oy = getattr(progress_visualizer, "origin_y", 0)
        hl = getattr(progress_visualizer, "h_lines", h_lines)
        vl = getattr(progress_visualizer, "v_lines", v_lines)
    else:
        ox, oy, hl, vl = 0, 0, h_lines, v_lines
    points = _centers_from_cells_absolute(cells, hl, vl, ox, oy)
    if not points:
        return
    beziers = _cm_bezier_segments(points, tau=0.5)
    step_px = float(globals().get("PATH_STEP_PX", 14.0))
    dense = _resample_bezier_segments(beziers, step_px=max(6.0, step_px))
    total_len = _path_length(dense)
    total_time = time_for_distance(total_len, base_time_per_px=float(globals().get("TIME_PER_PX", 0.0011)),
                                   clamp=(0.06, 1.8))
    steps = steps_for_distance(total_len, density_px=step_px, base_steps=max(4, int(globals().get("STEPS", 7))))
    per_step = per_step_delay(total_time, steps, min_per_step=float(globals().get("MIN_PER_STEP", 0.0020)))
    base_noise = (0.35 + JITTER_SCALE * 0.45) if jitter else 0.0
    dense_int_positions = {(int(round(px)), int(round(py))) for px, py in dense}
    whitelist_positions = BLOCKED_CURSOR_POSITIONS.intersection(dense_int_positions)
    whitelist_ctx = _whitelist_blocked_positions(whitelist_positions) if whitelist_positions else nullcontext()

    with whitelist_ctx:
        for i in range(1, len(dense)):
            abort_if_killed()
            ensure_cursor_not_blocked()

            t = _ease(i / float(len(dense) - 1))
            x, y = dense[i]

            if jitter and base_noise > 0.0 and i < len(dense) - 1:
                fall = 0.15 + 0.85 * (1.0 - t)
                x += random.uniform(-base_noise, base_noise) * fall
                y += random.uniform(-base_noise, base_noise) * fall

            interception.move_to(int(round(x)), int(round(y)))

            if position_callback is not None:
                try:
                    position_callback(get_cursor_pos())
                except Exception:
                    pass

            if trail_monitor is not None and progress_visualizer is not None:
                try:
                    px, py = x, y
                    nearest_idx = 0
                    if len(points) > 1:
                        best = 1e18
                        for idx, (cx, cy) in enumerate(points):
                            d = (cx - px)*(cx - px) + (cy - py)*(cy - py)
                            if d < best:
                                best = d; nearest_idx = idx
                    if 0 <= nearest_idx < len(cells):
                        trail_monitor.set_context(highlight=cells[nearest_idx])
                except Exception:
                    pass

            scaled_sleep(per_step)

SEARCH_REGION = {
    "top": 200,
    "left": 400,
    "width": 1100,
    "height": 800
}
COLOR_DISTANCE_THRESHOLD = 5
SATURATION_THRESHOLD = 8
VALUE_THRESHOLD = 50
BRIGHT_VALUE_THRESHOLD = 50
MIN_POINT_PIXELS = 50
CORE_CROP_RATIO = 0.15

def _unit(vx, vy):
    mag = (vx*vx + vy*vy) ** 0.5
    if mag <= 1e-9:
        return 0.0, 0.0
    return vx / mag, vy / mag

def _perp(vx, vy):
    return -vy, vx

def _min_jerk(t):
    return (10*t*t*t - 15*t*t*t*t + 6*t*t*t*t*t)

def _easing(t):
    return _min_jerk(t) if MIN_JERK else (t*t*(3 - 2*t))

def _bezier_point(p0, p1, p2, p3, t):
    mt = 1.0 - t
    return (
        mt*mt*mt*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t*t*t*p3[0],
        mt*mt*mt*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t*t*t*p3[1],
    )

def _curved_controls(start, end, curve_strength):
    sx, sy = start
    ex, ey = end
    vx, vy = (ex - sx), (ey - sy)
    ux, uy = _unit(vx, vy)
    px, py = _perp(ux, uy)
    dist = max(1.0, (vx*vx + vy*vy) ** 0.5)
    base = dist * 0.15 * max(0.0, min(1.0, curve_strength))
    k1 = (random.uniform(0.6, 1.0) * base) * (1 if random.random() < 0.5 else -1)
    k2 = (random.uniform(0.6, 1.0) * base) * (1 if random.random() < 0.5 else -1)
    c1 = (sx + vx * 0.33 + px * k1, sy + vy * 0.33 + py * k1)
    c2 = (sx + vx * 0.66 + px * k2, sy + vy * 0.66 + py * k2)
    return c1, c2

def smoothstep(t):
    return t * t * (3 - 2 * t)

def smooth_move_to(target, duration=None, jitter=True, position_callback=None):
    ensure_cursor_not_blocked()
    abort_if_killed()

    import random as _rnd
    human_curve = globals().get("HUMAN_CURVE", 0.65)
    step_noise_px = globals().get("STEP_NOISE_PX", 0.6)
    overshoot_prob = globals().get("OVERSHOOT_PROB", 0.30)
    overshoot_max = globals().get("OVERSHOOT_MAX_PX", 14)
    settle_px = globals().get("SETTLE_MICRO_PX", 6)
    start_x, start_y = get_cursor_pos()
    tx, ty = int(target[0]), int(target[1])
    dx, dy = (tx - start_x), (ty - start_y)
    distance = max(1.0, (dx*dx + dy*dy) ** 0.5)
    if duration is None:
        duration = time_for_distance(distance, clamp=(0.06, 1.6))
    steps = steps_for_distance(distance, density_px=globals().get("PATH_STEP_PX", 14.0),
                               base_steps=max(4, int(globals().get("STEPS", 7))))
    per_step = per_step_delay(duration, steps)

    p0 = (float(start_x), float(start_y))
    p3 = (float(tx), float(ty))
    c1, c2 = _curved_controls(p0, p3, human_curve)
    do_overshoot = (_rnd.random() < overshoot_prob and distance > 25.0)
    if do_overshoot:
        ux, uy = _unit(dx, dy)
        overshoot_len = _rnd.uniform(3.0, overshoot_max)
        p3_over = (p3[0] + ux * overshoot_len, p3[1] + uy * overshoot_len)
        c1_over, c2_over = _curved_controls(p0, p3_over, human_curve * 0.9)
        path_segments = [
            (p0, c1_over, c2_over, p3_over, int(steps * 0.8)),
            (p3_over, * _curved_controls(p3_over, p3, human_curve*0.4), p3, int(steps * 0.2))
        ]
    else:
        path_segments = [(p0, c1, c2, p3, steps)]
    base_noise = step_noise_px + (JITTER_SCALE * 0.5 if jitter else 0.0)
    planned_positions = set()
    for (q0, q1, q2, q3, seg_steps) in path_segments:
        seg_count = max(2, int(seg_steps))
        for i in range(1, seg_count + 1):
            t = _easing(i / float(seg_count))
            x, y = _bezier_point(q0, q1, q2, q3, t)
            planned_positions.add((int(round(x)), int(round(y))))
    whitelist_positions = planned_positions.intersection(BLOCKED_CURSOR_POSITIONS)
    if position_callback is not None:
        try:
            position_callback((int(start_x), int(start_y)))
        except Exception:
            pass

    whitelist_ctx = _whitelist_blocked_positions(whitelist_positions) if whitelist_positions else nullcontext()
    with whitelist_ctx:
        for (q0, q1, q2, q3, seg_steps) in path_segments:
            seg_steps = max(2, int(seg_steps))
            for i in range(1, seg_steps + 1):
                abort_if_killed()
                ensure_cursor_not_blocked()
                t = _easing(i / float(seg_steps))
                x, y = _bezier_point(q0, q1, q2, q3, t)
                if jitter and base_noise > 0.0 and i < seg_steps:
                    falloff = 0.2 + 0.8 * (1.0 - t)
                    x += _rnd.uniform(-base_noise, base_noise) * falloff
                    y += _rnd.uniform(-base_noise, base_noise) * falloff
                interception.move_to(int(round(x)), int(round(y)))
                if position_callback is not None:
                    try:
                        position_callback(get_cursor_pos())
                    except Exception:
                        pass
                scaled_sleep(per_step)

        if do_overshoot:
            for _ in range(2):
                abort_if_killed(); ensure_cursor_not_blocked()
                nx = tx + _rnd.randint(-settle_px, settle_px)
                ny = ty + _rnd.randint(-settle_px, settle_px)
                interception.move_to(int(nx), int(ny))
                if position_callback is not None:
                    try:
                        position_callback(get_cursor_pos())
                    except Exception:
                        pass
                scaled_sleep(per_step * _speed_factor())

        abort_if_killed(); ensure_cursor_not_blocked()
        interception.move_to(int(tx), int(ty))
        if position_callback is not None:
            try:
                position_callback(get_cursor_pos())
            except Exception:
                pass


TOLERANCE = 0
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

class MouseMovementMonitor:
    def __init__(self, idle_timeout=1.0):
        try:
            idle = float(idle_timeout)
        except (TypeError, ValueError):
            idle = 1.0
        self.idle_timeout = max(0.01, idle)
        self._active = False
        self._last_position = None
        self._last_change_ts = 0.0
        self._handlers = []
        self.add_handler(self._default_handler)

    @property
    def active(self):
        return self._active

    def start(self):
        self._active = True
        self._last_position = None
        self._last_change_ts = time.monotonic()

    def stop(self):
        self._active = False
        self._last_position = None

    def ensure_active(self):
        if not self._active:
            return
        now = time.monotonic()
        try:
            current = pyautogui.position()
        except Exception:
            return
        position = (int(current.x), int(current.y))
        if self._last_position is None or position != self._last_position:
            self._last_position = position
            self._last_change_ts = now
            return
        if now - self._last_change_ts >= self.idle_timeout:
            self._trigger(position)

    def add_handler(self, handler, *, prepend=False):
        if not callable(handler):
            return
        if prepend:
            self._handlers.insert(0, handler)
        else:
            self._handlers.append(handler)

    def _trigger(self, frozen_position):
        self._active = False
        captured_exc = None
        for handler in list(self._handlers):
            try:
                handler(self, frozen_position)
            except MouseMovementStalled as exc:
                if captured_exc is None:
                    captured_exc = exc
            except Exception as exc:
                if captured_exc is None:
                    captured_exc = exc
        if captured_exc is not None:
            raise captured_exc
        raise MouseMovementStalled(
            f"Mouse movement stalled at {frozen_position} for {self.idle_timeout:.2f}s."
        )

    def _default_handler(self, monitor, frozen_position):
        raise MouseMovementStalled(
            f"Mouse movement stalled at {frozen_position} for {monitor.idle_timeout:.2f}s."
        )

class PuzzlePresenceMonitor:
    def __init__(self, bbox, tolerance=25, check_interval=0.3):
        self.reference_bbox = tuple(int(v) for v in bbox) if bbox else None
        self.tolerance = max(0, int(round(tolerance)))
        self.check_interval = max(0.05, float(check_interval))
        self._last_check = 0.0

    def ensure_present(self):
        if self.reference_bbox is None:
            return
        now = time.time()
        if now - self._last_check < self.check_interval:
            return
        try:
            screenshot = capture_region()
        except Exception as exc:
            raise PuzzleNotAvailable(f"Failed to capture puzzle region: {exc}") from exc
        bbox = find_puzzle_bbox(screenshot)
        if not bbox:
            raise PuzzleNotAvailable("Puzzle area not found.")
        rx, ry, rw, rh = self.reference_bbox
        x, y, w, h = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        if (
            abs(x - rx) > self.tolerance
            or abs(y - ry) > self.tolerance
            or abs(w - rw) > self.tolerance
            or abs(h - rh) > self.tolerance
        ):
            raise PuzzleNotAvailable("Puzzle bounds changed unexpectedly.")
        self.reference_bbox = (x, y, w, h)
        self._last_check = now

LINE_MERGE_GAP = 3
MIN_EXPECTED_SPACING = 12 
MAX_JUMP_MULTIPLIER = 2 
def _merge_nearby_positions(positions, gap=LINE_MERGE_GAP):
    if not positions:
        return []
    positions = sorted(positions)
    merged = []
    cluster = [positions[0]]
    for p in positions[1:]:
        if p - cluster[-1] <= gap:
            cluster.append(p)
        else:
            merged.append(int(round(sum(cluster) / len(cluster))))
            cluster = [p]
    merged.append(int(round(sum(cluster) / len(cluster))))
    return merged

def _choose_expected_spacing(spacings, tolerance, min_spacing=MIN_EXPECTED_SPACING):
    spacings = [s for s in spacings if s > max(tolerance, LINE_MERGE_GAP)]
    if not spacings:
        return None
    bins = {}
    for s in spacings:
        b = round(s / tolerance) * tolerance
        bins[b] = bins.get(b, 0) + 1
    candidates = sorted(bins.items(), key=lambda kv: (-kv[1], -kv[0]))
    for spacing, _count in candidates:
        if spacing >= min_spacing:
            return int(spacing)
    spacings.sort()
    mid = len(spacings) // 2
    median = (spacings[mid] if len(spacings) % 2 else (spacings[mid - 1] + spacings[mid]) / 2)
    return int(round(max(median, min_spacing)))

def filter_lines(lines, tolerance=5, label="lines"):
    if not lines:
        print(f"[Grid] No {label} candidates to filter.")
        return []
    original_len = len(lines)
    lines = _merge_nearby_positions(sorted(lines), gap=max(LINE_MERGE_GAP, tolerance))
    if len(lines) != original_len:
        print(f"[Grid] {label.title()} merged near-dupes: {original_len} -> {len(lines)}")
    if len(lines) < 2:
        print(f"[Grid] Not enough {label} candidates after merge: {lines}")
        return lines
    raw_spacings = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
    spacing_counts = {}
    for s in raw_spacings:
        if s <= max(LINE_MERGE_GAP, tolerance):
            continue
        b = round(s / tolerance) * tolerance
        spacing_counts[b] = spacing_counts.get(b, 0) + 1
    print(f"[Grid] {label.title()} spacing histogram (post-merge): {spacing_counts or {'<empty>': 0}}")
    expected_spacing = _choose_expected_spacing(raw_spacings, tolerance, MIN_EXPECTED_SPACING)
    if not expected_spacing:
        print(f"[Grid] {label.title()} could not pick expected spacing, keeping merged lines.")
        return lines
    print(f"[Grid] {label.title()} spacing stats -> "
          f"min: {min(raw_spacings)}, max: {max(raw_spacings)}, "
          f"expected: {expected_spacing} (+/- {tolerance})")
    best_seq = None
    for start_idx in range(min(5, len(lines))):
        seq = [lines[start_idx]]
        last = lines[start_idx]
        for p in lines[start_idx + 1:]:
            delta = p - last
            k = int(round(delta / float(expected_spacing))) if expected_spacing > 0 else 0
            if k < 1 or k > MAX_JUMP_MULTIPLIER:
                continue
            if abs(delta - k * expected_spacing) <= tolerance * k:
                seq.append(p)
                last = p
        if best_seq is None or len(seq) > len(best_seq):
            best_seq = seq
    if best_seq is None:
        best_seq = lines
    if len(best_seq) != len(lines):
        print(f"[Grid] {label.title()} lines filtered from {len(lines)} to {len(best_seq)} using expected spacing {expected_spacing}.")
    return best_seq

def detect_grid(img):
    debug = img.copy()
    print(f"[Grid] detect_grid start | image shape={img.shape}")
    mask_div = cv2.inRange(img, (20, 20, 20), (20, 20, 20))
    mask_nonzero = int(np.count_nonzero(mask_div))
    mask_ratio = mask_nonzero / mask_div.size if mask_div.size else 0.0
    print(f"[Grid] Divider mask non-zero: {mask_nonzero}/{mask_div.size} ({mask_ratio:.2%})")
    def _summarize(values, limit=10):
        if not values:
            return "[]"
        preview = ", ".join(str(v) for v in values[:limit])
        if len(values) > limit:
            preview += ", ..."
        return f"[{preview}]"
    horizontal_sum = np.sum(mask_div, axis=1)
    row_threshold = 0.9 * mask_div.shape[1] * 255
    print(f"[Grid] Horizontal sum stats -> min: {int(horizontal_sum.min())}, max: {int(horizontal_sum.max())}, threshold: {row_threshold:.0f}")
    horizontal_candidates = [i for i, v in enumerate(horizontal_sum) if v > row_threshold]
    print(f"[Grid] Horizontal candidates ({len(horizontal_candidates)}): {_summarize(horizontal_candidates)}")
    horizontal_lines = filter_lines(horizontal_candidates, tolerance=5, label="horizontal")
    print(f"[Grid] Horizontal lines filtered ({len(horizontal_lines)}): {_summarize(horizontal_lines)}")
    vertical_sum = np.sum(mask_div, axis=0)
    col_threshold = 0.9 * mask_div.shape[0] * 255
    print(f"[Grid] Vertical sum stats -> min: {int(vertical_sum.min())}, max: {int(vertical_sum.max())}, threshold: {col_threshold:.0f}")
    vertical_candidates = [i for i, v in enumerate(vertical_sum) if v > col_threshold]
    print(f"[Grid] Vertical candidates ({len(vertical_candidates)}): {_summarize(vertical_candidates)}")
    vertical_lines = filter_lines(vertical_candidates, tolerance=5, label="vertical")
    print(f"[Grid] Vertical lines filtered ({len(vertical_lines)}): {_summarize(vertical_lines)}")
    if len(horizontal_lines) >= 2:
        horizontal_spacing = [horizontal_lines[i + 1] - horizontal_lines[i] for i in range(len(horizontal_lines) - 1)]
        print(f"[Grid] Horizontal spacing samples: {_summarize(horizontal_spacing)}")
    else:
        print("[Grid] Not enough horizontal lines to compute spacing.")

    if len(vertical_lines) >= 2:
        vertical_spacing = [vertical_lines[i + 1] - vertical_lines[i] for i in range(len(vertical_lines) - 1)]
        print(f"[Grid] Vertical spacing samples: {_summarize(vertical_spacing)}")
    else:
        print("[Grid] Not enough vertical lines to compute spacing.")
    rows = len(horizontal_lines) - 1
    cols = len(vertical_lines) - 1
    print(f"[Grid] Detected grid dimensions: {rows} rows x {cols} cols")
    if rows <= 0:
        print("[Grid] Warning: insufficient horizontal lines to form rows.")
    if cols <= 0:
        print("[Grid] Warning: insufficient vertical lines to form columns.")
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
            start_cell, end_cell = a, b
        else:
            start_cell, end_cell = b, a
        color_pairs[color] = (start_cell, end_cell)
        active_colors.append(color)
    if not active_colors:
        return None
    active_colors.sort(key=lambda c: (-manhattan(color_pairs[c][0], color_pairs[c][1]), c))
    progress = {color: [color_pairs[color][0]] for color in active_colors}
    progress_sets = {color: {color_pairs[color][0]} for color in active_colors}
    completed = set()
    solution_paths = {}
    color_ids = {color: idx for idx, color in enumerate(active_colors)}
    visited_failures = set()
    def move_degree(color, cell):
        target = color_pairs[color][1]
        degree = 0
        for nb in neighbors(cell):
            if grid[nb[0]][nb[1]] is None or nb == target:
                degree += 1
        return degree

    def legal_moves(color):
        current = progress[color][-1]
        target = color_pairs[color][1]
        moves = []
        visited_cells = progress_sets[color]
        for nb in neighbors(current):
            if nb == target:
                moves.append({'cell': nb, 'is_target': True, 'degree': move_degree(color, nb), 'distance': 0})
            elif grid[nb[0]][nb[1]] is None and nb not in visited_cells:
                moves.append({'cell': nb, 'is_target': False, 'degree': move_degree(color, nb), 'distance': manhattan(nb, target)})
        moves.sort(key=lambda info: (0 if info['is_target'] else 1, info['degree'], info['distance']))
        return moves

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

    def ensure_connectivity(specified_colors=None):
        if specified_colors is None:
            colors_to_check = [c for c in active_colors if c not in completed]
        else:
            colors_to_check = [c for c in specified_colors if c not in completed]
        for color in colors_to_check:
            if not path_possible(color):
                return False
        return True

    def has_dead_cells():
        incomplete = [c for c in active_colors if c not in completed]
        if not incomplete:
            return False
        head_lookup = {progress[c][-1]: c for c in incomplete}
        remaining_targets = {color_pairs[c][1]: c for c in incomplete}
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] is not None:
                    continue
                cell = (r, c)
                accessible = 0
                for nb in neighbors(cell):
                    if grid[nb[0]][nb[1]] is None:
                        accessible += 1
                    elif nb in remaining_targets:
                        accessible += 1
                    elif nb in head_lookup:
                        accessible += 1
                if accessible <= 1:
                    return True
        return False

    def state_signature():
        flat = []
        for r in range(rows):
            for c in range(cols):
                val = grid[r][c]
                if val is None:
                    flat.append('0')
                else:
                    if val not in color_ids:
                        color_ids[val] = len(color_ids)
                        print(f"[Solver] Added new flow mapping: {val} -> {color_ids[val]}")
                    flat.append(str(color_ids[val] + 1))
        heads = []
        for color in active_colors:
            head = progress[color][-1]
            heads.append(f"{head[0]}:{head[1]}")
        completed_bits = ''.join('1' if color in completed else '0' for color in active_colors)
        return ''.join(flat) + '|' + ';'.join(heads) + '|' + completed_bits

    def make_move(color, cell):
        progress[color].append(cell)
        progress_sets[color].add(cell)
        target = color_pairs[color][1]
        if cell == target:
            completed.add(color)
            return ('complete', cell)
        grid[cell[0]][cell[1]] = color
        return ('fill', cell)

    def undo_move(color, marker):
        kind, cell = marker
        if kind == 'fill':
            grid[cell[0]][cell[1]] = None
        else:
            completed.discard(color)
        progress_sets[color].discard(cell)
        progress[color].pop()

    def undo_history(history):
        for color, marker in reversed(history):
            undo_move(color, marker)
        history.clear()

    def apply_forced_moves(history):
        while True:
            changed = False
            for color in active_colors:
                if color in completed:
                    continue
                moves = legal_moves(color)
                if not moves:
                    return False
                if len(moves) == 1:
                    cell_info = moves[0]
                    marker = make_move(color, cell_info['cell'])
                    history.append((color, marker))
                    if has_dead_cells() or not ensure_connectivity():
                        return False
                    changed = True
                    break
            if not changed:
                return True

    def choose_color():
        best_color = None
        best_moves = None
        best_key = None
        for color in active_colors:
            if color in completed:
                continue
            moves = legal_moves(color)
            if not moves:
                return color, []
            key = (len(moves), moves[0]['degree'], moves[0]['distance'], manhattan(progress[color][-1], color_pairs[color][1]))
            if best_key is None or key < best_key:
                best_color = color
                best_moves = moves
                best_key = key
        return best_color, best_moves

    def search():
        if len(completed) == len(active_colors):
            for color in active_colors:
                solution_paths[color] = list(progress[color])
            return True
        state_key_before = state_signature()
        if state_key_before in visited_failures:
            return False
        forced_history = []
        if not apply_forced_moves(forced_history):
            undo_history(forced_history)
            visited_failures.add(state_key_before)
            return False
        if len(completed) == len(active_colors):
            for color in active_colors:
                solution_paths[color] = list(progress[color])
            return True
        state_key_after = state_signature()
        if state_key_after in visited_failures:
            undo_history(forced_history)
            return False
        if has_dead_cells() or not ensure_connectivity():
            visited_failures.add(state_key_after)
            undo_history(forced_history)
            return False
        color, moves = choose_color()
        if color is None or not moves:
            visited_failures.add(state_key_after)
            undo_history(forced_history)
            return False
        for move_info in moves:
            cell = move_info['cell']
            marker = make_move(color, cell)
            branch_history = []
            ok = True
            if has_dead_cells() or not ensure_connectivity():
                ok = False
            else:
                if not apply_forced_moves(branch_history):
                    ok = False
                elif has_dead_cells() or not ensure_connectivity():
                    ok = False
            if ok:
                branch_key = state_signature()
                if branch_key in visited_failures:
                    ok = False
            if ok and search():
                return True
            undo_history(branch_history)
            undo_move(color, marker)
        visited_failures.add(state_key_after)
        undo_history(forced_history)
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

def optimize_solution_paths(solutions, rows, cols, detected):
    if not solutions:
        return solutions, False
    endpoints = {}
    for r, c, color in detected:
        endpoints.setdefault(color, []).append((r, c))
    usable = {}
    for color, pts in endpoints.items():
        if len(pts) >= 2:
            usable[color] = (pts[0], pts[1])
    if not usable:
        return solutions, False
    optimized = {color: list(path) for color, path in solutions.items()}
    rows = int(rows)
    cols = int(cols)
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for color, path in optimized.items():
        for r, c in path:
            grid[r][c] = color
    def shortest_path(start, end):
        queue = deque([start])
        parents = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == end:
                break
            r, c = cell
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                nxt = (nr, nc)
                if nxt in parents:
                    continue
                occupant = grid[nr][nc]
                if occupant is None or nxt == end:
                    parents[nxt] = cell
                    queue.append(nxt)
        if end not in parents:
            return None
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = parents[cur]
        path.reverse()
        return path
    improved = False
    while True:
        changed = False
        for color, endpoints_pair in usable.items():
            path = optimized.get(color)
            if not path or len(path) < 2:
                continue
            start, end = endpoints_pair
            old_path = list(path)
            old_length = len(old_path)
            for r, c in old_path[1:-1]:
                grid[r][c] = None
            new_path = shortest_path(start, end)
            if new_path and len(new_path) < old_length:
                optimized[color] = new_path
                for r, c in new_path:
                    grid[r][c] = color
                improved = True
                changed = True
            else:
                for r, c in old_path[1:-1]:
                    grid[r][c] = color
        if not changed:
            break
    if not improved:
        return solutions, False
    return optimized, True

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
class PathProgressVisualizer:
    @staticmethod
    def _copy_image(image):
        if image is None:
            return None
        try:
            return image.copy()
        except AttributeError:
            return image

    def __init__(
        self,
        base_image,
        h_lines,
        v_lines,
        color_map,
        alpha=0.45,
        origin=None,
        trail_thickness=5,
    ):
        self._lock = RLock()
        self.base_image = self._copy_image(base_image)
        self.h_lines = list(h_lines)
        self.v_lines = list(v_lines)
        self.color_map = dict(color_map or {})
        try:
            alpha_value = float(alpha)
        except (TypeError, ValueError):
            alpha_value = 0.45
        self.alpha = max(0.1, min(0.85, alpha_value))
        self.completed = {}
        try:
            ox, oy = origin if origin is not None else (0, 0)
        except Exception:
            ox, oy = 0, 0
        self.origin_x = int(ox)
        self.origin_y = int(oy)
        try:
            thickness = int(round(trail_thickness))
        except (TypeError, ValueError):
            thickness = 3
        self.trail_thickness = max(1, thickness)
        self._trail_history = {}
        self._active_trail_color = None
        self._active_trail_points = []

    def _set_origin_locked(self, origin):
        try:
            ox, oy = origin
        except Exception:
            return
        self.origin_x = int(ox)
        self.origin_y = int(oy)

    def set_origin(self, origin):
        if origin is None:
            return
        with self._lock:
            self._set_origin_locked(origin)

    def _reset_trails_locked(self):
        self._trail_history.clear()
        self._active_trail_color = None
        self._active_trail_points = []

    def reset_trails(self):
        with self._lock:
            self._reset_trails_locked()

    def project_global(self, position):
        if position is None:
            return None
        with self._lock:
            if self.base_image is None:
                return None
            try:
                px, py = position
            except (TypeError, ValueError):
                return None
            local_x = int(round(px - self.origin_x))
            local_y = int(round(py - self.origin_y))
            height, width = self.base_image.shape[:2]
            if local_x < 0 or local_y < 0 or local_x >= width or local_y >= height:
                return None
            return local_x, local_y

    def begin_trail(self, color_label):
        with self._lock:
            self._commit_active_trail_locked()
            self._active_trail_color = color_label
            self._active_trail_points = []

    def add_mouse_point(self, position):
        with self._lock:
            if self.base_image is None or self._active_trail_color is None:
                return None
            local = self.project_global(position)
            if local is None:
                return None
            local_x, local_y = local
            if self._active_trail_points:
                last_x, last_y = self._active_trail_points[-1]
                if local_x == last_x and local_y == last_y:
                    return local
            self._active_trail_points.append((local_x, local_y))
            return local

    def end_trail(self):
        with self._lock:
            self._commit_active_trail_locked()
            self._active_trail_color = None
            self._active_trail_points = []

    def _commit_active_trail_locked(self):
        if self._active_trail_color is None:
            self._active_trail_points = []
            return
        if len(self._active_trail_points) < 2:
            self._active_trail_points = []
            return
        color_trails = self._trail_history.setdefault(self._active_trail_color, [])
        color_trails.append(list(self._active_trail_points))
        self._active_trail_points = []

    def update_base(self, base_image, origin=None, *, reset_trails=False):
        if base_image is None:
            return
        with self._lock:
            self.base_image = self._copy_image(base_image)
            if origin is not None:
                self._set_origin_locked(origin)
            if reset_trails:
                self._reset_trails_locked()

    def _cell_bounds_locked(self, row, col):
        if row < 0 or col < 0:
            return None
        if row >= len(self.h_lines) - 1 or col >= len(self.v_lines) - 1:
            return None
        x1, x2 = self.v_lines[col], self.v_lines[col + 1]
        y1, y2 = self.h_lines[row], self.h_lines[row + 1]
        return x1, y1, x2, y2

    def mark_cell(self, cell, color_label):
        if cell is None or color_label is None:
            return
        with self._lock:
            self.completed[cell] = color_label

    def _draw_trails_locked(self, canvas):
        if self.base_image is None:
            return
        for color_label, segments in self._trail_history.items():
            sanitized = [segment for segment in segments if segment]
            if not sanitized:
                continue
            color_bgr = self.color_map.get(color_label, (255, 255, 255))
            self._render_segments_locked(canvas, sanitized, color_bgr, active=False)
        if self._active_trail_color and self._active_trail_points:
            color_bgr = self.color_map.get(self._active_trail_color, (255, 255, 255))
            active_segment = [self._active_trail_points[:]]
            self._render_segments_locked(canvas, active_segment, color_bgr, active=True)

    def _render_segments_locked(self, canvas, segments, color, active=False):
        if not segments:
            return
        base_thickness = max(2, self.trail_thickness + (1 if active else 0))
        def _scale(col, factor=1.0, offset=0):
            return tuple(
                max(0, min(255, int(round(channel * factor + offset))))
                for channel in col
            )
        glow_color = _scale(color, 0.85, 10)
        shadow_color = _scale(color, 0.55, -15)
        stroke_color = _scale(color, 1.15, 25)
        highlight_color = _scale(color, 1.25 if active else 1.1, 45 if active else 35)
        point_radius = max(3, base_thickness // 2 + (2 if active else 1))
        overlay = np.zeros_like(canvas)
        overlay_drawn = False
        for segment in segments:
            if len(segment) < 2:
                continue
            pts = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                overlay,
                [pts],
                False,
                glow_color,
                thickness=base_thickness + 6,
                lineType=cv2.LINE_AA,
            )
            overlay_drawn = True
        if overlay_drawn:
            glow = cv2.GaussianBlur(overlay, (0, 0), sigmaX=3, sigmaY=3)
            cv2.addWeighted(glow, 0.35 if active else 0.25, canvas, 1.0, 0, canvas)
        for segment in segments:
            if not segment:
                continue
            if len(segment) >= 2:
                pts = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    canvas,
                    [pts],
                    False,
                    shadow_color,
                    thickness=base_thickness + 2,
                    lineType=cv2.LINE_AA,
                )
                cv2.polylines(
                    canvas,
                    [pts],
                    False,
                    stroke_color,
                    thickness=base_thickness,
                    lineType=cv2.LINE_AA,
                )
            for px, py in segment:
                center = (int(px), int(py))
                cv2.circle(canvas, center, point_radius + 2, shadow_color, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, center, point_radius, highlight_color, -1, lineType=cv2.LINE_AA)
        tip_source = segments[-1][-1] if segments and segments[-1] else None
        if active and tip_source is not None:
            tip_center = (int(tip_source[0]), int(tip_source[1]))
            cv2.circle(canvas, tip_center, point_radius + 3, shadow_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, tip_center, point_radius + 1, highlight_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, tip_center, max(point_radius - 1, 2), (255, 255, 255), -1, lineType=cv2.LINE_AA)

    def snapshot(self, current_cell=None, current_color=None, label=None):
        with self._lock:
            if self.base_image is None:
                return None
            canvas = self.base_image.copy()
            overlay = np.zeros_like(canvas)
            for (row, col), color_label in self.completed.items():
                bounds = self._cell_bounds_locked(row, col)
                if bounds is None:
                    continue
                x1, y1, x2, y2 = bounds
                cell_color = self.color_map.get(color_label, (255, 255, 255))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), cell_color, -1)
            cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)
            self._draw_trails_locked(canvas)
            if current_cell is not None:
                bounds = self._cell_bounds_locked(*current_cell)
                if bounds is not None:
                    x1, y1, x2, y2 = bounds
                    highlight_color = self.color_map.get(current_color, (255, 255, 255))
                    accent = tuple(int(min(255, channel + 70)) for channel in highlight_color)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), accent, 2)
            if label:
                padding = 8
                text_box_width = padding * 2 + int(len(label) * 9.5)
                cv2.rectangle(canvas, (8, 8), (8 + text_box_width, 40), (0, 0, 0), -1)
                cv2.putText(canvas, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            return canvas

class CursorTrailMonitor:
    def __init__(self, visualizer, progress_callback=None, interval=0.016):
        self.visualizer = visualizer
        self.progress_callback = progress_callback
        try:
            self.interval = max(0.006, float(interval))
        except (TypeError, ValueError):
            self.interval = 0.016
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._context_lock = RLock()
        self._label: str = ""
        self._highlight_cell = None
        self._color = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="CursorTrailMonitor", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._thread = None
        self.clear_context()

    def clear_context(self):
        with self._context_lock:
            self._label = ""
            self._highlight_cell = None
            self._color = None

    def set_context(self, *, color=None, highlight=None, label=None):
        with self._context_lock:
            if color is not None:
                self._color = color
            if highlight is not None:
                self._highlight_cell = highlight
            if label is not None:
                self._label = label

    def _run(self):
        last_emit = 0.0
        while not self._stop_event.is_set():
            position = get_cursor_pos()
            local = self.visualizer.add_mouse_point(position)
            if local is not None and self.progress_callback is not None:
                now = time.perf_counter()
                if now - last_emit >= self.interval:
                    with self._context_lock:
                        label = self._label
                        highlight = self._highlight_cell
                        color = self._color
                    cursor_text = f"Cursor {int(position[0])},{int(position[1])} | Puzzle {local[0]},{local[1]}"
                    display_label = f"{label} | {cursor_text}" if label else cursor_text
                    frame = self.visualizer.snapshot(
                        current_cell=highlight,
                        current_color=color,
                        label=display_label,
                    )
                    if frame is not None:
                        try:
                            self.progress_callback("cursor_trail|monitor", frame)
                        except Exception:
                            pass
                        last_emit = now
            self._stop_event.wait(self.interval)

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

def drag_path(
    color,
    cells,
    centers,
    hold=True,
    progress_callback=None,
    progress_visualizer=None,
    trail_monitor=None,
):
    if not centers or len(centers) < 2:
        return
    abort_if_killed()
    current_highlight_cell = cells[0] if cells else None
    last_progress_label = None
    use_monitor = (
        trail_monitor is not None
        and progress_callback is not None
        and progress_visualizer is not None
    )
    snapshot_interval = 0.02
    last_snapshot = 0.0

    if progress_visualizer is not None:
        progress_visualizer.begin_trail(color)

    def emit_snapshot(label=None, *, force=False):
        nonlocal last_snapshot
        if use_monitor or progress_callback is None or progress_visualizer is None:
            return
        now = time.monotonic()
        if not force and (now - last_snapshot) < snapshot_interval:
            return
        display_label = label or last_progress_label or f"{color}"
        frame = progress_visualizer.snapshot(
            current_cell=current_highlight_cell,
            current_color=color,
            label=display_label,
        )
        if frame is None:
            return
        try:
            progress_callback(f"cursor_trail|{color}", frame)
        except Exception:
            pass
        last_snapshot = now

    def record_point(position=None, *, force=False):
        if progress_visualizer is None:
            return
        if position is None:
            position = get_cursor_pos()
        local = progress_visualizer.add_mouse_point(position)
        if local is None:
            return
        if not use_monitor:
            emit_snapshot(force=force)

    if use_monitor:
        trail_monitor.set_context(color=color, highlight=current_highlight_cell, label=None)

    trail_callback = (lambda pos: record_point(pos)) if progress_visualizer is not None else None
    smooth_move_to(
        centers[0],
        duration=None,
        jitter=False,
        position_callback=trail_callback,
    )
    record_point(force=True)
    scaled_sleep(0.01)
    mouse_is_down = False
    try:
        if hold:
            abort_if_killed()
            mouse_down()
            mouse_is_down = True
            scaled_sleep(0.005)
        total_steps = len(cells) if cells else 0

        def send_progress(step_index):
            nonlocal last_progress_label, current_highlight_cell
            if (
                progress_callback is None
                or progress_visualizer is None
                or not cells
                or step_index <= 0
                or step_index > total_steps
            ):
                return
            cell = cells[step_index - 1]
            progress_visualizer.mark_cell(cell, color)
            label = f"{color}: {step_index}/{total_steps}"
            frame = progress_visualizer.snapshot(current_cell=cell, current_color=color, label=label)
            try:
                progress_callback(f"path_progress|{color}|{step_index}|{total_steps}", frame)
            except Exception:
                pass
            last_progress_label = label
            current_highlight_cell = cell
            if use_monitor:
                trail_monitor.set_context(highlight=current_highlight_cell, label=label)
            else:
                emit_snapshot(label=label, force=True)

        if total_steps:
            send_progress(1)
            record_point(force=True)

        human_drag_cells(
            cells,
            h_lines=getattr(progress_visualizer, "h_lines", []),
            v_lines=getattr(progress_visualizer, "v_lines", []),
            jitter=True,
            position_callback=trail_callback,
            progress_callback=progress_callback,
            progress_visualizer=progress_visualizer,
            color=color,
            trail_monitor=trail_monitor
        )

        if total_steps:
            send_progress(total_steps)
            record_point(force=True)

        if hold:
            scaled_sleep(0.01)
            abort_if_killed()

    finally:
        if progress_visualizer is not None:
            final_cell = cells[-1] if cells else current_highlight_cell
            record_point(force=True)
            if use_monitor:
                trail_monitor.set_context(highlight=final_cell, label=last_progress_label)
            else:
                emit_snapshot(label=last_progress_label, force=True)
            progress_visualizer.end_trail()
        if hold and mouse_is_down:
            try:
                mouse_up()
            except PuzzleNotAvailable:
                try:
                    interception.mouse_up(button="left")
                except Exception:
                    pass
            except KillSwitchActivated:
                try:
                    interception.mouse_up(button="left")
                except Exception:
                    pass
                raise
            except Exception:
                pass

def execute_single_path(
    color,
    cells,
    h_lines,
    v_lines,
    origin_x,
    origin_y,
    progress_callback=None,
    progress_visualizer=None,
    trail_monitor=None,
):
    if len(cells) < 2:
        print("Path too short, skipping.")
        return
    abort_if_killed()
    centers = path_pixel_centers(cells, h_lines, v_lines, origin_x, origin_y)
    drag_path(
        color,
        cells,
        centers,
        hold=True,
        progress_callback=progress_callback,
        progress_visualizer=progress_visualizer,
        trail_monitor=trail_monitor,
    )

def execute_all_paths(
    solutions,
    h_lines,
    v_lines,
    origin_x,
    origin_y,
    progress_callback=None,
    progress_visualizer=None,
    trail_monitor=None,
):
    abort_if_killed()
    ensure_cursor_not_blocked()
    movement_monitor = MouseMovementMonitor(idle_timeout=MOUSE_STALL_TIMEOUT)
    set_mouse_monitor(movement_monitor)
    movement_monitor.start()
    try:
        remaining = dict(solutions)
        while remaining:
            abort_if_killed()
            ordered = order_paths_nearest_mouse(remaining, h_lines, v_lines, origin_x, origin_y)
            if not ordered:
                break
            color, path, entry_index = ordered[0]
            if entry_index == -1:
                path = list(reversed(path))
            print(f"Executing path for {color}...")
            execute_single_path(
                color,
                path,
                h_lines,
                v_lines,
                origin_x,
                origin_y,
                progress_callback=progress_callback,
                progress_visualizer=progress_visualizer,
                trail_monitor=trail_monitor,
            )
            del remaining[color]
    finally:
        movement_monitor.stop()
        if get_mouse_monitor() is movement_monitor:
            clear_mouse_monitor()

def run_one_puzzle(return_debug=False, progress_callback=None, silent_no_puzzle=False):
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
    debug_image = None
    monitor = None
    cursor_monitor: Optional[CursorTrailMonitor] = None
    try:
        try:
            screenshot = capture_region()
        except Exception as exc:
            raise PuzzleNotAvailable(f"Failed to capture puzzle region: {exc}") from exc
        bbox = find_puzzle_bbox(screenshot)
        if not bbox:
            if not silent_no_puzzle:
                print("Puzzle not found.")
                _emit("no_puzzle", None)
            return _result(False, None)
        x, y, w, h = bbox
        monitor = PuzzlePresenceMonitor((x, y, w, h))
        set_puzzle_monitor(monitor)
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
        progress_visualizer = None
        if progress_callback is not None:
            progress_visualizer = PathProgressVisualizer(
                debug.copy(),
                h_lines,
                v_lines,
                color_map,
                origin=(origin_x, origin_y),
            )
            cursor_monitor = CursorTrailMonitor(progress_visualizer, progress_callback)
            cursor_monitor.start()
        print("\nDetected points:")
        for d in detected:
            print(f"Row {d[0]}, Col {d[1]} -> {d[2]}")
        print(f"Total unique paths needed: {len(color_map)}")
        solutions = solve_paths_flexible(rows, cols, detected)
        if solutions:
            total_cells_before = sum(len(path) for path in solutions.values())
            optimized_solutions, optimized = optimize_solution_paths(solutions, rows, cols, detected)
            solutions = optimized_solutions
            if optimized:
                total_cells_after = sum(len(path) for path in solutions.values())
                print(f"Optimized total path length: {total_cells_before} -> {total_cells_after}")
            preview = debug.copy()
            draw_paths(preview, solutions, h_lines, v_lines, color_map)
            debug_image = preview.copy()
            _emit("paths_ready", preview.copy())
            if progress_visualizer is not None:
                progress_visualizer.update_base(preview, origin=(origin_x, origin_y), reset_trails=True)
            for color, path in solutions.items():
                print(f"Path for {color}: {path}")
            execute_all_paths(
                solutions,
                h_lines,
                v_lines,
                origin_x,
                origin_y,
                progress_callback=progress_callback,
                progress_visualizer=progress_visualizer,
                trail_monitor=cursor_monitor,
            )
        else:
            print("Solver could not find a valid set of paths.")
            _emit("no_solution", debug.copy())
        print("Puzzle Complete/Canceled!")
        return _result(True, debug_image)
    except PuzzleNotAvailable as exc:
        reason = str(exc).strip()
        message = "Puzzle closed or canceled."
        if reason:
            message = f"{message} ({reason})"
        print(message)
        _emit("puzzle_missing", None)
        return _result(False, debug_image)
    except MouseBlockedAtStart:
        _emit("mouse_blocked", None)
        raise
    except MouseMovementStalled as exc:
        message = str(exc).strip() or "Mouse movement stalled. Canceling puzzle."
        print(message)
        _emit("mouse_stalled", None)
        return _result(False, debug_image)
    except KillSwitchActivated:
        print("Kill switch engaged. Canceling current puzzle.")
        _emit("killed", None)
        raise
    finally:
        if cursor_monitor is not None:
            cursor_monitor.stop()
        clear_puzzle_monitor()
        clear_mouse_monitor()

def main():
    print(f"Ready. Press PgDn for one puzzle, PgUp for continuous mode, Home for auto-complete. Press end to quit after current action. Press {KILL_SWITCH_KEY} to halt instantly.")
    while True:
        event = keyboard.read_event(suppress=False)
        if getattr(event, "event_type", None) != keyboard.KEY_DOWN:
            continue
        key = getattr(event, "name", None)
        if not key:
            continue
        key = key.lower()
        if key == KILL_SWITCH_KEY:
            continue
        if key == "page down":
            print("Running single puzzle...")
            try:
                reset_kill_switch()
                run_one_puzzle()
            except MouseBlockedAtStart as exc:
                _ = exc
                reset_kill_switch()
            except KillSwitchActivated:
                print("Kill switch engaged. Single puzzle canceled.")
                reset_kill_switch()
        elif key == "page up":
            print("Running continuous mode...")
            while True:
                try:
                    reset_kill_switch()
                    success = run_one_puzzle(silent_no_puzzle=True)
                except MouseBlockedAtStart as exc:
                    _ = exc
                    reset_kill_switch()
                    continue
                except KillSwitchActivated:
                    print("Kill switch engaged. Stopping continuous mode.")
                    reset_kill_switch()
                    break
                if not success:
                    print("No puzzle detected, stopping continuous mode.")
                    print("Generator Complete/Canceled!")
                    break
                try:
                    scaled_sleep(0.35)
                except KillSwitchActivated:
                    print("Kill switch engaged. Stopping continuous mode.")
                    reset_kill_switch()
                    break
                if keyboard.is_pressed("end"):
                    reset_kill_switch()
                    print("end pressed, stopping continuous mode.")
                    return
            reset_kill_switch()
        elif key == "home":
            print("Running auto-complete mode...")
            puzzles_completed = 0
            idle_message_printed = False
            while True:
                if keyboard.is_pressed("end"):
                    reset_kill_switch()
                    print("end pressed, stopping auto-complete mode.")
                    break
                try:
                    reset_kill_switch()
                    success = run_one_puzzle(silent_no_puzzle=True)
                except MouseBlockedAtStart as exc:
                    _ = exc
                    reset_kill_switch()
                    continue
                except KillSwitchActivated:
                    print("Kill switch engaged. Stopping auto-complete mode.")
                    reset_kill_switch()
                    break
                if success:
                    puzzles_completed += 1
                    idle_message_printed = False
                    print(f"Auto-complete solved puzzle #{puzzles_completed}.")
                    try:
                        scaled_sleep(0.35)
                    except KillSwitchActivated:
                        print("Kill switch engaged. Stopping auto-complete mode.")
                        reset_kill_switch()
                        break
                    if keyboard.is_pressed("end"):
                        reset_kill_switch()
                        print("end pressed, stopping auto-complete mode.")
                        break
                    continue
                if not idle_message_printed:
                    print("Waiting for puzzle to appear...")
                    idle_message_printed = True
                try:
                    scaled_sleep(AUTO_IDLE_DELAY)
                except KillSwitchActivated:
                    print("Kill switch engaged. Stopping auto-complete mode.")
                    reset_kill_switch()
                    break
            print("Auto-complete mode stopped.")
            reset_kill_switch()
        elif key == "end":
            print("Exiting script.")
            break
if __name__ == "__main__":
    main()

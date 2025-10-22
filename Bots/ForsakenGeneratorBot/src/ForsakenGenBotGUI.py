# type: ignore[import]
import threading
import queue
import contextlib
import traceback
import time
from typing import Optional, Tuple, Dict
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import keyboard
from PIL import Image, ImageTk, ImageOps
import ForsakenGenBot as bot
STATUS_STYLES = {
    "ready": {"bg": "#142d26", "fg": "#5af2bb"},
    "running": {"bg": "#132649", "fg": "#6fb4ff"},
    "stopping": {"bg": "#2d1b2f", "fg": "#f59ab4"},
    "warning": {"bg": "#2f260f", "fg": "#f8d16b"},
    "error": {"bg": "#311a1f", "fg": "#ff8a8a"},
}
DEFAULT_HOTKEYS = {
    "single": "page down",
    "continuous": "page up",
    "auto": "home",
    "stop": "end",
    "kill": bot.get_kill_switch_key(),
}
class QueueWriter:
    def __init__(self, output_queue: queue.Queue):
        self.queue = output_queue
        self._buffer = ""

    def write(self, message: str) -> None:
        if not message:
            return
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.queue.put(line + "\n")

    def flush(self) -> None:
        if self._buffer:
            self.queue.put(self._buffer)
            self._buffer = ""


class ForsakenBotGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Forsaken Generator Bot")
        self.preview_max_size = (720, 520)
        self.root.geometry("960x640")
        self.root.minsize(820, 560)
        self._init_styles()
        self.log_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pending_close = False
        self.running_mode: Optional[str] = None
        self.status_state = "ready"
        settings = bot.get_settings()
        self.status_var = tk.StringVar(value="Ready")
        self.speed_var = tk.DoubleVar(value=settings["speed"])
        self.jitter_var = tk.DoubleVar(value=settings["jitter_scale"])
        self.smoothness_var = tk.DoubleVar(value=settings["smoothness"])
        self.hotkeys: Dict[str, str] = dict(DEFAULT_HOTKEYS)
        self.hotkey_vars: Dict[str, tk.StringVar] = {
            action: tk.StringVar(value=hotkey)
            for action, hotkey in self.hotkeys.items()
        }
        self.hotkey_handles: Dict[str, int] = {}
        self.hotkey_buttons: Dict[str, ttk.Button] = {}
        self._pending_hotkey_action: Optional[str] = None
        self._capture_hook = None
        self._capture_previous: Optional[Dict[str, str]] = None
        self.preview_placeholder = "Waiting for puzzle..."
        self.current_puzzle_photo: Optional[ImageTk.PhotoImage] = None
        self.preview_status_var = tk.StringVar(value=self.preview_placeholder)
        self.preview_source_image: Optional[Image.Image] = None
        self.preview_last_message = self.preview_placeholder
        self.preview_last_size: Tuple[int, int] = (0, 0)
        self._has_confirmed_puzzle = False
        self._waiting_displayed = False
        self._auto_solver_logging = False
        self._build_ui()
        self._apply_status("Ready", "ready")
        self._set_puzzle_preview(None, self.preview_placeholder)
        self.process_log_queue()
        self._register_hotkeys()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_styles(self) -> None:
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.colors = {
            "background": "#0f111a",
            "panel": "#171c2b",
            "panel_alt": "#111624",
            "panel_border": "#20273b",
            "preview_bg": "#0a0f1d",
            "accent": "#3b82f6",
            "accent_hover": "#4c8bff",
            "accent_active": "#2e6de0",
            "text": "#e7ebff",
            "muted": "#8d9abc",
            "heading": "#cad6ff",
            "button_disabled": "#1d2435",
            "secondary_bg": "#1f2739",
            "secondary_hover": "#262f45",
            "danger": "#f87171",
            "danger_hover": "#fb8a8a",
            "danger_active": "#dc5a5a",
        }
        self.root.configure(bg=self.colors["background"])
        self.style.configure("Root.TFrame", background=self.colors["background"])
        self.style.configure("Panel.TFrame", background=self.colors["panel"])
        self.style.configure(
            "Panel.TLabelframe",
            background=self.colors["panel"],
            borderwidth=1,
            relief="solid",
            darkcolor=self.colors["panel_border"],
            lightcolor=self.colors["panel_border"],
        )
        self.style.configure(
            "Panel.TLabelframe.Label",
            background=self.colors["panel"],
            foreground=self.colors["heading"],
            font=("Segoe UI", 10, "bold"),
            highlightthickness=0,
        )
        self.style.configure(
            "Log.TLabelframe",
            background=self.colors["panel_alt"],
            borderwidth=1,
            relief="solid",
            darkcolor=self.colors["panel_border"],
            lightcolor=self.colors["panel_border"],
        )
        self.style.configure(
            "Log.TLabelframe.Label",
            background=self.colors["panel_alt"],
            foreground=self.colors["heading"],
            font=("Segoe UI", 10, "bold"),
        )
        self.style.configure(
            "Heading.TLabel",
            background=self.colors["background"],
            foreground=self.colors["heading"],
            font=("Segoe UI Semibold", 11),
        )
        self.style.configure(
            "PanelHeading.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["heading"],
            font=("Segoe UI", 10, "bold"),
        )
        self.style.configure(
            "PanelInfo.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 9),
        )
        self.style.configure(
            "PanelValue.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["text"],
            font=("Segoe UI", 10, "bold"),
        )
        self.style.configure(
            "PreviewStatus.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 9, "italic"),
        )
        self.style.configure(
            "HotkeyDisplay.TLabel",
            background=self.colors["panel_alt"],
            foreground=self.colors["text"],
            font=("Segoe UI", 9, "bold"),
            padding=(8, 4),
            borderwidth=1,
            relief="solid",
        )
        self.style.configure("Preview.TFrame", background=self.colors["preview_bg"])
        self.style.configure(
            "Settings.Horizontal.TScale",
            background=self.colors["panel"],
            troughcolor=self.colors["panel_alt"],
            bordercolor=self.colors["panel_border"],
            lightcolor=self.colors["panel_border"],
            darkcolor=self.colors["panel_border"],
        )
        self.style.configure("Settings.TSeparator", background=self.colors["panel_border"])
        self.style.configure(
            "Primary.TButton",
            background=self.colors["accent"],
            foreground="#f5f7ff",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 6),
            borderwidth=0,
            relief="flat",
        )
        self.style.map(
            "Primary.TButton",
            background=[
                ("disabled", self.colors["button_disabled"]),
                ("pressed", self.colors["accent_active"]),
                ("active", self.colors["accent_hover"]),
            ],
            foreground=[("disabled", self.colors["muted"])],
        )
        self.style.configure(
            "Secondary.TButton",
            background=self.colors["secondary_bg"],
            foreground=self.colors["heading"],
            font=("Segoe UI", 10),
            padding=(10, 6),
            borderwidth=0,
            relief="flat",
        )
        self.style.map(
            "Secondary.TButton",
            background=[
                ("disabled", self.colors["button_disabled"]),
                ("active", self.colors["secondary_hover"]),
            ],
            foreground=[("disabled", self.colors["muted"])],
        )
        self.style.configure(
            "Danger.TButton",
            background=self.colors["danger"],
            foreground="#fff6f7",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 6),
            borderwidth=0,
            relief="flat",
        )
        self.style.map(
            "Danger.TButton",
            background=[
                ("disabled", self.colors["button_disabled"]),
                ("active", self.colors["danger_hover"]),
                ("pressed", self.colors["danger_active"]),
            ],
            foreground=[("disabled", self.colors["muted"])],
        )
        self.style.configure(
            "HotkeyChange.TButton",
            background=self.colors["panel_alt"],
            foreground=self.colors["accent"],
            font=("Segoe UI", 9),
            padding=(8, 4),
            borderwidth=1,
            relief="solid",
        )
        self.style.map(
            "HotkeyChange.TButton",
            background=[
                ("disabled", self.colors["button_disabled"]),
                ("active", self.colors["accent"]),
            ],
            foreground=[
                ("disabled", self.colors["muted"]),
                ("active", "#f5faff"),
            ],
        )
        self.style.configure(
            "Vertical.TScrollbar",
            background=self.colors["panel_alt"],
            troughcolor=self.colors["panel"],
            bordercolor=self.colors["panel"],
            arrowcolor=self.colors["muted"],
        )

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=(12, 10, 12, 12), style="Root.TFrame")
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(2, weight=1)
        status_frame = ttk.Frame(container, style="Root.TFrame")
        status_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        status_frame.columnconfigure(1, weight=1)
        ttk.Label(status_frame, text="Status:", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            width=30,
            bd=0,
            relief="flat",
            padx=12,
            pady=4,
            font=("Segoe UI", 10, "bold"),
        )
        self.status_label.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.status_label.configure(bg=STATUS_STYLES["ready"]["bg"], fg=STATUS_STYLES["ready"]["fg"])
        buttons_frame = ttk.Frame(container, style="Root.TFrame")
        buttons_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        buttons_frame.columnconfigure((0, 1, 2, 3, 4), weight=1)
        self.single_button = ttk.Button(
            buttons_frame,
            text="Run Single Puzzle",
            command=self.start_single,
            style="Primary.TButton",
        )
        self.single_button.grid(row=0, column=0, padx=5, sticky="ew")
        self.continuous_button = ttk.Button(
            buttons_frame,
            text="Run Continuous",
            command=self.start_continuous,
            style="Primary.TButton",
        )
        self.continuous_button.grid(row=0, column=1, padx=5, sticky="ew")
        self.auto_button = ttk.Button(
            buttons_frame,
            text="Auto Complete",
            command=self.start_auto,
            style="Primary.TButton",
        )
        self.auto_button.grid(row=0, column=2, padx=5, sticky="ew")
        self.stop_button = ttk.Button(
            buttons_frame,
            text="Stop",
            command=self.stop,
            state="disabled",
            style="Danger.TButton",
        )
        self.stop_button.grid(row=0, column=3, padx=5, sticky="ew")
        self.clear_log_button = ttk.Button(
            buttons_frame,
            text="Clear Log",
            command=self.clear_log,
            style="Secondary.TButton",
        )
        self.clear_log_button.grid(row=0, column=4, padx=5, sticky="ew")
        main_area = ttk.Frame(container, style="Root.TFrame")
        main_area.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        main_area.columnconfigure(0, weight=5)
        main_area.columnconfigure(1, weight=2)
        main_area.rowconfigure(0, weight=1)
        left_column = ttk.Frame(main_area, style="Root.TFrame")
        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_column.columnconfigure(0, weight=1)
        left_column.rowconfigure(0, weight=6)
        left_column.rowconfigure(1, weight=2)
        preview_frame = ttk.LabelFrame(left_column, text="Current Puzzle", padding=(10, 10), style="Panel.TLabelframe")
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=0)
        preview_frame.rowconfigure(1, weight=1)
        status_label = ttk.Label(
            preview_frame,
            textvariable=self.preview_status_var,
            anchor="w",
            padding=(6, 3),
            style="PreviewStatus.TLabel",
        )
        status_label.grid(row=0, column=0, sticky="ew")
        image_container = ttk.Frame(preview_frame, style="Preview.TFrame")
        image_container.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 10))
        image_container.columnconfigure(0, weight=1)
        image_container.rowconfigure(0, weight=1)
        image_container.bind("<Configure>", self._on_preview_resize)
        self.puzzle_label = tk.Label(
            image_container,
            text="",
            anchor="center",
            justify="center",
            bg=self.colors["preview_bg"],
            fg=self.colors["text"],
            font=("Segoe UI", 10),
            wraplength=self.preview_max_size[0] - 40,
            highlightthickness=0,
            bd=0,
        )
        self.puzzle_label.grid(row=0, column=0, sticky="nsew")
        log_frame = ttk.LabelFrame(left_column, text="Debug Output", padding=10, style="Log.TLabelframe")
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            state="disabled",
            font=("Consolas", 10),
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(
            height=8,
            background=self.colors["panel_alt"],
            foreground=self.colors["text"],
            insertbackground=self.colors["text"],
            highlightthickness=0,
            borderwidth=0,
            relief="flat",
            selectbackground=self.colors["accent"],
            inactiveselectbackground=self.colors["accent"],
            selectforeground=self.colors["background"],
        )
        settings_frame = ttk.LabelFrame(main_area, text="Settings", padding=(12, 10), style="Panel.TLabelframe")
        settings_frame.grid(row=0, column=1, sticky="nsew")
        settings_frame.columnconfigure(0, weight=1)
        settings_frame.columnconfigure(1, weight=0)
        ttk.Label(settings_frame, text="Speed Multiplier", style="PanelInfo.TLabel").grid(row=0, column=0, sticky="w", pady=(4, 0))
        speed_scale = ttk.Scale(
            settings_frame,
            from_=0.5,
            to=2.5,
            variable=self.speed_var,
            command=self.on_speed_change,
            style="Settings.Horizontal.TScale",
        )
        speed_scale.grid(row=1, column=0, sticky="ew")
        self.speed_value = ttk.Label(settings_frame, text=f"{self.speed_var.get():.2f}x", style="PanelValue.TLabel")
        self.speed_value.grid(row=1, column=1, padx=(6, 0))
        ttk.Label(settings_frame, text="Jitteriness", style="PanelInfo.TLabel").grid(row=2, column=0, sticky="w", pady=(12, 0))
        jitter_scale = ttk.Scale(
            settings_frame,
            from_=0.0,
            to=3.0,
            variable=self.jitter_var,
            command=self.on_jitter_change,
            style="Settings.Horizontal.TScale",
        )
        jitter_scale.grid(row=3, column=0, sticky="ew")
        self.jitter_value = ttk.Label(settings_frame, text=f"{self.jitter_var.get():.2f}", style="PanelValue.TLabel")
        self.jitter_value.grid(row=3, column=1, padx=(6, 0))
        ttk.Label(settings_frame, text="Smoothness", style="PanelInfo.TLabel").grid(row=4, column=0, sticky="w", pady=(12, 0))
        smooth_scale = ttk.Scale(
            settings_frame,
            from_=0.5,
            to=2.5,
            variable=self.smoothness_var,
            command=self.on_smoothness_change,
            style="Settings.Horizontal.TScale",
        )
        smooth_scale.grid(row=5, column=0, sticky="ew")
        self.smooth_value = ttk.Label(settings_frame, text=f"{self.smoothness_var.get():.2f}", style="PanelValue.TLabel")
        self.smooth_value.grid(row=5, column=1, padx=(6, 0))
        ttk.Separator(settings_frame, orient="horizontal", style="Settings.TSeparator").grid(row=6, column=0, columnspan=2, sticky="ew", pady=(18, 10))
        ttk.Label(settings_frame, text="Hotkeys", style="PanelHeading.TLabel").grid(row=7, column=0, columnspan=2, sticky="w")
        keybinds = [
            ("single", "Run Single Puzzle"),
            ("continuous", "Run Continuous"),
            ("auto", "Auto Complete"),
            ("stop", "Stop"),
            ("kill", "Kill Switch"),
        ]
        for offset, (action, label_text) in enumerate(keybinds):
            row_index = 8 + offset * 2
            ttk.Label(settings_frame, text=label_text, style="PanelInfo.TLabel").grid(row=row_index, column=0, sticky="w", pady=(6, 0))
            row_frame = ttk.Frame(settings_frame, style="Panel.TFrame")
            row_frame.grid(row=row_index + 1, column=0, columnspan=2, sticky="ew")
            row_frame.columnconfigure(0, weight=1)
            display = ttk.Label(row_frame, textvariable=self.hotkey_vars[action], style="HotkeyDisplay.TLabel", anchor="w")
            display.grid(row=0, column=0, sticky="ew")
            change_button = ttk.Button(
                row_frame,
                text="Change",
                command=lambda act=action: self._begin_hotkey_capture(act),
                width=9,
                style="HotkeyChange.TButton",
            )
            change_button.grid(row=0, column=1, padx=(8, 0))
            self.hotkey_buttons[action] = change_button
        settings_frame.rowconfigure(20, weight=1)
        
    def process_log_queue(self) -> None:
        try:
            while True:
                message = self.log_queue.get_nowait()
                self._append_to_log(message)
        except queue.Empty:
            pass
        self.root.after(80, self.process_log_queue)

    def _append_to_log(self, message: str) -> None:
        if (
            self.running_mode == "auto"
            and not self._auto_solver_logging
            and message.strip()
            and not message.lstrip().startswith("[GUI]")
            and not message.lstrip().lower().startswith("puzzle closed or canceled")
        ):
            return
        self.log_text.configure(state="normal")
        timestamp = time.strftime("[%H:%M:%S] ")
        if message.endswith("\n"):
            content = message.rstrip("\n")
            self.log_text.insert(tk.END, timestamp + content + "\n")
        else:
            self.log_text.insert(tk.END, timestamp + message)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def start_single(self) -> None:
        self._start_worker("single")

    def start_continuous(self) -> None:
        self._start_worker("continuous")

    def start_auto(self) -> None:
        self._start_worker("auto")

    def _start_worker(self, mode: str) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.log_queue.put("[GUI] Bot is already running.\n")
            return
        self.stop_event.clear()
        self.running_mode = mode
        self._apply_status("Preparing...", "running")
        self.set_running_state(True)
        mode_label = {"single": "single", "continuous": "continuous", "auto": "auto-complete"}.get(mode, mode)
        if mode == "auto":
            self._has_confirmed_puzzle = False
            self._waiting_displayed = True
            self._set_preview_status("Waiting for puzzle...")
            self._auto_solver_logging = False
        else:
            self._waiting_displayed = False
        self.log_queue.put(f"[GUI] Starting {mode_label} mode.\n")
        self.worker_thread = threading.Thread(target=self._run_worker, args=(mode,), daemon=True)
        self.worker_thread.start()

    def _run_worker(self, mode: str) -> None:
        writer = QueueWriter(self.log_queue)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                if mode == "single":
                    self._update_status_async("Executing puzzle...", "running")
                    self._set_preview_status("Capturing puzzle...")
                    progress = lambda stage, img: self._handle_progress(stage, img)
                    try:
                        success, debug_img = bot.run_one_puzzle(
                            return_debug=True,
                            progress_callback=progress,
                        )
                    except bot.MouseBlockedAtStart as exc:
                        self._handle_blocked_cursor_event(exc)
                        return
                    if debug_img is None and not success:
                        self._set_preview_status("No puzzle detected.")
                    if success:
                        self.log_queue.put("[GUI] Puzzle completed.\n")
                        self._update_status_async("Ready", "ready")
                    else:
                        self.log_queue.put("[GUI] No puzzle detected.\n")
                        self._update_status_async("No puzzle detected", "warning")
                elif mode == "auto":
                    self._run_auto_loop()
                else:
                    self._run_continuous_loop()
        except bot.KillSwitchActivated:
            bot.reset_kill_switch()
            self.log_queue.put("[GUI] Kill switch engaged. Stopping bot.\n")
            self._update_status_async("Kill switch engaged", "warning")
            self._set_preview_status("Kill switch engaged.")
        except Exception:
            self.log_queue.put("[GUI] An unexpected error occurred.\n")
            self.log_queue.put(traceback.format_exc())
            self._update_status_async("Error", "error")
        finally:
            writer.flush()
            self.stop_event.clear()
            self._set_running_state_async(False)
            self.running_mode = None

    def _run_continuous_loop(self) -> None:
        self._update_status_async("Continuous mode active...", "running")
        puzzles_completed = 0
        final_status: Optional[Tuple[str, str]] = None
        while not self.stop_event.is_set():
            self._auto_solver_logging = False
            self._set_preview_status("Capturing puzzle...")
            progress = lambda stage, img: self._handle_progress(stage, img)
            try:
                success, debug_img = bot.run_one_puzzle(
                    return_debug=True,
                    progress_callback=progress,
                )
            except bot.MouseBlockedAtStart as exc:
                self._handle_blocked_cursor_event(exc)
                if self.stop_event.wait(0.35):
                    final_status = ("Stopped", "ready")
                    break
                if not self.stop_event.is_set():
                    self._update_status_async("Continuous mode active...", "running")
                    self._set_preview_status("Capturing puzzle...")
                continue
            except bot.KillSwitchActivated:
                bot.reset_kill_switch()
                self.log_queue.put("[GUI] Kill switch engaged, stopping continuous mode.\n")
                self._update_status_async("Kill switch engaged", "warning")
                self._set_preview_status("Kill switch engaged.")
                final_status = ("Kill switch engaged", "warning")
                break
            if debug_img is None and not success:
                self._set_preview_status("No puzzle detected.")
            if not success:
                self.log_queue.put("[GUI] No puzzle detected, stopping continuous mode.\n")
                final_status = ("No puzzle detected", "warning")
                break
            puzzles_completed += 1
            self.log_queue.put(f"[GUI] Puzzle {puzzles_completed} completed.\n")
            if self.stop_event.is_set():
                final_status = ("Stopped", "ready")
                break
            try:
                bot.scaled_sleep(0.35)
            except bot.KillSwitchActivated:
                bot.reset_kill_switch()
                self.log_queue.put("[GUI] Kill switch engaged, stopping continuous mode.\n")
                self._update_status_async("Kill switch engaged", "warning")
                self._set_preview_status("Kill switch engaged.")
                final_status = ("Kill switch engaged", "warning")
                break
        if final_status is None:
            if self.stop_event.is_set():
                final_status = ("Stopped", "ready")
            elif puzzles_completed > 0:
                final_status = ("Ready", "ready")
            else:
                final_status = ("Ready", "ready")
        self._update_status_async(*final_status)

    def _run_auto_loop(self) -> None:
        self._update_status_async("Auto-complete mode active...", "running")
        puzzles_completed = 0
        waiting_logged = False
        final_status: Optional[Tuple[str, str]] = None
        idle_delay = getattr(bot, "AUTO_IDLE_DELAY", 0.5)
        while not self.stop_event.is_set():
            self._auto_solver_logging = False
            progress = lambda stage, img: self._handle_progress(stage, img)
            try:
                success, debug_img = bot.run_one_puzzle(
                    return_debug=True,
                    progress_callback=progress,
                    silent_no_puzzle=True,
                )
            except bot.MouseBlockedAtStart as exc:
                self._handle_blocked_cursor_event(exc)
                waiting_logged = False
                self._has_confirmed_puzzle = False
                self._auto_solver_logging = False
                if self.stop_event.wait(idle_delay):
                    final_status = ("Stopped", "ready")
                    break
                if not self.stop_event.is_set():
                    self._update_status_async("Auto-complete mode active...", "running")
                    self._set_preview_status("Waiting for puzzle...")
                continue
            except bot.KillSwitchActivated:
                bot.reset_kill_switch()
                self.log_queue.put("[GUI] Kill switch engaged, stopping auto-complete.\n")
                self._update_status_async("Kill switch engaged", "warning")
                self._set_preview_status("Kill switch engaged.")
                final_status = ("Kill switch engaged", "warning")
                break
            if success:
                puzzles_completed += 1
                waiting_logged = False
                self._waiting_displayed = False
                self.log_queue.put(f"[GUI] Auto-complete solved puzzle {puzzles_completed}.\n")
                if self.stop_event.is_set():
                    final_status = ("Stopped", "ready")
                    break
                try:
                    bot.scaled_sleep(0.35)
                except bot.KillSwitchActivated:
                    bot.reset_kill_switch()
                    self.log_queue.put("[GUI] Kill switch engaged, stopping auto-complete.\n")
                    self._update_status_async("Kill switch engaged", "warning")
                    self._set_preview_status("Kill switch engaged.")
                    final_status = ("Kill switch engaged", "warning")
                    break
                continue
            had_puzzle = self._has_confirmed_puzzle
            self._has_confirmed_puzzle = False
            self._auto_solver_logging = False
            if debug_img is not None and had_puzzle:
                if self.running_mode == "auto":
                    self._auto_solver_logging = True
                self._waiting_displayed = False
                self._update_preview_async(debug_img, "Detection issue. Waiting for puzzle...")
            else:
                if (not self._waiting_displayed) or (self.preview_last_message != "Waiting for puzzle..."):
                    self._set_preview_status("Waiting for puzzle...")
                self._waiting_displayed = True
            if not waiting_logged:
                self.log_queue.put("[GUI] Auto-complete waiting for puzzle...\n")
                waiting_logged = True
            if self.stop_event.is_set():
                final_status = ("Stopped", "ready")
                break
            try:
                bot.scaled_sleep(idle_delay)
            except bot.KillSwitchActivated:
                bot.reset_kill_switch()
                self.log_queue.put("[GUI] Kill switch engaged, stopping auto-complete.\n")
                self._update_status_async("Kill switch engaged", "warning")
                self._set_preview_status("Kill switch engaged.")
                final_status = ("Kill switch engaged", "warning")
                break
        if final_status is None:
            if self.stop_event.is_set():
                final_status = ("Stopped", "ready")
            elif puzzles_completed:
                final_status = ("Auto-complete idle", "ready")
            else:
                final_status = ("Ready", "ready")
        self._update_status_async(*final_status)

    def stop(self) -> None:
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return
        self.log_queue.put("[GUI] Stop requested by user.\n")
        self.stop_event.set()
        self._update_status_async("Stopping...", "stopping")

    def set_running_state(self, running: bool) -> None:
        if running:
            self.single_button.configure(state="disabled")
            self.continuous_button.configure(state="disabled")
            self.auto_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
        else:
            self.single_button.configure(state="normal")
            self.continuous_button.configure(state="normal")
            self.auto_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def _set_running_state_async(self, running: bool) -> None:
        self.root.after(0, lambda: self.set_running_state(running))

    def on_speed_change(self, value: str) -> None:
        try:
            speed = float(value)
        except (TypeError, ValueError):
            return
        bot.set_speed(speed)
        self.speed_value.configure(text=f"{bot.get_settings()['speed']:.2f}x")

    def on_jitter_change(self, value: str) -> None:
        try:
            jitter = float(value)
        except (TypeError, ValueError):
            return
        bot.set_jitter_scale(jitter)
        self.jitter_value.configure(text=f"{bot.get_settings()['jitter_scale']:.2f}")

    def on_smoothness_change(self, value: str) -> None:
        try:
            smoothness = float(value)
        except (TypeError, ValueError):
            return
        bot.set_smoothness(smoothness)
        self.smooth_value.configure(text=f"{bot.get_settings()['smoothness']:.2f}")

    def _preview_updates_allowed(self) -> bool:
        if self.running_mode != "auto":
            return True
        return self._auto_solver_logging or self._has_confirmed_puzzle

    def _set_preview_status(self, message: str) -> None:
        self.preview_last_message = message
        def updater(msg=message):
            self.preview_status_var.set(msg)
            if self.preview_source_image is None:
                self.puzzle_label.configure(text=msg, bg=self.colors["preview_bg"], fg=self.colors["text"])
        self.root.after(0, updater)

    def _handle_blocked_cursor_event(self, exc: Optional[BaseException] = None, *, log: bool = True) -> None:
        position = getattr(exc, "position", None)
        if position is not None:
            try:
                px, py = position  # type: ignore[misc]
                position_text = f"({int(px)}, {int(py)})"
            except Exception:
                position_text = str(position)
        else:
            position_text = "a blocked position"
        if log:
            self.log_queue.put(f"[GUI] Cursor blocked at {position_text}. Move the mouse to continue.\n")
        self._update_status_async("Cursor blocked", "warning")
        self._set_preview_status("Cursor blocked. Move the mouse.")

    def _update_preview_async(self, image, message: Optional[str] = None) -> None:
        if not self._preview_updates_allowed():
            return
        if image is not None:
            try:
                payload = image.copy()
            except AttributeError:
                payload = image
        else:
            payload = None
        self.root.after(0, lambda img=payload, msg=message: self._set_puzzle_preview(img, msg))

    def _handle_progress(self, stage: str, image) -> None:
        stage_messages = {
            "puzzle_captured": "Puzzle captured.",
            "grid_detected": "Grid detected.",
            "grid_failed": "Grid detection failed.",
            "points_detected": "Points detected.",
            "paths_ready": "Paths ready. Executing...",
            "no_solution": "No valid solution found.",
            "killed": "Kill switch engaged.",
            "no_puzzle": "No puzzle detected.",
            "puzzle_missing": "Puzzle disappeared. Waiting...",
            "mouse_blocked": "Cursor blocked. Move the mouse.",
        }
        if stage == "puzzle_captured":
            if self.running_mode == "auto":
                self._auto_solver_logging = False
                self._has_confirmed_puzzle = False
            else:
                self._has_confirmed_puzzle = True
            self._waiting_displayed = False
        elif stage in {"puzzle_missing", "killed"}:
            if self.running_mode == "auto":
                self._auto_solver_logging = False
            self._has_confirmed_puzzle = False
        elif stage == "mouse_blocked":
            if self.running_mode == "auto":
                self._auto_solver_logging = False
                self._has_confirmed_puzzle = False
            self._handle_blocked_cursor_event(log=False)
            return
        elif stage == "no_puzzle":
            if self.running_mode == "auto":
                self._auto_solver_logging = False
            self._has_confirmed_puzzle = False
            if self.running_mode in {"auto", "continuous"}:
                return
        if stage == "no_solution" and self.running_mode == "auto":
            self._auto_solver_logging = False
        if stage == "paths_ready" and self.running_mode == "auto":
            self._auto_solver_logging = True
        if stage.startswith("path_progress") and self.running_mode == "auto":
            self._has_confirmed_puzzle = True
            self._auto_solver_logging = True
        if stage == "no_puzzle":
            self._set_preview_status("No puzzle detected.")
            return
        if stage == "killed":
            self._set_preview_status("Kill switch engaged.")
            return
        if stage.startswith("path_progress"):
            parts = stage.split("|")
            color = parts[1] if len(parts) > 1 else None
            step = parts[2] if len(parts) > 2 else None
            total = parts[3] if len(parts) > 3 else None
            if step and total and step.isdigit() and total.isdigit():
                status = f"Drawing {color} ({step}/{total})" if color else f"Drawing path ({step}/{total})"
            else:
                status = f"Drawing {color}" if color else "Drawing path"
            if image is not None:
                self._update_preview_async(image, status)
            else:
                self._set_preview_status(status)
            return
        message = stage_messages.get(stage)
        if stage == "puzzle_missing":
            msg = stage_messages.get(stage) or "Puzzle disappeared. Waiting..."
            self._waiting_displayed = True
            self._set_preview_status(msg)
            return
        if stage == "paths_ready":
            if self.running_mode == "auto":
                self._auto_solver_logging = True
            self._has_confirmed_puzzle = True
            if image is not None:
                self._update_preview_async(image, message or "Paths ready. Executing...")
            elif message:
                self._set_preview_status(message)
            return
        if message:
            self._set_preview_status(message)
    def _set_puzzle_preview(self, image, message: Optional[str] = None) -> None:
        if self.running_mode == "auto" and not self._preview_updates_allowed():
            return
        self._has_confirmed_puzzle = image is not None
        if self.running_mode == "auto":
            self._waiting_displayed = image is None
        else:
            self._waiting_displayed = False
        if image is None:
            if self.running_mode == "auto" and not self._auto_solver_logging:
                return
            self.preview_source_image = None
            self.current_puzzle_photo = None
            display_text = message or self.preview_placeholder
            self.puzzle_label.configure(image="", text=display_text, bg=self.colors["preview_bg"], fg=self.colors["text"])
            self.puzzle_label.image = None
            self._set_preview_status(display_text)
            return
        if isinstance(image, Image.Image):
            pil_original = image.copy()
        else:
            try:
                rgb_image = image[:, :, ::-1]
            except Exception:
                rgb_image = image
            pil_original = Image.fromarray(rgb_image)
        self.preview_source_image = pil_original
        self._refresh_preview_image()
        self.puzzle_label.configure(bg=self.colors["preview_bg"])
        status_text = message or "Puzzle preview ready."
        self._set_preview_status(status_text)

    def _refresh_preview_image(self) -> None:
        if self.preview_source_image is None:
            return
        target = ImageOps.contain(self.preview_source_image, self.preview_max_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(target)
        self.current_puzzle_photo = photo
        self.puzzle_label.configure(image=photo, text="", wraplength=max(120, self.preview_max_size[0] - 40), bg=self.colors["preview_bg"])
        self.puzzle_label.image = photo
        self.preview_last_size = self.preview_max_size

    def _on_preview_resize(self, event) -> None:
        width = max(120, event.width - 16)
        height = max(160, event.height - 16)
        new_size = (width, height)
        if abs(new_size[0] - self.preview_max_size[0]) < 4 and abs(new_size[1] - self.preview_max_size[1]) < 4:
            return
        self.preview_max_size = new_size
        if self.preview_source_image is not None:
            self._refresh_preview_image()
        else:
            self.puzzle_label.configure(wraplength=max(120, self.preview_max_size[0] - 40))
    def _register_hotkeys(self) -> None:
        for action, hotkey in self.hotkeys.items():
            try:
                self._set_hotkey(action, hotkey, notify=False)
            except ValueError:
                self.log_queue.put(f"[GUI] Failed to register default hotkey '{hotkey}' for {action}.\n")

    def _set_hotkey(self, action: str, hotkey: str, notify: bool = True) -> None:
        normalized = hotkey.strip().lower()
        if not normalized:
            raise ValueError("Hotkey cannot be empty.")
        if action == "kill":
            try:
                bot.set_kill_switch_key(normalized)
            except Exception as exc:
                raise ValueError(f"Could not register hotkey '{hotkey}': {exc}") from exc
            old_handle = self.hotkey_handles.pop(action, None)
            if old_handle is not None:
                try:
                    keyboard.remove_hotkey(old_handle)
                except KeyError:
                    pass
            self.hotkeys[action] = normalized
            self.hotkey_vars[action].set(normalized)
            if notify:
                self.log_queue.put(f"[GUI] Hotkey for {action} updated to '{normalized}'.\n")
            return
        try:
            handle = keyboard.add_hotkey(
                normalized,
                lambda act=action: self._hotkey_callback(act),
                suppress=False,
            )
        except Exception as exc:
            raise ValueError(f"Could not register hotkey '{hotkey}': {exc}") from exc
        old_handle = self.hotkey_handles.get(action)
        if old_handle is not None:
            keyboard.remove_hotkey(old_handle)
        self.hotkey_handles[action] = handle
        self.hotkeys[action] = normalized
        self.hotkey_vars[action].set(normalized)
        if notify:
            self.log_queue.put(f"[GUI] Hotkey for {action} updated to '{normalized}'.\n")

    def _begin_hotkey_capture(self, action: str) -> None:
        if self._pending_hotkey_action == action:
            return
        self._cancel_hotkey_capture()
        previous_value = self.hotkeys.get(action, "")
        self._capture_previous = {"hotkey": previous_value}
        if action != "kill":
            old_handle = self.hotkey_handles.pop(action, None)
            if old_handle is not None:
                try:
                    keyboard.remove_hotkey(old_handle)
                except KeyError:
                    pass
        self._pending_hotkey_action = action
        self.hotkey_vars[action].set("Press any key...")
        self.log_queue.put(f"[GUI] Waiting for new {action} hotkey...\n")
        for current_action, button in self.hotkey_buttons.items():
            if current_action == action:
                button.configure(text="Listening...", state="disabled")
            else:
                button.configure(state="disabled")
        try:
            self._capture_hook = keyboard.hook(lambda event, act=action: self._handle_hotkey_capture_event(act, event))
        except Exception as exc:
            self.log_queue.put(f"[GUI] Failed to begin hotkey capture: {exc}\n")
            messagebox.showerror("Hotkey Capture Failed", f"Could not start hotkey capture: {exc}")
            self._pending_hotkey_action = None
            self._capture_hook = None
            for button in self.hotkey_buttons.values():
                button.configure(text="Change", state="normal")
            self._restore_previous_hotkey(action, notify=False)
            self._capture_previous = None
            return
        try:
            self.root.focus_force()
        except Exception:
            pass

    def _handle_hotkey_capture_event(self, action: str, event) -> None:
        if self._pending_hotkey_action != action:
            return
        if getattr(event, "event_type", None) != "down":
            return
        name = getattr(event, "name", None)
        if not name:
            return
        if self._capture_hook is not None:
            try:
                keyboard.unhook(self._capture_hook)
            except Exception:
                pass
            self._capture_hook = None
        self.root.after(0, lambda n=name: self._complete_hotkey_capture(action, n))

    def _complete_hotkey_capture(self, action: str, key_name: Optional[str], error: Optional[str] = None) -> None:
        if self._capture_hook is not None:
            try:
                keyboard.unhook(self._capture_hook)
            except Exception:
                pass
            self._capture_hook = None
        if self._pending_hotkey_action != action:
            return
        self._pending_hotkey_action = None
        for btn_action, button in self.hotkey_buttons.items():
            button.configure(text="Change", state="normal")
        if error is not None:
            messagebox.showerror("Hotkey Capture Failed", error)
            self._restore_previous_hotkey(action, notify=False)
            self._capture_previous = None
            return
        if key_name is None:
            self._restore_previous_hotkey(action, notify=False)
            self._capture_previous = None
            return
        normalized = key_name.lower()
        if normalized == "esc":
            self.log_queue.put(f"[GUI] Hotkey capture for {action} canceled.\n")
            self._restore_previous_hotkey(action, notify=False)
            self._capture_previous = None
            return
        try:
            self._set_hotkey(action, normalized)
        except ValueError as exc:
            messagebox.showerror("Invalid Hotkey", str(exc))
            self._restore_previous_hotkey(action, notify=False)
        finally:
            self._capture_previous = None
            self.hotkey_vars[action].set(self.hotkeys[action])

    def _restore_previous_hotkey(self, action: str, notify: bool = False) -> None:
        previous_value = None
        if self._capture_previous is not None:
            previous_value = self._capture_previous.get("hotkey")
        if not previous_value:
            previous_value = self.hotkeys.get(action)
        if not previous_value:
            return
        try:
            self._set_hotkey(action, previous_value, notify=notify)
        except ValueError:
            self.hotkey_vars[action].set(previous_value)

    def _cancel_hotkey_capture(self, restore: bool = True) -> None:
        if self._pending_hotkey_action is None:
            return
        if self._capture_hook is not None:
            try:
                keyboard.unhook(self._capture_hook)
            except Exception:
                pass
            self._capture_hook = None
        action = self._pending_hotkey_action
        self._pending_hotkey_action = None
        for button in self.hotkey_buttons.values():
            button.configure(text="Change", state="normal")
        if restore:
            self._restore_previous_hotkey(action, notify=False)
        else:
            if action in self.hotkey_vars:
                self.hotkey_vars[action].set(self.hotkeys[action])
        self._capture_previous = None

    def _hotkey_callback(self, action: str) -> None:
        self.root.after(0, lambda act=action: self._trigger_hotkey(act))

    def _trigger_hotkey(self, action: str) -> None:
        if action == "single":
            self.log_queue.put(f"[Hotkey] Triggering single puzzle run ({self.hotkeys[action]}).\n")
            self.start_single()
        elif action == "continuous":
            self.log_queue.put(f"[Hotkey] Triggering continuous mode ({self.hotkeys[action]}).\n")
            self.start_continuous()
        elif action == "auto":
            self.log_queue.put(f"[Hotkey] Triggering auto-complete ({self.hotkeys[action]}).\n")
            self.start_auto()
        elif action == "stop":
            self.log_queue.put(f"[Hotkey] Triggering stop ({self.hotkeys[action]}).\n")
            self.stop()

    def _unregister_hotkeys(self) -> None:
        for handle in self.hotkey_handles.values():
            try:
                keyboard.remove_hotkey(handle)
            except KeyError:
                continue
        self.hotkey_handles.clear()

    def on_close(self) -> None:
        self._cancel_hotkey_capture()
        if self.worker_thread and self.worker_thread.is_alive():
            if messagebox.askokcancel("Quit", "The bot is still running. Stop and exit?"):
                self.pending_close = True
                self.stop_event.set()
                self._update_status_async("Stopping...", "stopping")
                self.log_queue.put("[GUI] Waiting for worker to stop before closing.\n")
                self._poll_worker_before_close()
        else:
            self._cleanup_and_destroy()

    def _poll_worker_before_close(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.root.after(200, self._poll_worker_before_close)
        else:
            self._cleanup_and_destroy()

    def _cleanup_and_destroy(self) -> None:
        self._cancel_hotkey_capture(restore=False)
        self._unregister_hotkeys()
        self.root.destroy()

    def _apply_status(self, text: str, state: str) -> None:
        style = STATUS_STYLES.get(state, STATUS_STYLES["ready"])
        self.status_state = state
        self.status_var.set(text)
        self.status_label.configure(bg=style["bg"], fg=style["fg"])

    def _update_status_async(self, text: str, state: str) -> None:
        self.root.after(0, lambda: self._apply_status(text, state))

def main() -> None:
    root = tk.Tk()
    app = ForsakenBotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

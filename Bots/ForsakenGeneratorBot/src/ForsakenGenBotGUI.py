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
    "ready": {"bg": "#d7f4d7", "fg": "#1f5131"},
    "running": {"bg": "#fff4ce", "fg": "#4b3b07"},
    "stopping": {"bg": "#fde2cf", "fg": "#6f3600"},
    "warning": {"bg": "#fce4e4", "fg": "#661d1d"},
    "error": {"bg": "#f8d7da", "fg": "#58151c"},
}

DEFAULT_HOTKEYS = {
    "single": "page down",
    "continuous": "page up",
    "stop": "end",
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

        self.preview_placeholder = "Waiting for puzzle..."
        self.current_puzzle_photo: Optional[ImageTk.PhotoImage] = None
        self.preview_status_var = tk.StringVar(value=self.preview_placeholder)
        self.preview_source_image: Optional[Image.Image] = None
        self.preview_last_message = self.preview_placeholder
        self.preview_last_size: Tuple[int, int] = (0, 0)

        self._build_ui()
        self._apply_status("Ready", "ready")
        self._set_puzzle_preview(None, self.preview_placeholder)
        self.process_log_queue()
        self._register_hotkeys()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=(12, 10, 12, 12))
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(2, weight=1)

        status_frame = ttk.Frame(container)
        status_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        status_frame.columnconfigure(1, weight=1)

        ttk.Label(status_frame, text="Status:", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, width=30, bd=1, relief="groove")
        self.status_label.grid(row=0, column=1, sticky="w", padx=(8, 0))

        buttons_frame = ttk.Frame(container)
        buttons_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        buttons_frame.columnconfigure((0, 1, 2, 3), weight=1)

        self.single_button = ttk.Button(buttons_frame, text="Run Single Puzzle", command=self.start_single)
        self.single_button.grid(row=0, column=0, padx=5, sticky="ew")

        self.continuous_button = ttk.Button(buttons_frame, text="Run Continuous", command=self.start_continuous)
        self.continuous_button.grid(row=0, column=1, padx=5, sticky="ew")

        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=5, sticky="ew")

        self.clear_log_button = ttk.Button(buttons_frame, text="Clear Log", command=self.clear_log)
        self.clear_log_button.grid(row=0, column=3, padx=5, sticky="ew")

        main_area = ttk.Frame(container)
        main_area.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        main_area.columnconfigure(0, weight=5)
        main_area.columnconfigure(1, weight=2)
        main_area.rowconfigure(0, weight=1)

        left_column = ttk.Frame(main_area)
        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_column.columnconfigure(0, weight=1)
        left_column.rowconfigure(0, weight=6)
        left_column.rowconfigure(1, weight=2)

        preview_frame = ttk.LabelFrame(left_column, text="Current Puzzle")
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=0)
        preview_frame.rowconfigure(1, weight=1)

        status_label = ttk.Label(
            preview_frame,
            textvariable=self.preview_status_var,
            anchor="w",
            padding=(4, 2),
        )
        status_label.grid(row=0, column=0, sticky="ew")

        image_container = ttk.Frame(preview_frame)
        image_container.grid(row=1, column=0, sticky="nsew", padx=6, pady=(2, 8))
        image_container.columnconfigure(0, weight=1)
        image_container.rowconfigure(0, weight=1)
        image_container.bind("<Configure>", self._on_preview_resize)

        self.puzzle_label = tk.Label(
            image_container,
            text="",
            anchor="center",
            justify="center",
            bg="#0f0f0f",
            fg="#d9d9d9",
            font=("Segoe UI", 10),
            wraplength=self.preview_max_size[0] - 40,
        )
        self.puzzle_label.grid(row=0, column=0, sticky="nsew")

        log_frame = ttk.LabelFrame(left_column, text="Debug Output")
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state="disabled", font=("Consolas", 10))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(height=8)

        settings_frame = ttk.LabelFrame(main_area, text="Settings")
        settings_frame.grid(row=0, column=1, sticky="nsew")
        settings_frame.columnconfigure(0, weight=1)
        settings_frame.columnconfigure(1, weight=0)

        ttk.Label(settings_frame, text="Speed Multiplier").grid(row=0, column=0, sticky="w", pady=(4, 0))
        speed_scale = ttk.Scale(settings_frame, from_=0.5, to=2.5, variable=self.speed_var, command=self.on_speed_change)
        speed_scale.grid(row=1, column=0, sticky="ew")
        self.speed_value = ttk.Label(settings_frame, text=f"{self.speed_var.get():.2f}x")
        self.speed_value.grid(row=1, column=1, padx=(6, 0))

        ttk.Label(settings_frame, text="Jitteriness").grid(row=2, column=0, sticky="w", pady=(12, 0))
        jitter_scale = ttk.Scale(settings_frame, from_=0.0, to=3.0, variable=self.jitter_var, command=self.on_jitter_change)
        jitter_scale.grid(row=3, column=0, sticky="ew")
        self.jitter_value = ttk.Label(settings_frame, text=f"{self.jitter_var.get():.2f}")
        self.jitter_value.grid(row=3, column=1, padx=(6, 0))

        ttk.Label(settings_frame, text="Smoothness").grid(row=4, column=0, sticky="w", pady=(12, 0))
        smooth_scale = ttk.Scale(settings_frame, from_=0.5, to=2.5, variable=self.smoothness_var, command=self.on_smoothness_change)
        smooth_scale.grid(row=5, column=0, sticky="ew")
        self.smooth_value = ttk.Label(settings_frame, text=f"{self.smoothness_var.get():.2f}")
        self.smooth_value.grid(row=5, column=1, padx=(6, 0))

        ttk.Separator(settings_frame, orient="horizontal").grid(row=6, column=0, columnspan=2, sticky="ew", pady=(18, 10))
        ttk.Label(settings_frame, text="Hotkeys").grid(row=7, column=0, columnspan=2, sticky="w")

        keybinds = [
            ("single", "Run Single Puzzle"),
            ("continuous", "Run Continuous"),
            ("stop", "Stop"),
        ]

        for offset, (action, label_text) in enumerate(keybinds):
            row_index = 8 + offset * 2
            ttk.Label(settings_frame, text=label_text).grid(row=row_index, column=0, sticky="w", pady=(6, 0))
            row_frame = ttk.Frame(settings_frame)
            row_frame.grid(row=row_index + 1, column=0, columnspan=2, sticky="ew")
            row_frame.columnconfigure(0, weight=1)
            entry = ttk.Entry(row_frame, textvariable=self.hotkey_vars[action])
            entry.grid(row=0, column=0, sticky="ew")
            ttk.Button(
                row_frame,
                text="Apply",
                command=lambda act=action: self.apply_hotkey(act),
                width=7,
            ).grid(row=0, column=1, padx=(6, 0))

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

    def _start_worker(self, mode: str) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.log_queue.put("[GUI] Bot is already running.\n")
            return
        self.stop_event.clear()
        self.running_mode = mode
        self._apply_status("Preparing...", "running")
        self.set_running_state(True)
        self.log_queue.put(f"[GUI] Starting {mode} mode.\n")
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
                    success, debug_img = bot.run_one_puzzle(
                        return_debug=True,
                        progress_callback=progress,
                    )
                    if debug_img is None and not success:
                        self._set_preview_status("No puzzle detected.")
                    if success:
                        self.log_queue.put("[GUI] Puzzle completed.\n")
                        self._update_status_async("Ready", "ready")
                    else:
                        self.log_queue.put("[GUI] No puzzle detected.\n")
                        self._update_status_async("No puzzle detected", "warning")
                else:
                    self._run_continuous_loop()
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
            self._set_preview_status("Capturing puzzle...")
            progress = lambda stage, img: self._handle_progress(stage, img)
            success, debug_img = bot.run_one_puzzle(
                return_debug=True,
                progress_callback=progress,
            )
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
            bot.scaled_sleep(0.35)
        if final_status is None:
            if self.stop_event.is_set():
                final_status = ("Stopped", "ready")
            elif puzzles_completed > 0:
                final_status = ("Ready", "ready")
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
            self.stop_button.configure(state="normal")
        else:
            self.single_button.configure(state="normal")
            self.continuous_button.configure(state="normal")
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

    def _set_preview_status(self, message: str) -> None:
        self.preview_last_message = message
        def updater(msg=message):
            self.preview_status_var.set(msg)
            if self.preview_source_image is None:
                self.puzzle_label.configure(text=msg)
        self.root.after(0, updater)

    def _update_preview_async(self, image, message: Optional[str] = None) -> None:
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
        }
        if stage == "no_puzzle":
            self._set_preview_status("No puzzle detected.")
            return
        message = stage_messages.get(stage)
        if stage == "paths_ready":
            if image is not None:
                self._update_preview_async(image, message or "Paths ready. Executing...")
            elif message:
                self._set_preview_status(message)
            return
        if message:
            self._set_preview_status(message)

    def _set_puzzle_preview(self, image, message: Optional[str] = None) -> None:
        if image is None:
            self.preview_source_image = None
            self.current_puzzle_photo = None
            display_text = message or self.preview_placeholder
            self.puzzle_label.configure(image="", text=display_text)
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
        status_text = message or "Puzzle preview ready."
        self._set_preview_status(status_text)

    def _refresh_preview_image(self) -> None:
        if self.preview_source_image is None:
            return
        target = ImageOps.contain(self.preview_source_image, self.preview_max_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(target)
        self.current_puzzle_photo = photo
        self.puzzle_label.configure(image=photo, text="", wraplength=max(120, self.preview_max_size[0] - 40))
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
    def apply_hotkey(self, action: str) -> None:
        desired = self.hotkey_vars[action].get().strip()
        if not desired:
            messagebox.showerror("Invalid Hotkey", "Hotkey cannot be empty.")
            self.hotkey_vars[action].set(self.hotkeys[action])
            return
        try:
            self._set_hotkey(action, desired)
            self.log_queue.put(f"[GUI] Hotkey for {action} set to '{self.hotkeys[action]}'.\n")
        except ValueError as exc:
            messagebox.showerror("Invalid Hotkey", str(exc))
            self.hotkey_vars[action].set(self.hotkeys[action])

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

    def _hotkey_callback(self, action: str) -> None:
        self.root.after(0, lambda act=action: self._trigger_hotkey(act))

    def _trigger_hotkey(self, action: str) -> None:
        if action == "single":
            self.log_queue.put(f"[Hotkey] Triggering single puzzle run ({self.hotkeys[action]}).\n")
            self.start_single()
        elif action == "continuous":
            self.log_queue.put(f"[Hotkey] Triggering continuous mode ({self.hotkeys[action]}).\n")
            self.start_continuous()
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

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import time
import ForsakenEntanglementBot as bot

class EntanglementBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Forsaken Entanglement Bot")
        self.root.geometry("750x550")
        self.running = tk.BooleanVar(value=False)
        self.click_count = tk.IntVar(value=0)
        self.log_queue = queue.Queue()
        self.status_var = tk.StringVar(value="Disabled")
        self.last_activity_time = 0
        style = ttk.Style()
        style.configure("Hint.TLabel", font=("Segoe UI", 8), foreground="gray")
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))
        top_frame = ttk.Frame(root, padding=10)
        top_frame.pack(fill="x")
        self.toggle_btn = ttk.Checkbutton(
            top_frame, text="Bot Enabled", variable=self.running,
            command=self.toggle_bot, style="Switch.TCheckbutton"
        )
        self.toggle_btn.pack(side="left", padx=5)
        ttk.Label(top_frame, text="Clicks:").pack(side="left", padx=(20, 5))
        self.counter_label = ttk.Label(top_frame, textvariable=self.click_count,
                                       font=("Segoe UI", 10, "bold"))
        self.counter_label.pack(side="left")
        reset_btn = ttk.Button(top_frame, text="Reset", command=self.reset_counter)
        reset_btn.pack(side="left", padx=5)
        ttk.Label(top_frame, text="Status:").pack(side="left", padx=(30, 5))
        self.status_label = ttk.Label(top_frame, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(side="left")
        settings_frame = ttk.LabelFrame(root, text="Settings", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)
        self.confidence_var = tk.DoubleVar(value=bot.CONFIDENCE)
        self.click_interval_var = tk.DoubleVar(value=bot.CLICK_INTERVAL)
        self.scale_var = tk.DoubleVar(value=bot.SCALE)
        self._add_slider(settings_frame, "Confidence", 0.5, 1.0, self.confidence_var,
                         self.update_settings, "Higher = stricter image match")
        self._add_slider(settings_frame, "Click Interval (s)", 0.01, 0.5, self.click_interval_var,
                         self.update_settings, "Delay between each click        ")
        self._add_slider(settings_frame, "Scale", 0.25, 1.0, self.scale_var,
                         self.update_settings, "Lower = faster, less precise     ")
        console_frame = ttk.LabelFrame(root, text="Debug Console", padding=5)
        console_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.console = scrolledtext.ScrolledText(console_frame, state="disabled",
                                                 wrap="word", font=("Consolas", 10))
        self.console.pack(fill="both", expand=True)
        self.update_console()
        self.update_status()
    def _add_slider(self, parent, label, from_, to, variable, command, description):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        label_frame = ttk.Frame(row)
        label_frame.pack(side="left", padx=5)
        ttk.Label(label_frame, text=label, width=20).pack(anchor="w")
        ttk.Label(label_frame, text=description, style="Hint.TLabel").pack(anchor="w")
        scale = ttk.Scale(row, from_=from_, to=to, variable=variable,
                          command=lambda _: command())
        scale.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(row, textvariable=variable, width=6).pack(side="left")

    def toggle_bot(self):
        if self.running.get():
            self.log_queue.put("[GUI] Bot enabled.\n")
            threading.Thread(target=self.run_bot, daemon=True).start()
            self.set_status("Idle", "gray")
        else:
            self.log_queue.put("[GUI] Bot disabled.\n")
            bot.stop_requested = True
            self.set_status("Disabled", "red")

    def run_bot(self):
        bot.DEBUG_MODE = True
        bot.CONFIDENCE = self.confidence_var.get()
        bot.CLICK_INTERVAL = self.click_interval_var.get()
        bot.SCALE = self.scale_var.get()
        bot.run_with_callbacks(
            on_click=self.on_bot_click,
            log_fn=self.log_queue.put,
            running_flag=lambda: self.running.get()
        )

    def on_bot_click(self, name, x, y):
        self.click_count.set(self.click_count.get() + 1)
        self.log_queue.put(f"[✓] Clicked {name} at ({x}, {y})\n")
        self.last_activity_time = time.time()
        self.set_status("Running", "green")

    def reset_counter(self):
        self.click_count.set(0)
        self.log_queue.put("[GUI] Click counter reset.\n")

    def set_status(self, text, color):
        self.status_var.set(text)
        self.status_label.configure(foreground=color)

    def update_settings(self):
        bot.CONFIDENCE = self.confidence_var.get()
        bot.CLICK_INTERVAL = self.click_interval_var.get()
        bot.SCALE = self.scale_var.get()

    def update_console(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.console.configure(state="normal")
                self.console.insert("end", message)
                self.console.see("end")
                self.console.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self.update_console)

    def update_status(self):
        if not self.running.get():
            self.set_status("Disabled", "red")
        else:
            if time.time() - self.last_activity_time > 1.0:
                self.set_status("Idle", "gray")
        self.root.after(200, self.update_status)

def main():
    root = tk.Tk()
    app = EntanglementBotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

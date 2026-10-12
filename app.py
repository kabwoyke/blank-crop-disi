"""Simple UI: add folders, run backup + auto-crop + blank page removal."""
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import processor


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Auto Crop & Blank Page Remover")
        self.geometry("720x560")
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None

        self.crop_var = tk.BooleanVar(value=True)
        self.blank_var = tk.BooleanVar(value=True)
        self.recursive_var = tk.BooleanVar(value=False)
        self.thresh_var = tk.IntVar(value=150)
        self.sens_var = tk.DoubleVar(value=0.05)

        pad = {"padx": 10, "pady": 4}
        ttk.Label(self, text="Folders to process (a 'backup' folder is created inside each):").pack(anchor="w", **pad)

        row = ttk.Frame(self)
        row.pack(fill="x", **pad)
        self.listbox = tk.Listbox(row, height=7, selectmode="extended")
        self.listbox.pack(side="left", fill="both", expand=True)
        btns = ttk.Frame(row)
        btns.pack(side="left", padx=(8, 0), anchor="n")
        ttk.Button(btns, text="Add folder…", command=self.add_folder).pack(fill="x", pady=2)
        ttk.Button(btns, text="Add path (UNC)…", command=self.add_path).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove", command=self.remove_selected).pack(fill="x", pady=2)
        ttk.Button(btns, text="Clear", command=lambda: self.listbox.delete(0, "end")).pack(fill="x", pady=2)

        opts = ttk.LabelFrame(self, text="Options")
        opts.pack(fill="x", **pad)
        ttk.Checkbutton(opts, text="Auto-crop borders", variable=self.crop_var).grid(row=0, column=0, sticky="w", padx=8, pady=2)
        ttk.Checkbutton(opts, text="Remove blank pages", variable=self.blank_var).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Checkbutton(opts, text="Include subfolders", variable=self.recursive_var).grid(row=0, column=2, sticky="w", padx=8)
        ttk.Label(opts, text="Crop threshold (0-255):").grid(row=1, column=0, sticky="w", padx=8, pady=2)
        ttk.Spinbox(opts, from_=0, to=255, textvariable=self.thresh_var, width=6).grid(row=1, column=1, sticky="w", padx=8)
        ttk.Label(opts, text="Blank sensitivity (% content):").grid(row=2, column=0, sticky="w", padx=8, pady=2)
        ttk.Spinbox(opts, from_=0.0, to=5.0, increment=0.01, textvariable=self.sens_var, width=6).grid(row=2, column=1, sticky="w", padx=8)
        ttk.Label(opts, text="Higher = more pages treated as blank", foreground="gray").grid(row=2, column=2, sticky="w", padx=8)

        ctl = ttk.Frame(self)
        ctl.pack(fill="x", **pad)
        self.start_btn = ttk.Button(ctl, text="Start", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(ctl, text="Stop", command=self.cancel.set, state="disabled")
        self.stop_btn.pack(side="left", padx=6)

        self.bar = ttk.Progressbar(self, mode="determinate")
        self.bar.pack(fill="x", **pad)
        self.status = ttk.Label(self, text="Idle")
        self.status.pack(anchor="w", padx=10)

        self.log = tk.Text(self, height=10, state="disabled")
        self.log.pack(fill="both", expand=True, **pad)
        self.after(100, self.poll)

    # -- folder list ------------------------------------------------------
    def add_folder(self):
        d = filedialog.askdirectory(title="Select image folder")
        if d and d not in self.listbox.get(0, "end"):
            self.listbox.insert("end", d)

    def add_path(self):
        d = simpledialog.askstring(
            "Add path", r"Folder path (local or UNC, e.g. \\192.168.1.10\scans):", parent=self)
        if not d:
            return
        d = processor.normalize_path(d)
        if not os.path.isdir(d):
            messagebox.showerror("Not reachable", f"Cannot access folder:\n{d}")
        elif d not in self.listbox.get(0, "end"):
            self.listbox.insert("end", d)

    def remove_selected(self):
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i)

    # -- run --------------------------------------------------------------
    def write(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        folders = list(self.listbox.get(0, "end"))
        if not folders:
            messagebox.showinfo("No folders", "Add at least one folder first.")
            return
        if not (self.crop_var.get() or self.blank_var.get()):
            messagebox.showinfo("Nothing to do", "Enable cropping and/or blank page removal.")
            return
        opts = dict(threshold=self.thresh_var.get(), sensitivity=self.sens_var.get(),
                    remove_blank=self.blank_var.get(), do_crop=self.crop_var.get(),
                    recursive=self.recursive_var.get())
        self.cancel.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.worker = threading.Thread(target=self.run_all, args=(folders, opts), daemon=True)
        self.worker.start()

    def run_all(self, folders, opts):
        for folder in folders:
            if self.cancel.is_set():
                break
            self.q.put(("log", f"== {folder}"))
            try:
                s = processor.process_folder(
                    folder, progress_cb=lambda d, t, n: self.q.put(("progress", d, t, n)),
                    cancel_check=self.cancel.is_set, **opts)
            except Exception as e:
                self.q.put(("log", f"   ERROR: {e}"))
                continue
            self.q.put(("log", f"   {s['total']} images: {s['cropped']} cropped, "
                               f"{s['unchanged']} unchanged, {s['blank_removed']} blank removed, "
                               f"{s['failed']} failed. Backup: {s['backup_dir']}. Blanks: {s['blank_backup_dir']}"))
            for err in s["errors"]:
                self.q.put(("log", f"   ! {err}"))
        self.q.put(("done", self.cancel.is_set()))

    def poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "log":
                    self.write(m[1])
                elif m[0] == "progress":
                    _, d, t, n = m
                    self.bar.configure(maximum=t, value=d)
                    self.status.configure(text=f"{d}/{t}  {n}")
                elif m[0] == "done":
                    self.status.configure(text="Stopped" if m[1] else "Finished")
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self.poll)


if __name__ == "__main__":
    App().mainloop()

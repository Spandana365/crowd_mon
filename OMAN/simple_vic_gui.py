import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class VICGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("OMAN VIC - Counting and Flow Rate")
        self.geometry("980x640")

        self.results = {}
        self.current_video = None

        self._build_ui()
        default_json = os.path.join("outputs", "json", "video_results_test.json")
        if os.path.exists(default_json):
            self._load_json(default_json)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(root)
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="Open JSON", command=self.open_json).pack(side=tk.LEFT)
        ttk.Label(controls, text="   FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="25")
        ttk.Entry(controls, width=8, textvariable=self.fps_var).pack(side=tk.LEFT)

        ttk.Label(controls, text="   Sampling Interval (frames):").pack(side=tk.LEFT)
        self.interval_var = tk.StringVar(value="15")
        ttk.Entry(controls, width=8, textvariable=self.interval_var).pack(side=tk.LEFT)

        ttk.Button(controls, text="Recompute", command=self.refresh_metrics).pack(side=tk.LEFT, padx=(10, 0))

        body = ttk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Videos").pack(anchor="w")
        self.video_list = tk.Listbox(left, width=28, height=30)
        self.video_list.pack(fill=tk.Y, expand=False)
        self.video_list.bind("<<ListboxSelect>>", self.on_video_select)

        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self.summary_var = tk.StringVar(value="Load a results JSON to begin.")
        ttk.Label(right, textvariable=self.summary_var, justify=tk.LEFT).pack(anchor="w")

        columns = ("step", "frame_idx", "inflow", "cum_count", "flow_per_min")
        self.table = ttk.Treeview(right, columns=columns, show="headings", height=23)
        for col, width in [
            ("step", 70),
            ("frame_idx", 100),
            ("inflow", 100),
            ("cum_count", 120),
            ("flow_per_min", 140),
        ]:
            self.table.heading(col, text=col)
            self.table.column(col, width=width, anchor=tk.CENTER)
        self.table.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def open_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video_results_test.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self._load_json(path)

    def _load_json(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or not data:
                raise ValueError("JSON must be a non-empty object.")
            self.results = data
            self._populate_videos()
            self.summary_var.set(f"Loaded: {path}\nVideos: {len(self.results)}")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _populate_videos(self) -> None:
        self.video_list.delete(0, tk.END)
        for name in sorted(self.results.keys()):
            self.video_list.insert(tk.END, name)
        if self.video_list.size() > 0:
            self.video_list.selection_set(0)
            self.on_video_select()

    def on_video_select(self, _evt=None) -> None:
        if not self.video_list.curselection():
            return
        idx = self.video_list.curselection()[0]
        self.current_video = self.video_list.get(idx)
        self.refresh_metrics()

    def refresh_metrics(self) -> None:
        if not self.current_video or self.current_video not in self.results:
            return

        try:
            fps = float(self.fps_var.get())
            interval = int(self.interval_var.get())
            if fps <= 0 or interval <= 0:
                raise ValueError("FPS and interval must be > 0.")
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        info = self.results[self.current_video]
        cnt_list = info.get("cnt_list", [])
        frame_num = int(info.get("frame_num", 0))
        first_frame_num = int(info.get("first_frame_num", 0))
        total_count = int(info.get("video_num", sum(cnt_list)))
        outflow_list = info.get("outflow_cnt_list", [0] * len(cnt_list))

        for row in self.table.get_children():
            self.table.delete(row)

        sampled_frames = [0] + [interval * i for i in range(1, len(cnt_list))]
        cumulative = 0
        for i, inflow in enumerate(cnt_list):
            outflow = outflow_list[i] if i < len(outflow_list) else 0
            net = int(inflow) - int(outflow)
            cumulative += net if i > 0 else int(inflow)
            elapsed_sec = sampled_frames[i] / fps if fps > 0 else 0.0
            flow_per_min = (cumulative / elapsed_sec * 60.0) if elapsed_sec > 0 else 0.0
            self.table.insert(
                "",
                tk.END,
                values=(
                    i,
                    sampled_frames[i],
                    int(inflow),
                    int(cumulative),
                    f"{flow_per_min:.2f}",
                ),
            )

        effective_secs = frame_num / fps if fps > 0 else 0.0
        overall_flow = (total_count / effective_secs * 60.0) if effective_secs > 0 else 0.0
        self.summary_var.set(
            f"Video: {self.current_video}\n"
            f"Frames: {frame_num} | First-frame count: {first_frame_num} | Final count: {total_count}\n"
            f"Overall flow rate: {overall_flow:.2f} persons/min "
            f"(fps={fps:g}, interval={interval})"
        )


if __name__ == "__main__":
    app = VICGui()
    app.mainloop()

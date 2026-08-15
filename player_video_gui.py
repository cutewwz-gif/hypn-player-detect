"""GUI for Player detection on videos using ONNX + DirectML (AMD GPU)."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from typing import Optional
from tkinter import filedialog, messagebox, ttk

from player_video_processor import default_output_path, process_video

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = APP_DIR / "runs" / "player_finetune" / "weights" / "best.onnx"


class PlayerVideoApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Player 视频检测导出")
        self.geometry("760x520")
        self.minsize(680, 460)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_path = tk.StringVar(value=str(DEFAULT_MODEL))
        self.conf_var = tk.DoubleVar(value=0.5)
        self.status_var = tk.StringVar(value="就绪")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 6}

        header = ttk.Label(
            self,
            text="Player 识别 · ONNX + DirectML (AMD 6600s)",
            font=("Segoe UI", 14, "bold"),
        )
        header.pack(anchor="w", **pad)

        model_frame = ttk.LabelFrame(self, text="模型")
        model_frame.pack(fill="x", **pad)
        ttk.Entry(model_frame, textvariable=self.model_path).pack(side="left", fill="x", expand=True, padx=8, pady=8)
        ttk.Button(model_frame, text="浏览", command=self._pick_model).pack(side="right", padx=8, pady=8)

        input_frame = ttk.LabelFrame(self, text="输入视频")
        input_frame.pack(fill="x", **pad)
        ttk.Entry(input_frame, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=8, pady=8)
        ttk.Button(input_frame, text="浏览", command=self._pick_input).pack(side="right", padx=8, pady=8)

        output_frame = ttk.LabelFrame(self, text="输出视频")
        output_frame.pack(fill="x", **pad)
        ttk.Entry(output_frame, textvariable=self.output_path).pack(side="left", fill="x", expand=True, padx=8, pady=8)
        ttk.Button(output_frame, text="浏览", command=self._pick_output).pack(side="right", padx=8, pady=8)

        settings = ttk.LabelFrame(self, text="参数")
        settings.pack(fill="x", **pad)
        ttk.Label(settings, text="置信度阈值").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        conf_scale = ttk.Scale(
            settings,
            from_=0.1,
            to=0.95,
            variable=self.conf_var,
            orient="horizontal",
            command=self._update_conf_label,
        )
        conf_scale.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        self.conf_label = ttk.Label(settings, text="0.50")
        self.conf_label.grid(row=0, column=2, padx=8, pady=8)
        settings.columnconfigure(1, weight=1)

        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", **pad)
        self.start_btn = ttk.Button(action_frame, text="开始处理", command=self._start)
        self.start_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = ttk.Button(action_frame, text="停止", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left")

        progress_frame = ttk.LabelFrame(self, text="进度")
        progress_frame.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill="x", padx=8, pady=8)
        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor="w", padx=8, pady=(0, 8))

        log_frame = ttk.LabelFrame(self, text="日志")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, height=10, wrap="word")
        self.log.pack(fill="both", expand=True, padx=8, pady=8)
        self.log.configure(state="disabled")

        hint = ttk.Label(
            self,
            text="提示：输出视频会绘制绿色 Player 框和置信度。首次运行会使用 AMD DirectML 加速。",
        )
        hint.pack(anchor="w", padx=12, pady=(0, 10))

    def _update_conf_label(self, _value: str = "") -> None:
        self.conf_label.configure(text=f"{self.conf_var.get():.2f}")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _pick_model(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 ONNX 模型",
            filetypes=[("ONNX 模型", "*.onnx"), ("所有文件", "*.*")],
        )
        if path:
            self.model_path.set(path)

    def _pick_input(self) -> None:
        path = filedialog.askopenfilename(
            title="选择输入视频",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mkv;*.mov"), ("所有文件", "*.*")],
        )
        if path:
            self.input_path.set(path)
            if not self.output_path.get().strip():
                self.output_path.set(str(default_output_path(path)))

    def _pick_output(self) -> None:
        initial = self.output_path.get().strip() or self.input_path.get().strip()
        initial_dir = str(Path(initial).parent) if initial else str(APP_DIR)
        initial_file = Path(initial).name if initial else "output_detected.mp4"
        path = filedialog.asksaveasfilename(
            title="保存输出视频",
            defaultextension=".mp4",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=[("MP4 视频", "*.mp4"), ("所有文件", "*.*")],
        )
        if path:
            self.output_path.set(path)

    def _set_running(self, running: bool) -> None:
        self.start_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")

    def _start(self) -> None:
        input_path = self.input_path.get().strip()
        output_path = self.output_path.get().strip()
        model_path = self.model_path.get().strip()

        if not input_path:
            messagebox.showwarning("缺少输入", "请先选择输入视频。")
            return
        if not output_path:
            output_path = str(default_output_path(input_path))
            self.output_path.set(output_path)
        if not model_path or not Path(model_path).exists():
            messagebox.showerror("模型不存在", f"找不到模型文件:\n{model_path}")
            return

        self._stop_event.clear()
        self.progress_var.set(0.0)
        self.status_var.set("正在启动...")
        self._set_running(True)
        self._append_log(f"输入: {input_path}")
        self._append_log(f"输出: {output_path}")
        self._append_log(f"模型: {model_path}")

        def worker() -> None:
            try:
                result = process_video(
                    input_path=input_path,
                    output_path=output_path,
                    model_path=model_path,
                    conf_threshold=float(self.conf_var.get()),
                    progress_callback=self._on_progress,
                    stop_flag=self._stop_event.is_set,
                )
                self.after(0, lambda: self._on_done(result, stopped=False))
            except Exception as exc:
                self.after(0, lambda: self._on_error(str(exc)))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _stop(self) -> None:
        self._stop_event.set()
        self.status_var.set("正在停止...")
        self._append_log("用户请求停止。")

    def _on_progress(self, percent: float, message: str) -> None:
        def update() -> None:
            self.progress_var.set(percent)
            self.status_var.set(message)

        self.after(0, update)

    def _on_done(self, result: dict, stopped: bool) -> None:
        self._set_running(False)
        self.progress_var.set(100.0 if not stopped else self.progress_var.get())
        gpu_text = "DirectML GPU" if result.get("gpu") else "CPU"
        summary = (
            f"完成: {result['processed_frames']} 帧, "
            f"检测到 {result['total_detections']} 个 Player, "
            f"平均 {result['avg_fps']:.1f} fps ({gpu_text})"
        )
        self.status_var.set(summary)
        self._append_log(summary)
        self._append_log(f"输出文件: {result['output_path']}")
        if not stopped:
            messagebox.showinfo("处理完成", f"{summary}\n\n已保存到:\n{result['output_path']}")

    def _on_error(self, message: str) -> None:
        self._set_running(False)
        self.status_var.set("处理失败")
        self._append_log(f"错误: {message}")
        messagebox.showerror("处理失败", message)


def main() -> None:
    app = PlayerVideoApp()
    app.mainloop()


if __name__ == "__main__":
    main()

"""Process video with Player ONNX detector and export annotated result."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2

from player_onnx_detector import PlayerOnnxDetector


def open_video_writer(output_path: Path, fourcc: int, fps: float, size: Tuple[int, int]) -> Tuple[cv2.VideoWriter, Path, bool]:
    """Open VideoWriter; use temp ASCII path when output contains non-ascii characters."""
    try:
        str(output_path).encode("ascii")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
        return writer, output_path, False
    except UnicodeEncodeError:
        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / f"player_detect_{os.getpid()}.mp4"
        writer = cv2.VideoWriter(str(temp_path), fourcc, fps, size)
        return writer, temp_path, True


def default_output_path(input_path: str | Path) -> Path:
    src = Path(input_path)
    return src.with_name(f"{src.stem}_player_detected{src.suffix}")


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector = PlayerOnnxDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        imgsz=320,
        use_directml=True,
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer, write_path, is_temp = open_video_writer(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    processed = 0
    total_detections = 0
    start = time.time()

    try:
        while True:
            if stop_flag and stop_flag():
                break

            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            total_detections += len(detections)
            annotated = detector.draw_detections(frame, detections)
            writer.write(annotated)
            processed += 1

            if progress_callback and (processed == 1 or processed % 5 == 0 or processed == total_frames):
                pct = (processed / total_frames * 100.0) if total_frames > 0 else 0.0
                elapsed = time.time() - start
                fps_now = processed / elapsed if elapsed > 0 else 0.0
                progress_callback(
                    pct,
                    f"帧 {processed}/{total_frames or '?'} | {fps_now:.1f} fps | 本帧 {len(detections)} 个 Player",
                )
    finally:
        cap.release()
        writer.release()
        if is_temp:
            shutil.move(str(write_path), str(output_path))

    elapsed = time.time() - start
    return {
        "processed_frames": processed,
        "total_detections": total_detections,
        "elapsed_sec": elapsed,
        "avg_fps": processed / elapsed if elapsed > 0 else 0.0,
        "output_path": str(output_path),
        "gpu": detector.using_gpu,
        "providers": detector.active_providers,
    }

import os
import cv2


def imwrite_unicode(path: str, img) -> bool:
    ext = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(encoded.tobytes())
    return True


def extract_frames(video_path: str, interval_sec: float, output_dir: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    duration = frame_count / fps if frame_count > 0 else 0

    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    t = 0.0
    while t <= duration + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"frame_{saved:05d}_{t:.1f}s.jpg")
        if not imwrite_unicode(out_path, frame):
            raise RuntimeError(f"写入失败: {out_path}")
        saved += 1
        t += interval_sec

    cap.release()
    return saved


def main():
    jobs = [
        (
            r"D:\视频\Captures\Badlion Minecraft Client v4.4.4-f8775e4-PRODUCTION4 (1.8.9) 2026-07-29 13-19-54.mp4",
            10,
            r"D:\视频\Captures\frames\2026-07-29 13-19-54",
        ),
        (
            r"D:\视频\Captures\Badlion Minecraft Client v4.4.4-f8775e4-PRODUCTION4 (1.8.9) 2026-07-29 11-54-39.mp4",
            10,
            r"D:\视频\Captures\frames\2026-07-29 11-54-39",
        ),
        (
            r"D:\视频\7月29日.mp4",
            1,
            r"D:\视频\frames\7月29日",
        ),
    ]

    for video_path, interval, output_dir in jobs:
        print(f"处理: {video_path}")
        print(f"  间隔: {interval}s -> {output_dir}")
        count = extract_frames(video_path, interval, output_dir)
        print(f"  完成: 共 {count} 帧\n")


if __name__ == "__main__":
    main()

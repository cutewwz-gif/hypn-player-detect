"""Resume: wait for remote training, export ONNX/YAML, download."""

from full_train_pipeline import (
    LOCAL_WEIGHTS_DIR,
    MAX_WAIT,
    POLL_INTERVAL,
    REMOTE_ROOT,
    REMOTE_WEIGHTS,
    PYTHON,
    connect,
    parse_completed,
    parse_error,
    run,
    training_running,
)
import re
import time


def main() -> None:
    client = connect()
    print("=== Wait for training ===")
    started = time.time()
    while time.time() - started < MAX_WAIT:
        _, log = run(client, f"tail -n 40 {REMOTE_ROOT}/train.log || true", timeout=60)
        if parse_completed(log):
            print("Training finished.")
            break
        if parse_error(log):
            raise SystemExit("Remote training failed.")
        if not training_running(client):
            _, full_log = run(client, f"tail -n 80 {REMOTE_ROOT}/train.log || true", timeout=60)
            if parse_completed(full_log):
                break
            raise SystemExit("Training stopped unexpectedly.")
        m = re.findall(r"^\s*(\d+)\s+[\d.]+G\s+([\d.]+)\s+([\d.]+)", log, re.MULTILINE)
        if m:
            ep, box, cls = m[-1]
            print(f"  epoch {ep} | box_loss {box} | cls_loss {cls}")
        time.sleep(POLL_INTERVAL)
    else:
        raise SystemExit("Training timeout.")

    print("=== Export + download ===")
    export_py = f"""from pathlib import Path
from ultralytics import YOLO
weights = Path('{REMOTE_WEIGHTS}/best.pt')
model = YOLO(str(weights))
model.export(format='onnx', imgsz=320, opset=12, simplify=True)
yaml_text = '''type: yolov8
name: player-detect-r20260801
provider: Ultralytics
display_name: Player Detect
model_path: best.onnx
input_width: 320
input_height: 320
iou_threshold: 0.45
conf_threshold: 0.45
classes:
  - Player
'''
(weights.parent / 'best.yaml').write_text(yaml_text, encoding='utf-8')
print('done', weights.parent)
"""
    sftp = client.open_sftp()
    with sftp.file(f"{REMOTE_ROOT}/export_remote.py", "w") as f:
        f.write(export_py)
    sftp.close()
    run(client, f"{PYTHON} {REMOTE_ROOT}/export_remote.py", timeout=900)

    LOCAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    for name in ("best.pt", "last.pt", "best.onnx", "best.yaml"):
        remote = f"{REMOTE_WEIGHTS}/{name}"
        local = LOCAL_WEIGHTS_DIR / name
        sftp.get(remote, str(local))
        print(f"Downloaded {name} ({local.stat().st_size / 1024 / 1024:.1f} MB)")
    sftp.close()
    client.close()
    print(f"Done -> {LOCAL_WEIGHTS_DIR}")


if __name__ == "__main__":
    main()

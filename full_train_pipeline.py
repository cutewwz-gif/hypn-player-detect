"""Prepare dataset, upload, train on remote GPU, export ONNX/YAML, download weights."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import paramiko

HOST = "region-41.seetacloud.com"
PORT = 13476
USER = "root"
PASSWORD = os.environ.get("SSH_PASSWORD")
if not PASSWORD:
    raise SystemExit("SSH_PASSWORD environment variable is required.")

LOCAL_DATASET = Path(r"C:\Users\35882\Desktop\hypn\player_dataset")
LOCAL_BASE_MODEL = Path(r"C:\Users\35882\Desktop\hypn\runs\player_finetune\weights\best.pt")
LOCAL_FALLBACK_MODEL = Path(r"d:\下载\mcbw.pt")
LOCAL_WEIGHTS_DIR = Path(r"C:\Users\35882\Desktop\hypn\runs\player_finetune\weights")

REMOTE_ROOT = "/root/player_train"
REMOTE_WEIGHTS = f"{REMOTE_ROOT}/runs/player_finetune/weights"
PYTHON = "/root/miniconda3/bin/python"

EPOCHS = 80
BATCH = 8
IMGSZ = 320
ONNX_IMGSZ = 320
POLL_INTERVAL = 90
MAX_WAIT = 7200


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> tuple[int, str]:
    print(f"$ {cmd}")
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = (stdout.read() + stderr.read()).decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out.strip():
        safe = out.rstrip()[-4000:].encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        try:
            print(safe)
        except UnicodeEncodeError:
            print(safe.encode("gbk", errors="replace").decode("gbk", errors="replace"))
    return code, out


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    transport = client.get_transport()
    if transport:
        transport.set_keepalive(15)
    return client


def upload_dir(sftp: paramiko.SFTPClient, local_dir: Path, remote_dir: str) -> None:
    for root, _, files in os.walk(local_dir):
        rel = os.path.relpath(root, local_dir).replace("\\", "/")
        remote_path = remote_dir if rel == "." else f"{remote_dir}/{rel}"
        try:
            sftp.stat(remote_path)
        except OSError:
            sftp.mkdir(remote_path)
        for name in files:
            local_file = Path(root) / name
            remote_file = f"{remote_path}/{name}"
            print(f"upload {local_file.relative_to(local_dir)}")
            sftp.put(str(local_file), remote_file)


def training_running(client: paramiko.SSHClient) -> bool:
    _, out = run(client, f"pgrep -af '{PYTHON}.*model.train|YOLO.*train' || true", timeout=30)
    return "model.train" in out or "player_finetune" in out


def parse_completed(log_text: str) -> bool:
    return "Training complete" in log_text or "Results saved to" in log_text


def parse_error(log_text: str) -> bool:
    tail = log_text[-3000:]
    return "Traceback (most recent call last)" in tail and "Training complete" not in tail


def main() -> None:
    import prepare_player_dataset

    print("=== 1/5 Prepare dataset ===")
    prepare_player_dataset.main()

    base_model = LOCAL_BASE_MODEL if LOCAL_BASE_MODEL.exists() else LOCAL_FALLBACK_MODEL
    if not base_model.exists():
        raise SystemExit(f"Base model not found: {base_model}")
    if not LOCAL_DATASET.exists():
        raise SystemExit(f"Dataset not found: {LOCAL_DATASET}")

    print(f"\n=== 2/5 Upload to {HOST}:{PORT} ===")
    client = connect()
    run(client, f"mkdir -p {REMOTE_ROOT}")
    run(client, "nvidia-smi || true")
    run(client, f"{PYTHON} -m pip install -q ultralytics onnx onnxsim || pip install -q ultralytics onnx onnxsim")

    sftp = client.open_sftp()
    upload_dir(sftp, LOCAL_DATASET, f"{REMOTE_ROOT}/player_dataset")
    sftp.put(str(base_model), f"{REMOTE_ROOT}/base.pt")
    remote_yaml = (
        f"path: {REMOTE_ROOT}/player_dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: Player\n"
    )
    with sftp.file(f"{REMOTE_ROOT}/player_dataset/data.yaml", "w") as f:
        f.write(remote_yaml)
    sftp.close()

    print("\n=== 3/5 Start training ===")
    remote_train_py = f"""from ultralytics import YOLO
model = YOLO('{REMOTE_ROOT}/base.pt')
model.train(
    data='{REMOTE_ROOT}/player_dataset/data.yaml',
    epochs={EPOCHS},
    batch={BATCH},
    imgsz={IMGSZ},
    device=0,
    project='{REMOTE_ROOT}/runs',
    name='player_finetune',
    exist_ok=True,
    pretrained=True,
    patience=20,
    save=True,
)
"""
    sftp = client.open_sftp()
    with sftp.file(f"{REMOTE_ROOT}/train_remote.py", "w") as f:
        f.write(remote_train_py)
    sftp.close()
    run(client, f"cd {REMOTE_ROOT} && nohup {PYTHON} -u train_remote.py > train.log 2>&1 &")
    time.sleep(5)

    print("\n=== 4/5 Wait for training ===")
    started = time.time()
    while time.time() - started < MAX_WAIT:
        _, log = run(client, f"tail -n 40 {REMOTE_ROOT}/train.log || true", timeout=60)
        if parse_completed(log):
            print("Training finished.")
            break
        if parse_error(log):
            raise SystemExit("Remote training failed. Check train.log")
        if not training_running(client):
            _, full_log = run(client, f"tail -n 80 {REMOTE_ROOT}/train.log || true", timeout=60)
            if parse_completed(full_log):
                break
            raise SystemExit("Training process stopped unexpectedly.")
        m = re.findall(r"^\s*(\d+)\s+[\d.]+G\s+([\d.]+)\s+([\d.]+)", log, re.MULTILINE)
        if m:
            ep, box, cls = m[-1]
            print(f"  epoch {ep} | box_loss {box} | cls_loss {cls}")
        time.sleep(POLL_INTERVAL)
    else:
        raise SystemExit("Training timeout.")

    print("\n=== 5/5 Export ONNX + YAML and download ===")
    export_py = f"""from pathlib import Path
from ultralytics import YOLO
weights = Path('{REMOTE_WEIGHTS}/best.pt')
model = YOLO(str(weights))
model.export(format='onnx', imgsz={ONNX_IMGSZ}, opset=12, simplify=True)
yaml_text = '''type: yolov8
name: player-detect-r20260801
provider: Ultralytics
display_name: Player Detect
model_path: best.onnx
input_width: {ONNX_IMGSZ}
input_height: {ONNX_IMGSZ}
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
        try:
            sftp.get(remote, str(local))
            print(f"Downloaded {name} ({local.stat().st_size / 1024 / 1024:.1f} MB)")
        except OSError as exc:
            print(f"Skip {name}: {exc}")
    sftp.close()
    client.close()

    print("\nDone.")
    print(f"Local weights: {LOCAL_WEIGHTS_DIR}")


if __name__ == "__main__":
    main()

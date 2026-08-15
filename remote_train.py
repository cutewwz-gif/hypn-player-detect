"""Upload dataset/model to remote GPU server and start YOLO training."""

import os
import sys
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
LOCAL_MODEL = Path(r"d:\下载\mcbw.pt")
REMOTE_ROOT = "/root/player_train"

EPOCHS = 80
BATCH = 8
IMGSZ = 640


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> tuple[int, str, str]:
    print(f"$ {cmd}")
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out.strip():
        print(out.rstrip())
    if err.strip():
        print(err.rstrip())
    return code, out, err


def upload_dir(sftp: paramiko.SFTPClient, local_dir: Path, remote_dir: str) -> None:
    for root, dirs, files in os.walk(local_dir):
        rel = os.path.relpath(root, local_dir)
        rel_posix = rel.replace("\\", "/")
        remote_path = remote_dir if rel == "." else f"{remote_dir}/{rel_posix}"
        try:
            sftp.stat(remote_path)
        except OSError:
            sftp.mkdir(remote_path)
        for name in files:
            local_file = Path(root) / name
            remote_file = f"{remote_path}/{name}"
            print(f"upload {local_file.name}")
            sftp.put(str(local_file), remote_file)


def main() -> None:
    if not LOCAL_DATASET.exists():
        raise SystemExit(f"Dataset not found: {LOCAL_DATASET}")
    if not LOCAL_MODEL.exists():
        raise SystemExit(f"Model not found: {LOCAL_MODEL}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST}:{PORT} ...")
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

    run(client, f"mkdir -p {REMOTE_ROOT}")
    run(client, "nvidia-smi || true")
    run(client, "python3 --version || python --version")
    run(client, "pip show ultralytics || pip install ultralytics -q")

    sftp = client.open_sftp()
    upload_dir(sftp, LOCAL_DATASET, f"{REMOTE_ROOT}/player_dataset")
    sftp.put(str(LOCAL_MODEL), f"{REMOTE_ROOT}/mcbw.pt")
    sftp.close()

    train_cmd = (
        f"cd {REMOTE_ROOT} && nohup python3 -u - <<'PY' > train.log 2>&1 &\n"
        "from ultralytics import YOLO\n"
        "model = YOLO('mcbw.pt')\n"
        "model.train(\n"
        "    data='player_dataset/data.yaml',\n"
        f"    epochs={EPOCHS},\n"
        f"    batch={BATCH},\n"
        f"    imgsz={IMGSZ},\n"
        "    device=0,\n"
        "    project='runs',\n"
        "    name='player_finetune',\n"
        "    exist_ok=True,\n"
        "    pretrained=True,\n"
        "    patience=20,\n"
        "    save=True,\n"
        ")\n"
        "PY"
    )
    run(client, train_cmd)
    time.sleep(3)
    run(client, f"tail -n 30 {REMOTE_ROOT}/train.log || true")

    print("\nTraining started in background.")
    print(f"Remote log: {REMOTE_ROOT}/train.log")
    print(f"Remote weights: {REMOTE_ROOT}/runs/player_finetune/weights/best.pt")
    client.close()


if __name__ == "__main__":
    main()

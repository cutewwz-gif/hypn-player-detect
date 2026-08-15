"""Resume upload and start YOLO training on remote GPU."""

import os
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


def run(client, cmd, timeout=600):
    print(f"$ {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out.strip():
        print(out.rstrip())
    if err.strip() and code != 0:
        print(err.rstrip())
    return code, out, err


def ensure_remote_dir(sftp, remote_dir):
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        password=PASSWORD,
        timeout=30,
        banner_timeout=60,
        auth_timeout=60,
    )
    transport = client.get_transport()
    if transport:
        transport.set_keepalive(30)
    return client


def upload_if_missing(local_path: Path, remote_path: str, retries: int = 5) -> bool:
    for attempt in range(retries):
        client = None
        try:
            client = connect()
            sftp = client.open_sftp()
            try:
                sftp.stat(remote_path)
                return False
            except OSError:
                ensure_remote_dir(sftp, os.path.dirname(remote_path))
                sftp.put(str(local_path), remote_path)
                print(f"uploaded {local_path.name}")
                return True
        except Exception as exc:
            print(f"retry {attempt + 1}/{retries} for {local_path.name}: {exc}")
            time.sleep(min(2 ** attempt, 10))
        finally:
            if client:
                client.close()
    raise RuntimeError(f"Failed to upload {local_path}")


def upload_dataset():
    uploaded = 0
    skipped = 0
    for local_file in sorted(LOCAL_DATASET.rglob("*")):
        if not local_file.is_file():
            continue
        rel = local_file.relative_to(LOCAL_DATASET).as_posix()
        remote_file = f"{REMOTE_ROOT}/player_dataset/{rel}"
        if upload_if_missing(local_file, remote_file):
            uploaded += 1
        else:
            skipped += 1
    return uploaded, skipped


def main():
    client = connect()
    run(client, f"mkdir -p {REMOTE_ROOT}")

    up, skip = upload_dataset()
    print(f"Dataset upload: new={up}, skipped={skip}")

    if upload_if_missing(LOCAL_MODEL, f"{REMOTE_ROOT}/mcbw.pt"):
        print("Model uploaded")
    else:
        print("Model already present")

    client = connect()

    # Verify counts
    run(client, f"find {REMOTE_ROOT}/player_dataset/images -type f | wc -l")
    run(client, f"find {REMOTE_ROOT}/player_dataset/labels -type f | wc -l")
    run(client, f"test -f {REMOTE_ROOT}/mcbw.pt && echo MODEL_OK")

    # Start training only if not already running
    _, proc_out, _ = run(client, "pgrep -af 'player_train|model.train|ultralytics' || true")
    if "model.train" in proc_out or "YOLO('mcbw.pt')" in proc_out:
        print("Training already running, skip start.")
    else:
        train_script = f"""cd {REMOTE_ROOT} && nohup python3 -u -c "
from ultralytics import YOLO
model = YOLO('mcbw.pt')
model.train(
    data='player_dataset/data.yaml',
    epochs={EPOCHS},
    batch={BATCH},
    imgsz={IMGSZ},
    device=0,
    project='runs',
    name='player_finetune',
    exist_ok=True,
    pretrained=True,
    patience=20,
    save=True,
)
" > train.log 2>&1 &
"""
        run(client, train_script)
        time.sleep(5)
        run(client, f"tail -n 30 {REMOTE_ROOT}/train.log || true")
        run(client, "nvidia-smi || true")

    client.close()


if __name__ == "__main__":
    main()

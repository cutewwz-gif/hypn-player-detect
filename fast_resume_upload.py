"""Fast resume upload: one remote listing, batch SFTP with reconnect on drop."""

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


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, port=PORT, username=USER, password=PASSWORD,
        timeout=30, banner_timeout=60, auth_timeout=60,
    )
    transport = client.get_transport()
    if transport:
        transport.set_keepalive(15)
    return client


def run(client, cmd, timeout=600):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    stdout.channel.recv_exit_status()
    return (out or err).strip()


def list_remote_files(client):
    out = run(client, f"find {REMOTE_ROOT} -type f 2>/dev/null")
    if not out:
        return set()
    return set(line.strip().replace("\\", "/") for line in out.splitlines() if line.strip())


def ensure_remote_dir(sftp, remote_dir):
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def upload_batch(jobs):
    uploaded = 0
    idx = 0
    while idx < len(jobs):
        client = connect()
        sftp = client.open_sftp()
        try:
            while idx < len(jobs):
                local_path, remote_path = jobs[idx]
                ensure_remote_dir(sftp, os.path.dirname(remote_path))
                sftp.put(str(local_path), remote_path)
                print(f"uploaded {local_path.name}")
                uploaded += 1
                idx += 1
        except Exception as exc:
            print(f"connection dropped at {jobs[idx][0].name}: {exc}; reconnecting...")
            time.sleep(3)
        finally:
            sftp.close()
            client.close()
    return uploaded


def main():
    client = connect()
    run(client, f"mkdir -p {REMOTE_ROOT}")
    remote_files = list_remote_files(client)
    client.close()

    jobs = []
    for local_file in sorted(LOCAL_DATASET.rglob("*")):
        if not local_file.is_file():
            continue
        rel = local_file.relative_to(LOCAL_DATASET).as_posix()
        remote = f"{REMOTE_ROOT}/player_dataset/{rel}"
        if remote not in remote_files:
            jobs.append((local_file, remote))

    model_remote = f"{REMOTE_ROOT}/mcbw.pt"
    if model_remote not in remote_files:
        jobs.append((LOCAL_MODEL, model_remote))

    print(f"Missing files to upload: {len(jobs)}")
    if jobs:
        up = upload_batch(jobs)
        print(f"Uploaded {up} files")

    client = connect()
    print(run(client, f"find {REMOTE_ROOT}/player_dataset/images -type f | wc -l"))
    print(run(client, f"find {REMOTE_ROOT}/player_dataset/labels -type f | wc -l"))
    print(run(client, f"test -f {REMOTE_ROOT}/mcbw.pt && echo MODEL_OK || echo NO_MODEL"))

    proc = run(client, "pgrep -af 'YOLO|model.train|player_finetune' || true")
    if "model.train" in proc or "YOLO('mcbw.pt')" in proc:
        print("Training already running")
    else:
        cmd = (
            f"cd {REMOTE_ROOT} && nohup python3 -u -c "
            "\"from ultralytics import YOLO; "
            "YOLO('mcbw.pt').train("
            "data='player_dataset/data.yaml',"
            f"epochs={EPOCHS},batch={BATCH},imgsz={IMGSZ},"
            "device=0,project='runs',name='player_finetune',"
            "exist_ok=True,pretrained=True,patience=20,save=True)\" "
            "> train.log 2>&1 &"
        )
        run(client, cmd)
        time.sleep(8)
        print(run(client, f"tail -n 25 {REMOTE_ROOT}/train.log"))
        print(run(client, "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"))
    client.close()


if __name__ == "__main__":
    main()

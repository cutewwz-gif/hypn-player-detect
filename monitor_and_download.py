"""Monitor remote YOLO training, download weights, report metrics."""

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
REMOTE = "/root/player_train"
PYTHON = "/root/miniconda3/bin/python"
LOCAL_WEIGHTS_DIR = Path(r"C:\Users\35882\Desktop\hypn\runs\player_finetune\weights")
REMOTE_WEIGHTS = f"{REMOTE}/runs/player_finetune/weights"

POLL_INTERVAL = 120  # seconds
MAX_WAIT = 7200  # 2 hours


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    transport = client.get_transport()
    if transport:
        transport.set_keepalive(15)
    return client


def run(client, cmd, timeout=120):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = (stdout.read() + stderr.read()).decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out


def is_training_running(client):
    code, out = run(client, f"pgrep -af '{PYTHON}.*YOLO.*mcbw' || true")
    return "YOLO" in out and "mcbw" in out


def get_log_tail(client, n=50):
    _, out = run(client, f"tail -n {n} {REMOTE}/train.log 2>/dev/null || echo NO_LOG")
    return out


def parse_metrics(log_text):
    metrics = {}
    epoch_matches = re.findall(
        r"^\s*(\d+)\s+([\d.]+)G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        log_text,
        re.MULTILINE,
    )
    if epoch_matches:
        last = epoch_matches[-1]
        metrics["last_epoch"] = int(last[0])
        metrics["box_loss"] = float(last[2])
        metrics["cls_loss"] = float(last[3])
        metrics["dfl_loss"] = float(last[4])
        metrics["precision"] = float(last[5])
        metrics["recall"] = float(last[6])
        metrics["mAP50"] = float(last[7])
        metrics["mAP50-95"] = float(last[8])

    best_map = 0.0
    best_epoch = 0
    for m in epoch_matches:
        map50 = float(m[7])
        if map50 >= best_map:
            best_map = map50
            best_epoch = int(m[0])
    if best_epoch:
        metrics["best_epoch"] = best_epoch
        metrics["best_mAP50"] = best_map

    if "Results saved to" in log_text or "Training complete" in log_text:
        metrics["completed"] = True
    if "Traceback" in log_text and "epoch" not in log_text.lower():
        metrics["failed"] = True

    return metrics


def download_weights(client):
    LOCAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    downloaded = []
    for name in ("best.pt", "last.pt"):
        remote = f"{REMOTE_WEIGHTS}/{name}"
        local = LOCAL_WEIGHTS_DIR / name
        try:
            sftp.stat(remote)
            sftp.get(remote, str(local))
            downloaded.append(str(local))
            print(f"Downloaded {name} -> {local} ({local.stat().st_size / 1024 / 1024:.1f} MB)")
        except OSError:
            print(f"Missing remote weight: {remote}")
    sftp.close()
    return downloaded


def restart_training(client):
    start_cmd = (
        f"cd {REMOTE} && nohup {PYTHON} -u -c "
        "'from ultralytics import YOLO; "
        "YOLO(\"mcbw.pt\").train("
        "data=\"player_dataset/data.yaml\","
        "epochs=80,batch=8,imgsz=640,device=0,"
        "project=\"runs\",name=\"player_finetune\","
        "exist_ok=True,pretrained=True,patience=20,save=True)' "
        "> train.log 2>&1 &"
    )
    run(client, start_cmd)
    print("Training restarted.")


def main():
    client = connect()

    print("=== Initial status ===")
    running = is_training_running(client)
    log_tail = get_log_tail(client, 30)
    print(log_tail.encode("ascii", errors="replace").decode("ascii"))
    _, gpu = run(client, "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
    print(f"GPU: {gpu.strip()}")
    print(f"Training running: {running}")

    waited = 0
    while running and waited < MAX_WAIT:
        print(f"\nWaiting {POLL_INTERVAL}s... (waited {waited}s)")
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
        client.close()
        client = connect()
        running = is_training_running(client)
        log_tail = get_log_tail(client, 15)
        last_line = log_tail.strip().splitlines()[-1] if log_tail.strip() else ""
        print(f"Still running: {running} | log tail: {last_line[:120]}")

    print("\n=== Final log tail ===")
    full_tail = get_log_tail(client, 80)
    print(full_tail.encode("ascii", errors="replace").decode("ascii"))

    metrics = parse_metrics(full_tail)
    failed = metrics.get("failed") or ("Traceback" in full_tail and not metrics.get("completed"))

    if failed and not metrics.get("last_epoch"):
        print("\nTraining appears failed. Attempting restart...")
        restart_training(client)
        client.close()
        return

    print("\n=== Downloading weights ===")
    downloaded = download_weights(client)

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if downloaded:
        print(f"\nLocal best model: {LOCAL_WEIGHTS_DIR / 'best.pt'}")
    else:
        print("\nNo weights downloaded yet.")

    client.close()


if __name__ == "__main__":
    main()

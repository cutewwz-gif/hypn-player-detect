import os
from pathlib import Path

import paramiko

HOST = "region-41.seetacloud.com"
PORT = 13476
USER = "root"
PASSWORD = os.environ.get("SSH_PASSWORD")
if not PASSWORD:
    raise SystemExit("SSH_PASSWORD environment variable is required.")
REMOTE = "/root/player_train"
LOCAL = Path(r"C:\Users\35882\Desktop\hypn\player_dataset")
LOCAL_MODEL = Path(r"d:\下载\mcbw.pt")


def run(client, cmd, timeout=60):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def main():
    local_files = [p for p in LOCAL.rglob("*") if p.is_file()]
    local_count = len(local_files) + 1  # + model
    local_bytes = sum(p.stat().st_size for p in local_files) + LOCAL_MODEL.stat().st_size
    local_images = len(list((LOCAL / "images" / "train").glob("*"))) + len(
        list((LOCAL / "images" / "val").glob("*"))
    )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

    checks = [
        ("dir", f"ls -la {REMOTE} 2>/dev/null || echo MISSING"),
        ("remote_file_count", f"find {REMOTE} -type f 2>/dev/null | wc -l"),
        ("remote_size", f"du -sb {REMOTE} 2>/dev/null || echo 0"),
        ("model", f"test -f {REMOTE}/mcbw.pt && stat -c %s {REMOTE}/mcbw.pt || echo 0"),
        ("yaml", f"test -f {REMOTE}/player_dataset/data.yaml && echo YES || echo NO"),
        ("remote_images", f"find {REMOTE}/player_dataset/images -type f 2>/dev/null | wc -l"),
        ("remote_labels", f"find {REMOTE}/player_dataset/labels -type f 2>/dev/null | wc -l"),
        ("log", f"tail -n 25 {REMOTE}/train.log 2>/dev/null || echo NO_LOG"),
        ("proc", "ps aux | grep python | grep -v grep || echo NO_PROC"),
        ("gpu", "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo NO_GPU"),
    ]

    results = {}
    for name, cmd in checks:
        _, out, err = run(client, cmd)
        results[name] = (out or err).strip()
        print(f"[{name}]")
        print(results[name])
        print()

    remote_files = int(results["remote_file_count"].split()[0]) if results["remote_file_count"].split()[0].isdigit() else 0
    remote_bytes_line = results["remote_size"].split()[0]
    remote_bytes = int(remote_bytes_line) if remote_bytes_line.isdigit() else 0
    remote_images = int(results["remote_images"].split()[0]) if results["remote_images"].split()[0].isdigit() else 0

    pct_files = min(100.0, remote_files / local_count * 100) if local_count else 0
    pct_bytes = min(100.0, remote_bytes / local_bytes * 100) if local_bytes else 0
    pct_images = min(100.0, remote_images / local_images * 100) if local_images else 0

    print("=== SUMMARY ===")
    print(f"Local expected: {local_count} files, {local_bytes/1024/1024:.1f} MB, {local_images} images")
    print(f"Remote now:     {remote_files} files, {remote_bytes/1024/1024:.1f} MB, {remote_images} images")
    print(f"Upload progress: files {pct_files:.0f}%, bytes {pct_bytes:.0f}%, images {pct_images:.0f}%")

    if "NO_PROC" in results["proc"] and "NO_LOG" in results["log"]:
        print("Training: NOT STARTED")
    elif "NO_PROC" in results["proc"] and "NO_LOG" not in results["log"]:
        print("Training: FINISHED or STOPPED (check log)")
    else:
        print("Training: RUNNING")

    client.close()


if __name__ == "__main__":
    main()

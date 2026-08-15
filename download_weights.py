import os
from pathlib import Path

import paramiko

HOST = "region-41.seetacloud.com"
PORT = 13476
USER = "root"
PASSWORD = os.environ.get("SSH_PASSWORD")
if not PASSWORD:
    raise SystemExit("SSH_PASSWORD environment variable is required.")
REMOTE_WEIGHTS = "/root/player_train/runs/player_finetune/weights"
LOCAL_DIR = Path(r"C:\Users\35882\Desktop\hypn\runs\player_finetune\weights")

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
sftp = client.open_sftp()

for name in ("best.pt", "last.pt"):
    remote = f"{REMOTE_WEIGHTS}/{name}"
    local = LOCAL_DIR / name
    sftp.get(remote, str(local))
    size_mb = local.stat().st_size / 1024 / 1024
    print(f"Downloaded {name}: {local} ({size_mb:.1f} MB)")

sftp.close()
client.close()

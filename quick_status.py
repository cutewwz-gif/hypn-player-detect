import os
import paramiko

HOST = "region-41.seetacloud.com"
PORT = 13476
USER = "root"
PASSWORD = os.environ.get("SSH_PASSWORD")
if not PASSWORD:
    raise SystemExit("SSH_PASSWORD environment variable is required.")
REMOTE = "/root/player_train"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

cmds = [
    "pgrep -af player_finetune || echo NO_TRAIN",
    f"tail -n 25 {REMOTE}/train.log",
    "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
    f"ls -lh {REMOTE}/runs/player_finetune/weights/ 2>/dev/null || echo NO_WEIGHTS",
]
for cmd in cmds:
    _, o, e = c.exec_command(cmd, timeout=60)
    out = (o.read() + e.read()).decode("utf-8", errors="replace")
    print("===", cmd)
    print(out.encode("ascii", errors="replace").decode("ascii"))
c.close()

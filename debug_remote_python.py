import os
import paramiko

HOST = "region-41.seetacloud.com"
PORT = 13476
USER = "root"
PASSWORD = os.environ.get("SSH_PASSWORD")
if not PASSWORD:
    raise SystemExit("SSH_PASSWORD environment variable is required.")
PYTHON = "/root/miniconda3/bin/python"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

cmds = [
    "pwd",
    f"{PYTHON} -c \"import sys; print('\\n'.join(sys.path))\"",
    f"{PYTHON} -m pip show ultralytics",
    "ls /root/miniconda3/lib/python3.12/site-packages/ultralytics 2>/dev/null | head -5 || echo missing",
    f"cd /root && {PYTHON} -c \"import ultralytics; print(ultralytics.__file__)\"",
    f"cd /root/player_train && {PYTHON} -c \"import ultralytics; print(ultralytics.__file__)\"",
    f"{PYTHON} -m pip install --force-reinstall ultralytics -q && {PYTHON} -c \"import ultralytics; print('ok', ultralytics.__version__)\"",
]
for cmd in cmds:
    print("===", cmd)
    _, o, e = c.exec_command(cmd, timeout=180)
    print((o.read() + e.read()).decode("utf-8", errors="replace"))
c.close()

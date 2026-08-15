import os
import time
import paramiko

HOST = "region-41.seetacloud.com"
PORT = 13476
USER = "root"
PASSWORD = os.environ.get("SSH_PASSWORD")
if not PASSWORD:
    raise SystemExit("SSH_PASSWORD environment variable is required.")
REMOTE = "/root/player_train"
PYTHON = "/root/miniconda3/bin/python"


def run(client, cmd, timeout=600):
    print(f"=== {cmd[:100]}")
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = (stdout.read() + stderr.read()).decode("utf-8", errors="replace")
    safe = out.encode("ascii", errors="replace").decode("ascii")
    print(safe.strip())
    print()
    return out


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

    run(client, f"{PYTHON} -m pip uninstall -y ultralytics || true")
    run(client, f"rm -rf /root/miniconda3/lib/python3.12/site-packages/ultralytics* /root/miniconda3/lib/python3.12/site-packages/__editable__* || true")
    run(client, f"{PYTHON} -m pip install ultralytics==8.3.63")
    run(client, f"{PYTHON} -c \"import ultralytics; print('ok', ultralytics.__version__)\"")

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
    time.sleep(20)
    run(client, f"tail -n 40 {REMOTE}/train.log")
    run(client, "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
    client.close()


if __name__ == "__main__":
    main()

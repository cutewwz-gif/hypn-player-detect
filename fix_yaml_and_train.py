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
LOCAL_YAML = r"C:\Users\35882\Desktop\hypn\player_dataset\data.yaml"


def run(client, cmd, timeout=600):
    print(f"=== {cmd[:120]}")
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = (stdout.read() + stderr.read()).decode("utf-8", errors="replace")
    print(out.encode("ascii", errors="replace").decode("ascii").strip())
    print()
    return out


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

    sftp = client.open_sftp()
    sftp.put(LOCAL_YAML, f"{REMOTE}/player_dataset/data.yaml")
    sftp.close()
    print("Uploaded fixed data.yaml")

    run(client, f"cat {REMOTE}/player_dataset/data.yaml")
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
    time.sleep(25)
    run(client, f"tail -n 30 {REMOTE}/train.log")
    run(client, "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
    client.close()


if __name__ == "__main__":
    main()

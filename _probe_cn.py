import paramiko
import time

def connect(host, user, password):
    last = None
    for i in range(5):
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            c.connect(host, username=user, password=password, timeout=30, banner_timeout=60, auth_timeout=30, allow_agent=False, look_for_keys=False)
            return c
        except Exception as e:
            last = e
            print("try", i+1, type(e).__name__, e)
            time.sleep(3)
    raise last

pw = "Wallace@Mc114"
print("trying antony.fan ...")
c = connect("antony.fan", "root", pw)
stdin, stdout, stderr = c.exec_command("hostname; ls /tmp/easy_cover_build/next.config.ts 2>/dev/null; echo ok")
print(stdout.read().decode("utf-8", "replace"))
c.close()
print("china ssh ok")

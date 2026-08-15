import paramiko

def run(host, user, password, cmd, timeout=30):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=user, password=password, timeout=20, allow_agent=False, look_for_keys=False)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    c.close()
    print(f"===== {user}@{host} exit={code} =====")
    print(out.encode("ascii", "replace").decode("ascii"))
    if err.strip():
        print("ERR:", err[-1500:].encode("ascii", "replace").decode("ascii"))


pw = "Wallace@Mc114"
run(
    "badpixel.lol",
    "ubuntu",
    pw,
    r"""
whoami; hostname; uname -a
which git node npm npx pnpm yarn python3 scp ssh curl
node -v 2>/dev/null; npm -v 2>/dev/null; pnpm -v 2>/dev/null
free -h | head -2
df -h / | tail -1
""",
)
run(
    "antony.fan",
    "root",
    pw,
    r"""
ls -la /tmp/easy_cover_build 2>/dev/null | head
test -f /tmp/easy_cover_build/next.config.ts && cat /tmp/easy_cover_build/next.config.ts
ls /tmp/easy_cover_build/node_modules 2>/dev/null | wc -l
which ssh scp rsync
sshd -T 2>/dev/null | grep -E 'permitrootlogin|passwordauthentication' | head
""",
)

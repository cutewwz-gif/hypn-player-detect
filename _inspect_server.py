import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("antony.fan", username="root", password="Wallace@Mc114", timeout=20, allow_agent=False, look_for_keys=False)
cmd = r"""
ps -ef | grep -E 'pnpm|node|git clone|deploy_easy' | grep -v grep
echo '=== dir ==='
ls -la /tmp/easy_cover_build 2>/dev/null | head
echo '=== node_modules ==='
ls /tmp/easy_cover_build/node_modules 2>/dev/null | wc -l
echo '=== swap ==='
swapon --show
free -h
"""
stdin, stdout, stderr = c.exec_command(cmd, timeout=20)
print(stdout.read().decode("utf-8", "replace"))
print(stderr.read().decode("utf-8", "replace"))
c.close()

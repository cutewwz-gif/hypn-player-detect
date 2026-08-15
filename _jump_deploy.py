#!/usr/bin/env python3
"""Jump via badpixel.lol, then build EasyCover on antony.fan with npmmirror."""
from __future__ import annotations

import sys
import time

import paramiko

PW = "Wallace@Mc114"
CN_HOSTS = ["8.138.120.241", "antony.fan"]

REMOTE_SH = r"""#!/bin/bash
set -euo pipefail
echo "=== host $(hostname) ==="
free -h | head -2
echo "=== clone ==="
if [ ! -d /tmp/easy_cover_build/.git ]; then
  rm -rf /tmp/easy_cover_build
  git clone --depth 1 https://github.com/afoim/easy_cover.git /tmp/easy_cover_build
fi
cd /tmp/easy_cover_build
echo "=== patch next.config.ts ==="
cat > next.config.ts << 'EOF'
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/easy_cover",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
EOF
printf 'registry=https://registry.npmmirror.com\n' > .npmrc
echo "=== install ==="
export CI=1
export PNPM_PROGRESS=false
pnpm install --frozen-lockfile --registry https://registry.npmmirror.com --reporter=append-only
echo "=== build ==="
export NODE_OPTIONS="--max-old-space-size=1024"
pnpm run build
echo "=== publish ==="
rm -rf /www/sites/homepage/index/easy_cover
mkdir -p /www/sites/homepage/index/easy_cover
cp -a /tmp/easy_cover_build/out/. /www/sites/homepage/index/easy_cover/
ls -la /www/sites/homepage/index/easy_cover | head
python3 /tmp/patch_easy_cover_nginx.py
cname=$(docker ps --format '{{.Names}}' | grep -i openresty | head -1)
echo "openresty=$cname"
docker exec "$cname" nginx -t
docker exec "$cname" nginx -s reload
echo "=== curl ==="
curl -sS -D - -o /tmp/ec_body.html -w 'code=%{http_code} size=%{size_download}\n' http://127.0.0.1/easy_cover/ -H 'Host: antony.fan' | head
head -c 300 /tmp/ec_body.html; echo
echo DONE
"""

NGINX_PY = r"""from pathlib import Path
p = Path("/opt/1panel/apps/openresty/openresty/conf/conf.d/homepage.conf")
text = p.read_text(encoding="utf-8")
snippet = (
    "\n"
    "    location = /easy_cover {\n"
    "        return 301 /easy_cover/;\n"
    "    }\n"
    "\n"
    "    location ^~ /easy_cover/ {\n"
    "        try_files $uri $uri/ /easy_cover/index.html;\n"
    "    }\n"
)
if "location ^~ /easy_cover/" in text:
    print("nginx location already present")
else:
    idx = text.rfind("}")
    if idx < 0:
        raise SystemExit("unexpected homepage.conf ending")
    p.write_text(text[:idx] + snippet + "\n" + text[idx:], encoding="utf-8")
    print("nginx location added")
"""


def log(msg: str) -> None:
    sys.stdout.write(msg.encode("ascii", "replace").decode("ascii") + "\n")
    sys.stdout.flush()


def connect(host: str, user: str, sock=None, retries: int = 3) -> paramiko.SSHClient:
    last = None
    for i in range(retries):
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            kw = dict(
                hostname=host,
                username=user,
                password=PW,
                timeout=25,
                banner_timeout=45,
                auth_timeout=30,
                allow_agent=False,
                look_for_keys=False,
            )
            if sock is not None:
                kw["sock"] = sock
            c.connect(**kw)
            return c
        except Exception as e:
            last = e
            log(f"connect {user}@{host} try {i+1}: {type(e).__name__} {e}")
            time.sleep(2)
    raise last


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 120) -> tuple[int, str]:
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out_chunks = []
    while True:
        line = stdout.readline()
        if not line:
            break
        out_chunks.append(line)
        sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
        sys.stdout.flush()
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if err.strip():
        sys.stdout.write("STDERR: " + err[-3000:].encode("ascii", "replace").decode("ascii") + "\n")
        sys.stdout.flush()
    return code, "".join(out_chunks) + err


def jump_to_cn(us: paramiko.SSHClient) -> paramiko.SSHClient:
    transport = us.get_transport()
    last = None
    for host in CN_HOSTS:
        try:
            log(f"opening jump channel to {host}:22")
            chan = transport.open_channel("direct-tcpip", (host, 22), ("127.0.0.1", 0))
            cn = connect(host, "root", sock=chan, retries=2)
            log(f"jumped to root@{host}")
            return cn
        except Exception as e:
            last = e
            log(f"jump {host} failed: {type(e).__name__} {e}")
    raise last


def main() -> int:
    log("=== connect badpixel.lol ===")
    us = connect("badpixel.lol", "ubuntu", retries=4)
    code, out = run(us, "hostname; whoami; echo US_OK")
    if code != 0:
        return code

    log("=== jump to antony.fan ===")
    cn = jump_to_cn(us)
    run(cn, "hostname; whoami; echo CN_OK")

    # unban likely local fail2ban if present
    run(
        cn,
        "command -v fail2ban-client >/dev/null && fail2ban-client status sshd 2>/dev/null | tail -20 || echo no_fail2ban; "
        "sshd -T 2>/dev/null | grep -E 'permitrootlogin|passwordauthentication' || true",
        timeout=30,
    )

    log("=== upload scripts ===")
    sftp = cn.open_sftp()
    with sftp.file("/tmp/deploy_easy_cover.sh", "w") as f:
        f.write(REMOTE_SH)
    with sftp.file("/tmp/patch_easy_cover_nginx.py", "w") as f:
        f.write(NGINX_PY)
    sftp.chmod("/tmp/deploy_easy_cover.sh", 0o755)
    sftp.close()

    log("=== build and publish on antony.fan ===")
    code, _ = run(cn, "bash /tmp/deploy_easy_cover.sh", timeout=600)
    log(f"deploy exit {code}")
    cn.close()
    us.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

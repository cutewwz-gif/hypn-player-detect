#!/usr/bin/env python3
"""Clone, build, and deploy EasyCover to antony.fan/easy_cover."""

from __future__ import annotations

import sys

import paramiko

HOST = "antony.fan"
USER = "root"
PASSWORD = "Wallace@Mc114"

REMOTE_SH = r"""#!/bin/bash
set -euo pipefail
echo "=== memory ==="
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
echo "=== install ==="
export CI=1
export PNPM_PROGRESS=false
printf 'registry=https://registry.npmmirror.com\n' > .npmrc
corepack enable >/dev/null 2>&1 || true
if command -v pnpm >/dev/null 2>&1; then
  pnpm install --frozen-lockfile --registry https://registry.npmmirror.com --reporter=append-only
elif command -v npm >/dev/null 2>&1; then
  npm ci
else
  echo "no pnpm/npm" >&2
  exit 1
fi
echo "=== build ==="
export NODE_OPTIONS="--max-old-space-size=1536"
if command -v pnpm >/dev/null 2>&1; then
  pnpm run build
else
  npm run build
fi
echo "=== publish ==="
rm -rf /www/sites/homepage/index/easy_cover
mkdir -p /www/sites/homepage/index/easy_cover
cp -a /tmp/easy_cover_build/out/. /www/sites/homepage/index/easy_cover/
echo "=== files ==="
ls -la /www/sites/homepage/index/easy_cover | head
python3 /tmp/patch_easy_cover_nginx.py
cname=$(docker ps --format '{{.Names}}' | grep -i openresty | head -1)
echo "openresty=$cname"
docker exec "$cname" nginx -t
docker exec "$cname" nginx -s reload
echo "=== local curl ==="
curl -sS -D - -o /tmp/ec_body.html -w 'code=%{http_code} size=%{size_download}\n' http://127.0.0.1/easy_cover/ -H 'Host: antony.fan' | head
head -c 400 /tmp/ec_body.html; echo
echo DONE
"""

REMOTE_PY = r"""from pathlib import Path

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


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> int:
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    while True:
        line = stdout.readline()
        if not line:
            break
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if err.strip():
        print("STDERR:", err[-4000:].encode("ascii", "replace").decode("ascii"), file=sys.stderr)
    return code


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        username=USER,
        password=PASSWORD,
        timeout=20,
        allow_agent=False,
        look_for_keys=False,
    )
    sftp = client.open_sftp()
    with sftp.file("/tmp/deploy_easy_cover.sh", "w") as f:
        f.write(REMOTE_SH)
    with sftp.file("/tmp/patch_easy_cover_nginx.py", "w") as f:
        f.write(REMOTE_PY)
    sftp.chmod("/tmp/deploy_easy_cover.sh", 0o755)
    sftp.close()
    code = run(client, "bash /tmp/deploy_easy_cover.sh")
    print("exit", code)
    client.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

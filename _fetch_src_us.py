#!/usr/bin/env python3
"""Clone EasyCover on US relay, prepare for local build."""
from __future__ import annotations

import sys
import time

import paramiko

PW = "Wallace@Mc114"


def connect(host: str, user: str, retries: int = 4) -> paramiko.SSHClient:
    last = None
    for i in range(retries):
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            c.connect(
                host,
                username=user,
                password=PW,
                timeout=30,
                banner_timeout=60,
                auth_timeout=30,
                allow_agent=False,
                look_for_keys=False,
            )
            return c
        except Exception as e:
            last = e
            print(f"connect {user}@{host} try {i+1} failed: {e}")
            time.sleep(2 + i * 2)
    raise last


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 180) -> tuple[int, str, str]:
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    sys.stdout.write(out.encode("ascii", "replace").decode("ascii"))
    if err.strip():
        sys.stdout.write("ERR: " + err[-2000:].encode("ascii", "replace").decode("ascii") + "\n")
    return code, out, err


def main() -> int:
    print("=== clone on badpixel.lol ===")
    us = connect("badpixel.lol", "ubuntu")
    code, _, _ = run(
        us,
        r"""
set -euo pipefail
rm -rf /tmp/easy_cover_build /tmp/easy_cover_src.tgz
git clone --depth 1 https://github.com/afoim/easy_cover.git /tmp/easy_cover_build
cat > /tmp/easy_cover_build/next.config.ts << 'EOF'
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
tar -C /tmp/easy_cover_build --exclude .git -czf /tmp/easy_cover_src.tgz .
ls -lh /tmp/easy_cover_src.tgz
""",
        timeout=180,
    )
    if code != 0:
        print("clone/tar failed", code)
        us.close()
        return code
    sftp = us.open_sftp()
    local_tar = r"C:\Users\35882\AppData\Local\Temp\easy_cover_src.tgz"
    print("downloading tar...")
    sftp.get("/tmp/easy_cover_src.tgz", local_tar)
    sftp.close()
    us.close()
    print("saved", local_tar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

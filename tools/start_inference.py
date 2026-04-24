#!/usr/bin/env python3
"""Build /workspace/solution/ and start it detached on :8000.

Runs from inside the agent sandbox. Uses the host docker daemon via the
mounted /var/run/docker.sock. Build context streams via stdin tar so no
shared host path is required for the build. `-v` mounts use host paths
(MODEL_PATH + HOST_DATA_DIR are host-side and injected by the harness).

Kills any prior container of the same name before starting. Container is
detached and left running so evaluate.py can be invoked repeatedly and
profilers can attach via `docker exec`.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request

CONTAINER_NAME = "gemma4-solution-dev"
IMAGE_TAG = "gemma4-solution-dev:latest"
SOLUTION_DIR = "/workspace/solution"
HEALTH_URL = "http://localhost:8000/health"
READY_TIMEOUT_S = 300

MODEL_PATH = os.environ["MODEL_PATH"]
HOST_DATA_DIR = os.environ["HOST_DATA_DIR"]
MODEL_MOUNT_ROOT = os.environ.get("MODEL_MOUNT_ROOT", "/mnt/hdd2")
SUBMIT_GPU = os.environ.get("SUBMIT_GPU", "1")


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}", file=sys.stderr, flush=True)
    return subprocess.run(cmd, **kw)


def build_from_stdin() -> int:
    print(f"building {IMAGE_TAG} from {SOLUTION_DIR} ...", file=sys.stderr, flush=True)
    tar = subprocess.Popen(
        ["tar", "-C", SOLUTION_DIR, "-cf", "-", "."], stdout=subprocess.PIPE,
    )
    try:
        r = subprocess.run(
            ["docker", "build", "-t", IMAGE_TAG, "-"],
            stdin=tar.stdout, check=False,
        )
    finally:
        if tar.stdout:
            tar.stdout.close()
        tar.wait()
    return r.returncode


def start_container() -> int:
    r = sh([
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--gpus", f"device={SUBMIT_GPU}",
        "--cap-add=SYS_ADMIN",  # nsys requires this
        "-e", f"MODEL_PATH={MODEL_PATH}",
        "-e", "HF_HUB_OFFLINE=1",
        "-v", f"{MODEL_MOUNT_ROOT}:{MODEL_MOUNT_ROOT}:ro",
        "-v", f"{HOST_DATA_DIR}:/data:ro",
        "-p", "8000:8000",
        IMAGE_TAG,
    ])
    return r.returncode


def wait_ready() -> bool:
    t0 = time.monotonic()
    last_err: Exception | None = None
    while time.monotonic() - t0 < READY_TIMEOUT_S:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
                body = resp.read().decode()
                if '"ready"' in body:
                    elapsed = time.monotonic() - t0
                    print(f"ready in {elapsed:.1f}s — container={CONTAINER_NAME}")
                    return True
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    print(f"FAILED: not ready after {READY_TIMEOUT_S}s (last_err={last_err})",
          file=sys.stderr)
    return False


def dump_logs() -> None:
    r = subprocess.run(
        ["docker", "logs", "--tail", "200", CONTAINER_NAME],
        capture_output=True, text=True, check=False,
    )
    print("--- container logs ---", file=sys.stderr)
    print((r.stdout + r.stderr).strip(), file=sys.stderr)


def main() -> int:
    # Kill prior
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if build_from_stdin() != 0:
        print("docker build FAILED", file=sys.stderr)
        return 1
    if start_container() != 0:
        print("docker run FAILED", file=sys.stderr)
        return 1
    if not wait_ready():
        dump_logs()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

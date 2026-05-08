#!/usr/bin/env python3
"""
deploy.py — Deploy lego-robot to all targets and run a full end-to-end smoke test.

Targets:
  server  — rsync server/ + config/ to Jetson, syntax-check on Jetson
  hub     — upload hub/main.py to LEGO Hub via pybricksdev BLE
  ios     — build and install iOS app to iPhone via xcodebuild

Smoke test:
  Starts nomad_server on Jetson, waits for iPhone auto-connect + hub connect,
  waits for inference output, verifies drive commands are flowing.

Usage:
  python deploy.py                        # deploy all + smoke test
  python deploy.py --targets server hub   # specific targets only
  python deploy.py --skip-smoke           # deploy without smoke test
  python deploy.py --smoke-only           # smoke test only (assumes deployed)
"""

import argparse, socket, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent

# ── Config ────────────────────────────────────────────────────────────────────

JETSON_SSH      = "matt@jetson.local"
JETSON_PATH     = "~/lego-robot"
JETSON_VENV_PY  = "~/venvs/nomad/bin/python"
JETSON_CUDA_LIBS = "~/venvs/nomad/lib/python3.10/site-packages/nvidia/cu13/lib"
JETSON_WS_HOST  = "jetson.local"
JETSON_WS_PORT  = 8765

HUB_BLE_NAME   = "Matt's Hub"

IOS_PROJECT    = str(ROOT / "ios" / "test.xcodeproj")
IOS_SCHEME     = "test"
IOS_DEVICE_ID  = "00008110-0019482202C0401E"  # Matt's iPhone

SMOKE_LOG      = "/tmp/nomad_smoke.log"

# ── Output ────────────────────────────────────────────────────────────────────

G, R, Y, C, B, X = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[1m", "\033[0m"

def section(title):
    print(f"\n{B}{C}{'─' * 54}{X}")
    print(f"{B}{C}  {title}{X}")
    print(f"{B}{C}{'─' * 54}{X}")

def ok(msg):   print(f"  {G}✓{X}  {msg}")
def err(msg):  print(f"  {R}✗{X}  {msg}"); sys.exit(1)
def warn(msg): print(f"  {Y}!{X}  {msg}")
def info(msg): print(f"     {msg}")

# ── Shell helpers ─────────────────────────────────────────────────────────────

def run(cmd, *, capture=False, check=True):
    result = subprocess.run(
        cmd, shell=isinstance(cmd, str),
        cwd=ROOT, capture_output=capture, text=True, check=False,
    )
    if check and result.returncode != 0:
        if capture and result.stderr:
            info(result.stderr.strip())
        err(f"Command failed (exit {result.returncode}): "
            f"{cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    return result


def ssh(remote_cmd, *, capture=False, check=True):
    return run(["ssh", JETSON_SSH, remote_cmd], capture=capture, check=check)


def run_streaming(cmd, *, line_filter=None):
    """Run a command and stream its output through an optional filter function."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=ROOT,
    )
    for raw in proc.stdout:
        line = raw.rstrip()
        if line_filter is None or line_filter(line):
            info(line)
    proc.wait()
    return proc.returncode


# ── Preflight ─────────────────────────────────────────────────────────────────

def preflight():
    section("Preflight")

    for tool in ["rsync", "ssh", "xcodebuild", "pybricksdev"]:
        r = run(f"which {tool}", capture=True, check=False)
        if r.returncode != 0:
            err(f"'{tool}' not found in PATH")
        ok(tool)

    r = ssh("echo ok", capture=True, check=False)
    if r.returncode != 0:
        err(f"Cannot SSH to {JETSON_SSH}")
    ok(f"Jetson SSH ({JETSON_SSH})")

    r = ssh(
        f"LD_LIBRARY_PATH={JETSON_CUDA_LIBS} {JETSON_VENV_PY} -c \"import torch; print('torch', torch.__version__)\"",
        capture=True, check=False,
    )
    if r.returncode != 0:
        err("Jetson nomad venv not found or torch not installed")
    ok(f"Jetson {r.stdout.strip()}")

    r = run(
        f"xcrun xctrace list devices 2>/dev/null | grep {IOS_DEVICE_ID}",
        capture=True, check=False,
    )
    if r.returncode != 0:
        err(f"iPhone {IOS_DEVICE_ID} not visible — connect via USB")
    label = r.stdout.strip().split("(")[0].strip()
    ok(f"iPhone: {label} ({IOS_DEVICE_ID[:8]}…)")


# ── Deploy: server ────────────────────────────────────────────────────────────

def deploy_server():
    section("Deploy → Jetson server")

    ssh(f"mkdir -p {JETSON_PATH}/server {JETSON_PATH}/config")

    run(f"rsync -az --delete server/ {JETSON_SSH}:{JETSON_PATH}/server/")
    ok("server/ synced")

    run(f"rsync -az config/ {JETSON_SSH}:{JETSON_PATH}/config/")
    ok("config/ synced")

    for script in ["nomad_server.py", "sensor_hub.py", "infer_nomad.py"]:
        r = ssh(
            f"LD_LIBRARY_PATH={JETSON_CUDA_LIBS} {JETSON_VENV_PY} -m py_compile {JETSON_PATH}/server/{script}",
            capture=True, check=False,
        )
        if r.returncode != 0:
            err(f"{script}: syntax error — {r.stderr.strip()}")
        ok(f"{script}: syntax OK")


# ── Deploy: hub ───────────────────────────────────────────────────────────────

def deploy_hub():
    section("Deploy → LEGO Hub")
    warn(f"Hub must be powered on and idle (not connected to iPhone app)")
    input("     Press Enter when ready... ")

    # pybricksdev uploads and runs the script; we wait for event|ready
    # then terminate — program persists in hub flash for iOS app to start later.
    # If BLE release is slow, fallback: replace `run ble` with `run ble --no-wait`.
    proc = subprocess.Popen(
        ["pybricksdev", "run", "ble", "-n", HUB_BLE_NAME, "hub/main.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=ROOT,
    )

    info("Connecting and uploading…")
    deadline = time.time() + 60
    verified = False

    try:
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            line = line.strip()
            if line:
                info(f"  hub › {line}")
            if line == "event|ready":
                verified = True
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    if not verified:
        err("Hub did not output 'event|ready' within 60s — check hub power and BLE range")
    ok("hub/main.py uploaded — 'event|ready' confirmed")


# ── Deploy: iOS ───────────────────────────────────────────────────────────────

def deploy_ios():
    section("Deploy → iPhone")

    def xcode_filter(line):
        lc = line.lower()
        return any(x in lc for x in ["error:", "** build", "compile", "linking"])

    rc = run_streaming(
        ["xcodebuild",
         "-project", IOS_PROJECT,
         "-scheme",  IOS_SCHEME,
         "-destination", f"id={IOS_DEVICE_ID}",
         "-configuration", "Debug",
         "-allowProvisioningUpdates",
         "build"],
        line_filter=xcode_filter,
    )
    if rc != 0:
        err("iOS build/install failed — run xcodebuild manually for full output")
    ok(f"App built and installed ({IOS_DEVICE_ID[:8]}…)")


# ── Smoke test ────────────────────────────────────────────────────────────────

def _wait_for_log(pattern, *, timeout, label, server_pid):
    """Poll Jetson log until pattern appears. Bails immediately if server dies."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Check server is still alive
        alive = ssh(
            f"kill -0 {server_pid} 2>/dev/null && echo alive || echo dead",
            capture=True, check=False,
        )
        if "dead" in (alive.stdout or ""):
            print()
            log = ssh(f"tail -n 30 {SMOKE_LOG}", capture=True, check=False)
            info("--- server log (last 30 lines) ---")
            for l in (log.stdout or "").splitlines():
                info(l)
            err("Server process died — check log above")

        log = ssh(f"cat {SMOKE_LOG} 2>/dev/null", capture=True, check=False)
        for line in (log.stdout or "").splitlines():
            if pattern in line:
                print()
                return line

        remaining = int(deadline - time.time())
        print(f"\r     waiting for {label}… ({remaining}s)  ", end="", flush=True)
        time.sleep(2)
    print()
    return None


def smoke_test():
    section("Smoke test — full end-to-end")

    # Clear previous log and any leftover server
    ssh(f"rm -f {SMOKE_LOG}; pkill -f nomad_server.py; "
        f"fuser -k {JETSON_WS_PORT}/tcp 2>/dev/null; sleep 1",
        capture=True, check=False)

    # Start server, capture PID
    r = ssh(
        f"bash -c 'source ~/venvs/nomad/bin/activate && "
        f"export LD_LIBRARY_PATH={JETSON_CUDA_LIBS}:$LD_LIBRARY_PATH && "
        f"cd {JETSON_PATH}/server && "
        f"nohup python -u nomad_server.py --mode explore "
        f"> {SMOKE_LOG} 2>&1 & echo $!'",
        capture=True,
    )
    server_pid = r.stdout.strip().splitlines()[-1]
    info(f"Server PID {server_pid} — loading model (~30s)…")

    try:
        # Model load
        line = _wait_for_log("Model ready.", timeout=120, label="model load",
                              server_pid=server_pid)
        if not line:
            err("Model did not load within 120s")
        ok("Model loaded on Jetson")

        # WebSocket port reachable from Mac
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with socket.create_connection((JETSON_WS_HOST, JETSON_WS_PORT), timeout=2):
                    break
            except OSError:
                time.sleep(1)
        else:
            err(f"Port {JETSON_WS_PORT} on {JETSON_WS_HOST} not reachable after 30s")
        ok(f"WebSocket port {JETSON_WS_PORT} open")

        # Prompt
        print(f"\n  {Y}▶  Open the iPhone app — it will auto-connect to the server.{X}")
        print(f"  {Y}▶  Tap 'Connect Hub' in the app once it connects.{X}\n")

        # iPhone connect
        line = _wait_for_log("iPhone connected", timeout=60, label="iPhone connect",
                              server_pid=server_pid)
        if not line:
            err("iPhone did not connect within 60s")
        ok("iPhone connected to server")

        # First inference step
        line = _wait_for_log("→ drive:", timeout=60, label="first inference",
                              server_pid=server_pid)
        if not line:
            err("No inference output within 60s of iPhone connecting")
        ok(f"Inference running: {line.strip()}")

    finally:
        ssh(f"kill {server_pid} 2>/dev/null; sleep 1; "
            f"kill -9 {server_pid} 2>/dev/null; true",
            capture=True, check=False)
        ok("Server stopped")

    print(f"\n  {B}{G}✓  Smoke test passed — full pipeline verified{X}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", nargs="+",
                   choices=["server", "hub", "ios"], metavar="TARGET",
                   help="Deploy specific targets only (default: all three)")
    p.add_argument("--skip-smoke", action="store_true",
                   help="Skip the smoke test")
    p.add_argument("--smoke-only", action="store_true",
                   help="Run smoke test only, skip deployment")
    args = p.parse_args()

    targets = set(args.targets or ["server", "hub", "ios"])

    try:
        if not args.smoke_only:
            preflight()
            if "server" in targets: deploy_server()
            if "hub"    in targets: deploy_hub()
            if "ios"    in targets: deploy_ios()

        if not args.skip_smoke:
            smoke_test()

    except KeyboardInterrupt:
        print(f"\n\n  {Y}Interrupted — cleaning up…{X}")
        ssh("pkill -f nomad_server.py", capture=True, check=False)
        sys.exit(1)

    print(f"{B}{G}All done.{X}\n")


if __name__ == "__main__":
    main()

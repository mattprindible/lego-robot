"""
Deploy hub/main.py without touching the iPhone connection.

Usage:
    from deploy import deploy
    deploy(hub)          # hub already connected to iPhone

    # or standalone (starts server, deploys, leaves hub ready):
    python deploy.py
"""

import subprocess
import time


HUB_NAME  = "Matt's Hub"
HUB_FILE  = "hub/main.py"
BLE_SETTLE_S = 3   # seconds to wait after pybricksdev releases BLE


def deploy(hub, hub_file: str = HUB_FILE) -> bool:
    """
    Disconnect hub BLE via iPhone, upload new code, reconnect.
    Returns True if hub reaches Ready state after deploy.
    """
    print("── Deploy ──────────────────────────────")

    # 1. Disconnect hub from iPhone
    print("  Disconnecting hub from iPhone...")
    hub.hub_disconnect()
    ok = hub.wait_hub_state("Disconnected", timeout=10)
    if not ok:
        print(f"  WARNING: hub did not reach Disconnected (state={hub.hub_state}), continuing anyway")

    # 2. Upload code (--no-start: upload only, don't run)
    print(f"  Uploading {hub_file}...")
    result = subprocess.run(
        ["pybricksdev", "run", "ble", "--no-start", "--name", HUB_NAME, hub_file],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"  ERROR: pybricksdev failed:\n{result.stderr[-300:]}")
        return False
    print("  Upload complete.")

    # 3. Wait for BLE to be available again
    print(f"  Waiting {BLE_SETTLE_S}s for BLE to settle...")
    time.sleep(BLE_SETTLE_S)

    # 4. Reconnect iPhone to hub (iOS will auto-start the program)
    print("  Reconnecting...")
    hub.hub_connect()

    deadline = time.time() + 30
    last = ""
    while time.time() < deadline:
        s = hub.hub_state
        if s != last:
            print(f"  hub_state → {s}")
            last = s
            if s == "Ready":
                break
        time.sleep(0.1)

    if hub.hub_state == "Ready":
        print("── Ready ───────────────────────────────")
        return True
    else:
        print(f"  ERROR: hub did not reach Ready (state={hub.hub_state})")
        return False


if __name__ == "__main__":
    from server import start_background

    hub = start_background()
    print("Waiting for iPhone...")
    deadline = time.time() + 20
    while not hub.connected and time.time() < deadline:
        time.sleep(0.2)

    if not hub.connected:
        print("iPhone not connected.")
    else:
        deploy(hub)

from pybricks.hubs import InventorHub
from pybricks.pupdevices import Motor, UltrasonicSensor
from pybricks.parameters import Port, Direction
from pybricks.robotics import DriveBase
from pybricks.tools import multitask, run_task, wait
from uselect import poll
from usys import stdin

hub = InventorHub()
left = Motor(Port.A, Direction.COUNTERCLOCKWISE)
right = Motor(Port.B)
drive = DriveBase(left, right, 56, 114)
ultrasonic = UltrasonicSensor(Port.D)

running = True
safe = True
last_obstacle = 2000
_obs_history = [2000, 2000, 2000]   # rolling window for ultrasonic smoothing

# Motion state — set by stdin_reader, consumed by drive_updater
commanded_speed     = 0
commanded_turn_rate = 0
effective_speed     = 0
drive_active        = False

stdin_poll = poll()
stdin_poll.register(stdin)

# ── Thresholds ────────────────────────────────────────────────────────────────

SAFETY_STOP_MM      = 150
SAFETY_CLEAR_MM     = 200
SLOW_START_MM       = 500
OBSTACLE_WARN_MM    = 300   # soft warning: something ahead, Python should know

TELEM_INTERVAL_MS   = 200
DRIVE_UPDATE_MS     = 100

STALL_SPEED_THRESHOLD = 20   # mm/s
STALL_CONFIRM_MS      = 400

TILT_WARN_DEG       = 15    # pitch or roll beyond this → surface_change warning
TILT_STOP_DEG       = 35    # pitch or roll beyond this → immediate stop (tipping)
IMPACT_ACCEL_MG     = 3000  # mg — spike above this (beyond gravity) → impact
HEADING_DRIFT_DEG   = 10    # heading error persists beyond this → heading_drift warning
HEADING_DRIFT_MS    = 1000  # must persist this long to fire (avoids transients)

BATTERY_WARN_MV     = 7200  # mV — below this, fire battery_low once

HDG_KP = 1.5


# ── Speed scale ───────────────────────────────────────────────────────────────

def _speed_scale():
    if last_obstacle >= SLOW_START_MM:
        return 1.0
    if last_obstacle <= SAFETY_STOP_MM:
        return 0.0
    return (last_obstacle - SAFETY_STOP_MM) / (SLOW_START_MM - SAFETY_STOP_MM)


# ── stdin ─────────────────────────────────────────────────────────────────────

async def stdin_reader():
    global running, commanded_speed, commanded_turn_rate, effective_speed, drive_active
    buf = b""
    while running:
        if stdin_poll.poll(0):
            byte = stdin.buffer.read(1)
            if byte == b"\n":
                line = str(buf, "utf-8").strip()
                buf = b""
                if not line:
                    pass
                elif line == "shutdown":
                    drive.stop()
                    commanded_speed = 0
                    commanded_turn_rate = 0
                    drive_active = False
                    running = False
                    print("event|done|shutdown")
                elif line == "stop":
                    drive.stop()
                    commanded_speed = 0
                    commanded_turn_rate = 0
                    drive_active = False
                    print("event|done|stop")
                elif line == "brake":
                    drive.brake()
                    commanded_speed = 0
                    commanded_turn_rate = 0
                    drive_active = False
                    print("event|done|brake")
                elif line.startswith("drive:"):
                    parts = line.split(":")
                    if len(parts) == 3:
                        try:
                            commanded_speed = int(parts[1])
                            commanded_turn_rate = int(parts[2])
                            drive_active = True
                            print("event|done|" + line)
                        except ValueError:
                            print("event|error|bad drive params")
                    else:
                        print("event|error|bad drive params")
                elif line.startswith("turn:"):
                    parts = line.split(":")
                    if len(parts) == 2:
                        try:
                            commanded_speed = 0
                            commanded_turn_rate = 0
                            drive_active = False
                            await _gyro_turn(int(parts[1]))
                            print("event|done|" + line)
                        except Exception as e:
                            print("event|error|" + str(e))
                    else:
                        print("event|error|bad turn params")
                else:
                    print("event|error|unknown:" + line)
            else:
                buf += byte
        else:
            await wait(10)


# ── Gyro turn ─────────────────────────────────────────────────────────────────

async def _gyro_turn(target_deg: int):
    start = hub.imu.heading()
    await drive.turn(int(target_deg * 1.2))
    await wait(150)
    for _ in range(3):
        actual = hub.imu.heading() - start
        error = target_deg - actual
        if abs(error) < 2.0:
            break
        await drive.turn(int(error))
        await wait(100)


# ── Drive updater (fast inner loop) ──────────────────────────────────────────

async def drive_updater():
    global effective_speed
    hdg_target = hub.imu.heading()

    while running:
        await wait(DRIVE_UPDATE_MS)

        if not drive_active or not safe:
            hdg_target = hub.imu.heading()
            continue

        scale = _speed_scale()
        scaled_speed = int(commanded_speed * scale)
        effective_speed = scaled_speed

        if scale == 0:
            drive.stop()
            continue

        if abs(commanded_turn_rate) < 5 and abs(commanded_speed) > 0:
            hdg_err = hub.imu.heading() - hdg_target
            if hdg_err > 180:
                hdg_err -= 360
            elif hdg_err < -180:
                hdg_err += 360
            correction = max(-25, min(25, int(-HDG_KP * hdg_err)))
        else:
            hdg_target = hub.imu.heading()
            correction = 0

        drive.drive(scaled_speed, commanded_turn_rate + correction)


# ── Safety monitor ────────────────────────────────────────────────────────────

async def safety_monitor():
    global safe, last_obstacle, commanded_speed, commanded_turn_rate
    global effective_speed, drive_active, _obs_history
    obstacle_warned = False

    while running:
        try:
            raw = await ultrasonic.distance()
            _obs_history.append(raw)
            _obs_history.pop(0)
            last_obstacle = min(_obs_history)

            # Hard stop
            if safe and last_obstacle < SAFETY_STOP_MM:
                drive.stop()
                safe = False
                commanded_speed = 0
                commanded_turn_rate = 0
                effective_speed = 0
                drive_active = False
                obstacle_warned = False
                print("event|safety_stop|" + str(last_obstacle))

            # Soft warning — once on approach, reset when clear
            elif safe and last_obstacle < OBSTACLE_WARN_MM and not obstacle_warned:
                obstacle_warned = True
                print("event|obstacle_near|" + str(last_obstacle))

            elif last_obstacle >= OBSTACLE_WARN_MM:
                obstacle_warned = False

            # Clear
            if not safe and last_obstacle >= SAFETY_CLEAR_MM:
                safe = True
                print("event|safety_clear")

        except Exception as e:
            drive.stop()
            if safe:
                safe = False
                print("event|safety_stop|sensor_error:" + str(e))
        await wait(50)


# ── Stall monitor ─────────────────────────────────────────────────────────────

async def stall_monitor():
    stall_since = 0
    while running:
        await wait(100)
        if abs(effective_speed) < STALL_SPEED_THRESHOLD:
            stall_since = 0
            continue
        _, actual_spd, _, actual_tr = drive.state()
        actually_moving = (
            abs(actual_spd) > STALL_SPEED_THRESHOLD or
            abs(actual_tr)  > 5
        )
        if actually_moving:
            stall_since = 0
        else:
            stall_since += 100
            if stall_since >= STALL_CONFIRM_MS:
                drive.stop()
                print("event|stall|spd:" + str(actual_spd) + "|tr:" + str(actual_tr))
                stall_since = 0


# ── IMU monitor ───────────────────────────────────────────────────────────────

async def imu_monitor():
    """
    Watches pitch/roll for surface changes and tipping,
    acceleration for impacts, and heading for persistent drift.
    All events are facts — no interpretation.
    """
    hdg_drift_since  = 0
    battery_warned   = False
    _was_tilted      = False
    _impact_cooldown = 0

    while running:
        await wait(100)

        pitch, roll = hub.imu.tilt()
        ax, ay, az  = hub.imu.acceleration()

        # ── Tilt: immediate stop if tipping ──────────────────────────────────
        if abs(pitch) > TILT_STOP_DEG or abs(roll) > TILT_STOP_DEG:
            if drive_active:
                drive.stop()
            print("event|tipping|p:" + str(int(pitch)) + "|r:" + str(int(roll)))
            await wait(500)
            continue

        # ── Tilt: soft warning for surface change — fires once on entry ──────
        tilted = abs(pitch) > TILT_WARN_DEG or abs(roll) > TILT_WARN_DEG
        if tilted and not _was_tilted:
            print("event|surface_change|p:" + str(int(pitch)) + "|r:" + str(int(roll)))
        _was_tilted = tilted

        # ── Impact: lateral spike while flat, stopped, and settled ───────────
        # Gated on: upright, not driving, and a short cooldown to avoid
        # safety_stop deceleration triggering a false positive.
        lateral = (ax * ax + ay * ay) ** 0.5
        if (lateral > IMPACT_ACCEL_MG
                and not drive_active
                and abs(pitch) < TILT_WARN_DEG and abs(roll) < TILT_WARN_DEG
                and _impact_cooldown == 0):
            print("event|impact|lat:" + str(int(lateral)))
            _impact_cooldown = 5   # 500ms cooldown (5 × 100ms ticks)
        if _impact_cooldown > 0:
            _impact_cooldown -= 1

        # ── Heading drift: IMU heading diverging while driving straight ───────
        if drive_active and abs(commanded_turn_rate) < 5 and abs(commanded_speed) > 0:
            _, _, _, tr = drive.state()
            hdg_err = abs(tr)   # turn rate should be near 0 on a straight
            if hdg_err > HEADING_DRIFT_DEG:
                hdg_drift_since += 100
                if hdg_drift_since >= HEADING_DRIFT_MS:
                    print("event|heading_drift|tr:" + str(int(tr)))
                    hdg_drift_since = 0
            else:
                hdg_drift_since = 0

        # ── Battery low ───────────────────────────────────────────────────────
        if not battery_warned:
            bat = hub.battery.voltage()
            if bat < BATTERY_WARN_MV:
                battery_warned = True
                print("event|battery_low|" + str(bat))


# ── Telemetry ─────────────────────────────────────────────────────────────────

async def telemetry():
    while running:
        ds, spd, ang, tr = drive.state()
        hdg = hub.imu.heading()
        pitch, roll = hub.imu.tilt()
        ax, ay, az = hub.imu.acceleration()
        bat = hub.battery.voltage()
        print("|".join([
            "telem",
            "ds:" + str(ds),
            "spd:" + str(spd),
            "ang:" + str(ang),
            "tr:" + str(tr),
            "hdg:" + str(hdg),
            "obs:" + str(last_obstacle),
            "spd_scale:" + str(int(_speed_scale() * 100)),
            "safe:" + ("1" if safe else "0"),
            "p:" + str(pitch),
            "r:" + str(roll),
            "ax:" + str(ax),
            "ay:" + str(ay),
            "az:" + str(az),
            "bat:" + str(bat),
        ]))
        await wait(TELEM_INTERVAL_MS)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    try:
        print("event|ready")
        await multitask(
            stdin_reader(),
            drive_updater(),
            safety_monitor(),
            stall_monitor(),
            imu_monitor(),
            telemetry(),
        )
    except BaseException as e:
        from pybricks.parameters import Icon
        hub.display.icon(Icon.SAD)
        print("event|crash|" + str(e))


run_task(main())

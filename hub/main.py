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

commanded_speed     = 0
commanded_turn_rate = 0
drive_active        = False

stdin_poll = poll()
stdin_poll.register(stdin)

SAFETY_STOP_MM   = 150
SAFETY_CLEAR_MM  = 200
SLOW_START_MM    = 500
OBSTACLE_WARN_MM = 300

TELEM_INTERVAL_MS = 200
DRIVE_UPDATE_MS   = 100


def _speed_scale():
    if last_obstacle >= SLOW_START_MM:
        return 1.0
    if last_obstacle <= SAFETY_STOP_MM:
        return 0.0
    return (last_obstacle - SAFETY_STOP_MM) / (SLOW_START_MM - SAFETY_STOP_MM)


async def stdin_reader():
    global running, commanded_speed, commanded_turn_rate, drive_active
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
                else:
                    print("event|error|unknown:" + line)
            else:
                buf += byte
        else:
            await wait(10)


async def drive_updater():
    while running:
        await wait(DRIVE_UPDATE_MS)
        if not drive_active or not safe:
            continue
        scale = _speed_scale()
        if scale == 0:
            drive.stop()
            continue
        drive.drive(int(commanded_speed * scale), commanded_turn_rate)


async def safety_monitor():
    global safe, last_obstacle, commanded_speed, commanded_turn_rate, drive_active
    obstacle_warned = False
    while running:
        try:
            last_obstacle = await ultrasonic.distance()
            if safe and last_obstacle < SAFETY_STOP_MM:
                drive.stop()
                safe = False
                commanded_speed = 0
                commanded_turn_rate = 0
                drive_active = False
                obstacle_warned = False
                print("event|safety_stop|" + str(last_obstacle))
            elif safe and last_obstacle < OBSTACLE_WARN_MM and not obstacle_warned:
                obstacle_warned = True
                print("event|obstacle_near|" + str(last_obstacle))
            elif last_obstacle >= OBSTACLE_WARN_MM:
                obstacle_warned = False
            if not safe and last_obstacle >= SAFETY_CLEAR_MM:
                safe = True
                print("event|safety_clear")
        except Exception as e:
            drive.stop()
            if safe:
                safe = False
                print("event|safety_stop|sensor_error:" + str(e))
        await wait(50)


async def telemetry():
    while running:
        ds, spd, _, _ = drive.state()
        hdg = hub.imu.heading()
        bat = hub.battery.voltage()
        print("|".join([
            "telem",
            "ds:"  + str(ds),
            "spd:" + str(spd),
            "hdg:" + str(hdg),
            "obs:" + str(last_obstacle),
            "safe:" + ("1" if safe else "0"),
            "bat:" + str(bat),
        ]))
        await wait(TELEM_INTERVAL_MS)


async def main():
    try:
        print("event|ready")
        await multitask(
            stdin_reader(),
            drive_updater(),
            safety_monitor(),
            telemetry(),
        )
    except BaseException as e:
        from pybricks.parameters import Icon
        hub.display.icon(Icon.SAD)
        print("event|crash|" + str(e))


run_task(main())

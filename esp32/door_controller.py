
"""MicroPython ESP32 controller: authenticated, idempotent HTTP relay release."""

import gc
import uasyncio as asyncio
import ujson as json
import network
import time
from machine import Pin

from config import (
    WIFI_SSID,
    WIFI_PASSWORD,
    DOOR_RELAY_GPIO,
    DOOR_RELAY_ACTIVE_LOW,
    DOOR_UNLOCK_DURATION,
    DOOR_MAX_UNLOCK_DURATION,
    DOOR_API_TOKEN,
    DOOR_RELAY_ENABLED,
    DOOR_SIMULATION,
    HTTP_PORT,
)


# --------------------------------------------------
# Configuration validation
# --------------------------------------------------

if not WIFI_SSID or not DOOR_API_TOKEN:
    raise RuntimeError(
        "Configure WIFI_SSID and DOOR_API_TOKEN before starting"
    )

if not 1 <= DOOR_UNLOCK_DURATION <= DOOR_MAX_UNLOCK_DURATION:
    raise RuntimeError(
        "DOOR_UNLOCK_DURATION must be between 1 and DOOR_MAX_UNLOCK_DURATION"
    )

if not 1 <= HTTP_PORT <= 65535:
    raise RuntimeError("Invalid HTTP_PORT")


# --------------------------------------------------
# Relay initialization
# --------------------------------------------------

SECURE_VALUE = 1 if DOOR_RELAY_ACTIVE_LOW else 0
UNLOCK_VALUE = 0 if DOOR_RELAY_ACTIVE_LOW else 1

# Initialize relay in secure state before Wi-Fi startup.
relay = Pin(DOOR_RELAY_GPIO, Pin.OUT, value=SECURE_VALUE)

unlock_task = None
active_command = None

# Bounded in-memory replay protection.
recent_commands = []


# --------------------------------------------------
# Logging
# --------------------------------------------------

def log(message):
    print("[DOOR] " + message)


# --------------------------------------------------
# Authentication
# --------------------------------------------------

def constant_time_equal(left, right):
    if not isinstance(left, str):
        return False

    if len(left) != len(right):
        return False

    mismatch = 0

    for a, b in zip(left.encode(), right.encode()):
        mismatch |= a ^ b

    return mismatch == 0


# --------------------------------------------------
# Command validation
# --------------------------------------------------

def valid_command_id(command_id):

    if not isinstance(command_id, str):
        return False

    if not 1 <= len(command_id) <= 128:
        return False

    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789._-"
    )

    for char in command_id:
        if char not in allowed:
            return False

    return True


def valid_expiry(value):
    """Fail closed when the board has no synchronized epoch clock."""
    if not isinstance(value, int):
        return False
    now = time.time()
    return now >= 1700000000 and now <= value and value - now <= 60


# --------------------------------------------------
# Relay control
# --------------------------------------------------

def set_secure():
    relay.value(SECURE_VALUE)
    log("Relay restored to secure state")


async def restore_after_delay(command_id):
    global unlock_task, active_command

    try:
        await asyncio.sleep(DOOR_UNLOCK_DURATION)

    finally:
        set_secure()
        active_command = None
        unlock_task = None

        log(
            "Unlock duration completed | commandId="
            + command_id
        )


def remember(command_id):
    recent_commands.append(command_id)

    if len(recent_commands) > 64:
        recent_commands.pop(0)


async def unlock(command_id):
    global unlock_task, active_command

    # Reject replayed commands.
    if command_id in recent_commands:
        return "duplicate"

    # Remember the command before processing.
    remember(command_id)

    # Do not extend an existing unlock window.
    if unlock_task is not None:
        return "already_unlocked"

    active_command = command_id

    if DOOR_RELAY_ENABLED and not DOOR_SIMULATION:

        relay.value(UNLOCK_VALUE)

        log(
            "Relay activated | commandId="
            + command_id
        )

    else:

        log(
            "Simulation unlock initiated; "
            "relay remains secure | commandId="
            + command_id
        )

    unlock_task = asyncio.create_task(
        restore_after_delay(command_id)
    )

    return "initiated"


# --------------------------------------------------
# HTTP response
# --------------------------------------------------

def response(status, body):

    phrases = {
        200: "OK",
        400: "Bad Request",
        401: "Unauthorized",
        404: "Not Found",
        405: "Method Not Allowed",
        413: "Payload Too Large",
    }

    payload = json.dumps(body)

    return (
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s"
    ) % (
        status,
        phrases.get(status, "Error"),
        len(payload),
        payload,
    )


async def send_response(writer, status, body):
    writer.write(response(status, body))
    await writer.drain()


# --------------------------------------------------
# HTTP request handler
# --------------------------------------------------

async def handle(reader, writer):

    try:

        # Read request line.
        request_line = await reader.readline()

        if not request_line:
            await send_response(
                writer, 400,
                {"success": False, "error": "Empty request"}
            )
            return

        try:
            parts = request_line.decode().strip().split()

        except UnicodeError:
            await send_response(
                writer, 400,
                {"success": False, "error": "Invalid request encoding"}
            )
            return

        if len(parts) != 3:
            await send_response(
                writer, 400,
                {"success": False, "error": "Invalid request"}
            )
            return

        method, path, _ = parts

        # Read headers.
        headers = {}

        while True:

            line = await reader.readline()

            if line in (b"\r\n", b"\n", b""):
                break

            try:
                name, value = line.decode().split(":", 1)
                headers[name.lower().strip()] = value.strip()

            except (ValueError, UnicodeError):
                await send_response(
                    writer, 400,
                    {"success": False, "error": "Invalid HTTP header"}
                )
                return

        # Validate method and endpoint.
        if method != "POST":
            await send_response(
                writer, 405,
                {"success": False, "error": "POST required"}
            )
            return

        if path != "/api/door":
            await send_response(
                writer, 404,
                {"success": False, "error": "Not found"}
            )
            return

        # Validate content type.
        content_type = headers.get("content-type", "")

        if content_type.split(";", 1)[0].strip().lower() != "application/json":

            await send_response(
                writer, 400,
                {
                    "success": False,
                    "error": "Content-Type must be application/json"
                }
            )
            return

        # Authenticate request.
        auth = headers.get("authorization", "")

        if (
            not auth.startswith("Bearer ")
            or not constant_time_equal(auth[7:], DOOR_API_TOKEN)
        ):

            log("Authentication rejected")

            await send_response(
                writer, 401,
                {"success": False, "error": "Unauthorized"}
            )
            return

        # Validate content length.
        try:
            length = int(headers.get("content-length", "0"))

        except ValueError:
            await send_response(
                writer, 400,
                {"success": False, "error": "Invalid Content-Length"}
            )
            return

        if length < 1:
            await send_response(
                writer, 400,
                {"success": False, "error": "Empty request body"}
            )
            return

        if length > 512:
            await send_response(
                writer, 413,
                {"success": False, "error": "Payload too large"}
            )
            return

        # Read and decode JSON.
        try:
            raw_body = await reader.readexactly(length)
            body = json.loads(raw_body.decode())

        except (ValueError, UnicodeError, OSError):
            await send_response(
                writer, 400,
                {"success": False, "error": "Invalid JSON"}
            )
            return

        # Validate JSON structure.
        if not isinstance(body, dict):
            await send_response(
                writer, 400,
                {"success": False, "error": "JSON object required"}
            )
            return

        recognized = body.get("recognized")

        if not isinstance(recognized, bool):
            await send_response(
                writer, 400,
                {"success": False, "error": "recognized must be boolean"}
            )
            return

        # Unknown / non-recognized person.
        if recognized is False:

            await send_response(
                writer, 200,
                {
                    "success": True,
                    "message": "No unlock requested"
                }
            )
            return

        # Validate command ID.
        command_id = (
            body.get("command_id")
            or headers.get("x-command-id")
        )

        if not valid_command_id(command_id):

            await send_response(
                writer, 400,
                {
                    "success": False,
                    "error": "Valid command_id required"
                }
            )
            return

        # Remote approvals are short-lived one-time commands. An ESP32 must
        # have NTP/RTC time synchronized before it will release the relay.
        if body.get("alert_id") is not None and not valid_expiry(body.get("expires_at_unix")):
            await send_response(
                writer, 400,
                {"success": False, "error": "Valid unexpired command expiry required"}
            )
            return

        log(
            "HTTP request authenticated and validated | commandId="
            + command_id
        )

        # Execute door command.
        state = await unlock(command_id)

        await send_response(
            writer, 200,
            {
                "success": True,
                "message": "Door unlock initiated",
                "state": state
            }
        )

    except Exception as error:

        log("Request error: " + str(error))

        try:
            await send_response(
                writer, 400,
                {"success": False, "error": "Request processing failed"}
            )
        except Exception:
            pass

    finally:

        try:
            await writer.aclose()

        except AttributeError:
            writer.close()


# --------------------------------------------------
# Wi-Fi connection
# --------------------------------------------------

async def connect_wifi():

    station = network.WLAN(network.STA_IF)

    station.active(True)

    if not station.isconnected():

        log("Connecting Wi-Fi")

        station.connect(WIFI_SSID, WIFI_PASSWORD)

        for _ in range(30):

            if station.isconnected():
                break

            await asyncio.sleep(1)

    if not station.isconnected():
        raise RuntimeError("Wi-Fi connection failed")

    log(
        "Wi-Fi connected | ip="
        + station.ifconfig()[0]
    )


# --------------------------------------------------
# Main application
# --------------------------------------------------

async def main():

    set_secure()

    await connect_wifi()

    gc.collect()

    server = await asyncio.start_server(
        handle,
        "0.0.0.0",
        HTTP_PORT
    )

    log(
        "HTTP server listening on port %d"
        % HTTP_PORT
    )

    await server.wait_closed()


# --------------------------------------------------
# Startup
# --------------------------------------------------

try:

    asyncio.run(main())

finally:

    set_secure()

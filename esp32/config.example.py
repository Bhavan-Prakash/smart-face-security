# Copy to config.py on the ESP32 filesystem. Never commit real Wi-Fi or API secrets.
WIFI_SSID = ""
WIFI_PASSWORD = ""

# Use a GPIO compatible with your exact ESP32 board and relay module.
DOOR_RELAY_GPIO = 26
DOOR_RELAY_ACTIVE_LOW = True
DOOR_UNLOCK_DURATION = 5
DOOR_MAX_UNLOCK_DURATION = 10
DOOR_API_TOKEN = ""
# Remote-approval commands require a synchronized epoch clock. Configure NTP
# for your MicroPython build before enabling the relay.

# Safe defaults: initialize relay in secure/locked state and simulate requests.
DOOR_RELAY_ENABLED = False
DOOR_SIMULATION = True
HTTP_PORT = 80

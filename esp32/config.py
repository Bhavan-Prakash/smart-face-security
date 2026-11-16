# Copy to config.py on the ESP32 filesystem. Never commit real Wi-Fi or API secrets.
WIFI_SSID = "hogwarts 4G"
WIFI_PASSWORD = "Nandan123@"

# Use a GPIO compatible with your exact ESP32 board and relay module.
DOOR_RELAY_GPIO = 26
DOOR_RELAY_ACTIVE_LOW = True
DOOR_UNLOCK_DURATION = 40
DOOR_MAX_UNLOCK_DURATION = 100
DOOR_API_TOKEN = "6be24704554363d39f051d713caefe4164914861cefb1c940aff51a62f20d908"

# Safe defaults: initialize relay in secure/locked state and simulate requests.
# DOOR_RELAY_ENABLED = False
# DOOR_SIMULATION = True
HTTP_PORT = 80

DOOR_RELAY_ENABLED = True
DOOR_SIMULATION = False

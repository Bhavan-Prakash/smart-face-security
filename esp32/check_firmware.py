
with open("door_controller.py", "r") as f:
    code = f.read()

print("================================")
print("ESP32 FIRMWARE DIAGNOSTICS")
print("================================")

print(
    "New validator:",
    "def valid_command_id" in code
)

print(
    "Old validator:",
    ".isalnum()" in code
)

print(
    "File size:",
    len(code),
    "bytes"
)

print("================================")

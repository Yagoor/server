"""Constants for Auracast provider configuration."""

CONF_SERIAL_PORT = "serial_port"
CONF_BAUDRATE = "baudrate"
CONF_BROADCAST_NAME = "broadcast_name"
CONF_MANUFACTURER_DATA = "manufacturer_data"

DEFAULT_BAUDRATE = 1_000_000
MIN_BAUDRATE = 9_600
MAX_BAUDRATE = 4_000_000
DEFAULT_BROADCAST_NAME = "Music Assistant Auracast"

# TODO: replace the placeholder entries with the (USB vendor ID, product ID) of tested controllers
SUPPORTED_USB_IDS = frozenset({(0xFFFF, 0x0001), (0xFFFF, 0x0002)})

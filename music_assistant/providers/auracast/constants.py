"""Constants for Auracast provider configuration."""

from __future__ import annotations

from typing import NamedTuple


class AdvertisingIntervals(NamedTuple):
    """Primary advertising (0.625 ms units) and periodic advertising (1.25 ms units) intervals."""

    primary: int
    periodic: int


CONF_SERIAL_PORT = "serial_port"
CONF_BAUDRATE = "baudrate"
CONF_BROADCAST_NAME = "broadcast_name"
CONF_MANUFACTURER_DATA = "manufacturer_data"
CONF_ADVERTISING_RATE = "advertising_rate"

ADVERTISING_RATE_FAST = "fast"
ADVERTISING_RATE_SLOW = "slow"
DEFAULT_ADVERTISING_RATE = ADVERTISING_RATE_FAST

ADVERTISING_INTERVALS = {
    ADVERTISING_RATE_FAST: AdvertisingIntervals(primary=96, periodic=48),
    ADVERTISING_RATE_SLOW: AdvertisingIntervals(primary=240, periodic=120),
}

DEFAULT_BAUDRATE = 1_000_000
MIN_BAUDRATE = 9_600
MAX_BAUDRATE = 4_000_000
DEFAULT_BROADCAST_NAME = "Music Assistant Auracast"
MIN_BROADCAST_NAME_BYTES = 4
MAX_BROADCAST_NAME_BYTES = 32

SUPPORTED_USB_IDS = frozenset({(0x2FE3, 0x0004)})

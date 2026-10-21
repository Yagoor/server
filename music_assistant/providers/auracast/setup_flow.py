"""Setup flow for the Auracast player provider."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType
from serial.tools import list_ports

from music_assistant.models.setup_flow import SetupFlowError

from .broadcaster import (
    CONTROLLER_STACK_SETUP_KEY,
    parse_manufacturer_data,
    probe_controller,
)
from .constants import (
    CONF_BAUDRATE,
    CONF_BROADCAST_NAME,
    CONF_MANUFACTURER_DATA,
    CONF_SERIAL_PORT,
    DEFAULT_BAUDRATE,
    DEFAULT_BROADCAST_NAME,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType

    from music_assistant.models.setup_flow import SetupSession

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DiscoveredController:
    """Controller label and optional HCI identity found during setup."""

    label: str
    controller_stack: str | None


async def run_setup(session: SetupSession) -> None:
    """
    Select a compatible controller and configure its broadcast.

    :param session: The setup session driving the flow.
    """
    setup_data = {**session.context.values, **session.context.setup_data}
    baudrate = int(setup_data.get(CONF_BAUDRATE, DEFAULT_BAUDRATE))
    controllers = await discover_controllers(baudrate)
    if not controllers:
        raise SetupFlowError("No compatible Auracast controllers were found")

    existing_port = str(setup_data.get(CONF_SERIAL_PORT) or "")
    selected_port = existing_port if existing_port in controllers else next(iter(controllers))
    errors: dict[str, str | SetupFlowError] | None = None
    while True:
        values = await session.form(
            [
                ConfigEntry(
                    key=CONF_SERIAL_PORT,
                    type=ConfigEntryType.STRING,
                    required=True,
                    value=selected_port,
                    options=[
                        ConfigValueOption(port, title=controllers[port].label)
                        for port in controllers
                    ],
                ),
                ConfigEntry(
                    key=CONF_BROADCAST_NAME,
                    type=ConfigEntryType.STRING,
                    required=True,
                    default_value=DEFAULT_BROADCAST_NAME,
                    value=setup_data.get(CONF_BROADCAST_NAME, DEFAULT_BROADCAST_NAME),
                ),
                ConfigEntry(
                    key=CONF_MANUFACTURER_DATA,
                    type=ConfigEntryType.STRING,
                    required=False,
                    default_value="",
                    value=setup_data.get(CONF_MANUFACTURER_DATA, ""),
                    validate=_validate_manufacturer_data,
                ),
            ],
            step_id="controller",
            errors=errors,
            last_step=True,
        )
        setup_data.update(values)
        selected_controller = controllers.get(str(setup_data.get(CONF_SERIAL_PORT, "")))
        setup_data.pop(CONTROLLER_STACK_SETUP_KEY, None)
        if selected_controller and selected_controller.controller_stack:
            setup_data[CONTROLLER_STACK_SETUP_KEY] = selected_controller.controller_stack
        previous_values: dict[str, Any] = {}
        reconfigure_instance_id: str | None = None
        if session.context.kind == "reconfigure":
            reconfigure_instance_id = session.context.instance_id
            if reconfigure_instance_id is None:
                raise SetupFlowError("Cannot reconfigure Auracast without a provider instance")
            previous_values = {key: session.context.values.get(key) for key in values}
            for key, value in values.items():
                session.mass.config.set_raw_provider_config_value(
                    reconfigure_instance_id, key, value
                )
        try:
            await session.finish(setup_data)
            return
        except BaseException as err:
            if session.context.kind == "reconfigure" and reconfigure_instance_id:
                for key, value in previous_values.items():
                    session.mass.config.set_raw_provider_config_value(
                        reconfigure_instance_id, key, value
                    )
            if not isinstance(err, SetupFlowError):
                raise
            errors = {"base": err}


async def discover_controllers(baudrate: int) -> dict[str, DiscoveredController]:
    """
    Enumerate serial devices and return compatible Auracast controllers.

    :param baudrate: UART baud rate used to probe each serial device.
    """
    ports = await asyncio.to_thread(list_ports.comports)
    controllers: dict[str, DiscoveredController] = {}
    for port in ports:
        LOGGER.debug(
            "Checking serial port candidate %s (%s) at %d baud",
            port.device,
            port.description,
            baudrate,
        )
        try:
            probe_result = await asyncio.wait_for(
                probe_controller(port.device, baudrate), timeout=8
            )
        except TimeoutError:
            LOGGER.debug("Timed out probing serial port candidate %s", port.device)
            probe_result = None
        if probe_result and probe_result.supports_le_periodic_advertising:
            controllers[port.device] = DiscoveredController(
                label=f"{port.description} ({port.device})",
                controller_stack=probe_result.controller_stack,
            )
            LOGGER.debug(
                "Found Auracast-capable controller at %s (stack=%s)",
                port.device,
                probe_result.controller_stack,
            )
    LOGGER.debug(
        "Auracast controller discovery completed: %d of %d serial ports are compatible",
        len(controllers),
        len(ports),
    )
    return controllers


def _validate_manufacturer_data(value: Any) -> bool:
    """Return whether optional manufacturer data uses the supported format."""
    try:
        parse_manufacturer_data(str(value or ""))
    except ValueError:
        return False
    return True


def _validate_baudrate(value: ConfigValueType) -> bool:
    """Return whether a configured UART baud rate is within the supported range."""
    if not isinstance(value, str | int) or isinstance(value, bool):
        return False
    try:
        numeric_value = int(value)
    except ValueError:
        return False
    return 9_600 <= numeric_value <= 4_000_000

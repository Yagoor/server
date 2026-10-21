"""Tests for Auracast controller discovery and setup flow."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from serial.tools import list_ports

from music_assistant.models.setup_flow import SetupFlowError
from music_assistant.providers.auracast import setup_flow
from music_assistant.providers.auracast.broadcaster import ControllerProbeResult


def controller(label: str, stack: str | None = None) -> setup_flow.DiscoveredController:
    """Build one discovered controller for setup-flow tests."""
    return setup_flow.DiscoveredController(label, stack)


@pytest.mark.asyncio
async def test_discover_controllers_returns_only_compatible_serial_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filter serial ports using the selected baud rate and Bumble probe."""
    monkeypatch.setattr(
        list_ports,
        "comports",
        lambda: [
            MagicMock(device="/dev/controller", description="Auracast controller"),
            MagicMock(device="/dev/other", description="Other UART"),
        ],
    )
    probe = AsyncMock(
        side_effect=[
            ControllerProbeResult(True, "Bluetooth Core 5.2; company ID 0x0057"),
            ControllerProbeResult(False, None),
        ]
    )
    monkeypatch.setattr(setup_flow, "probe_controller", probe)

    assert await setup_flow.discover_controllers(1_000_000) == {
        "/dev/controller": controller(
            "Auracast controller (/dev/controller)", "Bluetooth Core 5.2; company ID 0x0057"
        ),
    }
    assert probe.await_args_list[0].args == ("/dev/controller", 1_000_000)
    assert probe.await_args_list[1].args == ("/dev/other", 1_000_000)


@pytest.mark.asyncio
async def test_setup_reports_when_no_compatible_controller_is_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not finish provider setup without a compatible controller choice."""
    monkeypatch.setattr(setup_flow, "discover_controllers", AsyncMock(return_value={}))
    session = MagicMock()
    session.context.kind = "setup"
    session.context.setup_data = {}
    session.context.values = {}
    session.form = AsyncMock()
    session.finish = AsyncMock()

    with pytest.raises(SetupFlowError, match="No compatible Auracast controllers"):
        await setup_flow.run_setup(cast("Any", session))

    session.form.assert_not_awaited()
    session.finish.assert_not_awaited()


@pytest.mark.asyncio
async def test_setup_offers_only_discovered_controllers_and_keeps_optional_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Present valid controllers and retain the optional manufacturer data."""
    monkeypatch.setattr(
        setup_flow,
        "discover_controllers",
        AsyncMock(
            return_value={
                "/dev/controller": controller("Auracast controller (/dev/controller)", "Details"),
                "/dev/controller2": controller("Auracast controller 2 (/dev/controller2)"),
            }
        ),
    )
    session = MagicMock()
    session.context.kind = "setup"
    session.context.setup_data = {"manufacturer_data": "87:abcd"}
    session.context.values = {}
    session.form = AsyncMock(return_value={"serial_port": "/dev/controller"})
    session.finish = AsyncMock()

    await setup_flow.run_setup(cast("Any", session))

    entries = session.form.await_args.args[0]
    port_entry = next(entry for entry in entries if entry.key == "serial_port")
    manufacturer_entry = next(entry for entry in entries if entry.key == "manufacturer_data")
    assert [entry.key for entry in entries] == [
        "serial_port",
        "broadcast_name",
        "manufacturer_data",
    ]
    assert [option.value for option in port_entry.options] == [
        "/dev/controller",
        "/dev/controller2",
    ]
    assert manufacturer_entry.required is False
    session.finish.assert_awaited_once_with(
        {
            "manufacturer_data": "87:abcd",
            "serial_port": "/dev/controller",
            "_controller_stack": "Details",
        }
    )


@pytest.mark.asyncio
async def test_reconfigure_persists_submitted_values_as_provider_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply setup-flow edits to ordinary provider values as well as setup data."""
    monkeypatch.setattr(
        setup_flow,
        "discover_controllers",
        AsyncMock(
            return_value={
                "/dev/controller": controller("Auracast controller (/dev/controller)", "Details")
            }
        ),
    )
    session = MagicMock()
    session.context.kind = "reconfigure"
    session.context.instance_id = "auracast_1"
    session.context.values = {"serial_port": "/dev/old", "broadcast_name": "Old name"}
    session.context.setup_data = {"serial_port": "/dev/old"}
    session.form = AsyncMock(
        return_value={"serial_port": "/dev/controller", "broadcast_name": "New name"}
    )
    session.finish = AsyncMock()

    await setup_flow.run_setup(cast("Any", session))

    assert session.mass.config.set_raw_provider_config_value.call_args_list[0].args == (
        "auracast_1",
        "serial_port",
        "/dev/controller",
    )
    assert session.mass.config.set_raw_provider_config_value.call_args_list[1].args == (
        "auracast_1",
        "broadcast_name",
        "New name",
    )
    session.finish.assert_awaited_once_with(
        {
            "serial_port": "/dev/controller",
            "broadcast_name": "New name",
            "_controller_stack": "Details",
        }
    )

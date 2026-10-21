"""Tests for Auracast provider configuration updates."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ConfigEntryType, PlaybackState

from music_assistant.providers.auracast import AuracastProvider


@pytest.mark.asyncio
async def test_setup_data_values_seed_provider_settings() -> None:
    """Expose submitted setup-flow answers as initial provider setting values."""
    provider = MagicMock()
    setup_values = {
        "serial_port": "/dev/controller",
        "baudrate": 1_000_000,
        "broadcast_name": "Auracast test",
        "manufacturer_data": "87:abcd",
    }
    provider.get_setup_value.side_effect = lambda key, default=None: setup_values.get(key, default)

    entries = await AuracastProvider.get_config_entries(cast("Any", provider))

    assert {entry.key: entry.default_value for entry in entries} == {
        **setup_values,
        "baudrate": "1000000",
    }
    assert {entry.type for entry in entries} == {ConfigEntryType.STRING}


@pytest.mark.asyncio
async def test_config_update_reloads_controller_and_resumes_playing_queue() -> None:
    """Stop active playback, reload changed controller settings, then resume."""
    provider = MagicMock()
    player = MagicMock()
    queue = MagicMock()
    queue.queue_id = "auracast_test"
    queue.state = PlaybackState.PLAYING
    provider.mass.players.get_player.return_value = player
    provider.mass.players.get_active_queue.return_value = queue
    provider.mass.player_queues.stop = AsyncMock()
    provider.mass.player_queues.resume = AsyncMock()
    provider.mass.load_provider_config = AsyncMock()
    updated_config = MagicMock()

    await AuracastProvider.update_config(
        cast("Any", provider), updated_config, {"values/broadcast_name"}
    )

    provider.mass.player_queues.stop.assert_awaited_once_with("auracast_test")
    provider.mass.load_provider_config.assert_awaited_once_with(updated_config)
    provider.mass.player_queues.resume.assert_awaited_once_with("auracast_test", False)


@pytest.mark.asyncio
async def test_config_update_does_not_resume_idle_queue() -> None:
    """Reload settings without starting playback when the queue was idle."""
    provider = MagicMock()
    player = MagicMock()
    queue = MagicMock()
    queue.queue_id = "auracast_test"
    queue.state = PlaybackState.IDLE
    provider.mass.players.get_player.return_value = player
    provider.mass.players.get_active_queue.return_value = queue
    provider.mass.player_queues.stop = AsyncMock()
    provider.mass.player_queues.resume = AsyncMock()
    provider.mass.load_provider_config = AsyncMock()

    await AuracastProvider.update_config(
        cast("Any", provider), MagicMock(), {"values/manufacturer_data"}
    )

    provider.mass.player_queues.stop.assert_not_awaited()
    provider.mass.player_queues.resume.assert_not_awaited()
    provider.mass.load_provider_config.assert_awaited_once()

"""Tests for the Music Assistant Auracast player lifecycle."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import PlayerFeature

from music_assistant.models.player import PlayerMedia
from music_assistant.providers.auracast.broadcaster import CONTROLLER_STACK_SETUP_KEY
from music_assistant.providers.auracast.player import PCM_FORMAT, AuracastPlayer, _apply_volume


def create_player() -> AuracastPlayer:
    """Build a player with a minimal provider and mass stub."""
    mass = MagicMock()
    mass.config.get_base_player_config.return_value = MagicMock()
    mass.config.get_raw_player_config_value.return_value = 100
    provider = MagicMock()
    provider.mass = mass
    provider.logger = logging.getLogger("test.auracast")
    provider.instance_id = "auracast_test"
    provider.config.get_value.side_effect = lambda _key, default=None: default
    provider.get_setup_value.side_effect = lambda _key, default=None: default
    return AuracastPlayer(provider, "auracast_test")


async def pcm_source(chunk: bytes) -> AsyncGenerator[bytes]:
    """Yield a single PCM chunk."""
    yield chunk


@pytest.mark.asyncio
async def test_play_media_streams_pcm_and_closes_broadcast_on_eof() -> None:
    """Request decoded PCM, forward it, and release the broadcast after EOF."""
    player = cast("Any", create_player())
    media = MagicMock(spec=PlayerMedia)
    chunk = bytes(480 * 2 * 2)
    player.mass.streams.get_stream.return_value = pcm_source(chunk)
    player._broadcaster.start = AsyncMock()
    player._broadcaster.stop = AsyncMock()
    player._broadcaster.send_pcm = AsyncMock()
    player._broadcaster.controller_stack = (
        "Bluetooth Core 5.2; company ID 0x0057; HCI revision 0x1234; "
        "LMP version 11; firmware revision 0x5678"
    )
    player.update_state = MagicMock()

    await player.play_media(media)
    assert player._stream_task is not None
    await player._stream_task

    player.mass.streams.get_stream.assert_called_once_with(media, PCM_FORMAT, player.player_id)
    player._broadcaster.start.assert_awaited_once()
    player._broadcaster.send_pcm.assert_awaited_once_with(chunk)
    assert player._broadcaster.stop.await_count == 2
    assert player.current_media is None
    assert player.device_info.manufacturer == "Bluetooth SIG"
    assert player.device_info.model == (
        "Host stack: Bumble; Controller stack: Bluetooth Core 5.2; company ID 0x0057; "
        "HCI revision 0x1234; LMP version 11; firmware revision 0x5678"
    )


@pytest.mark.asyncio
async def test_volume_set_is_supported_and_scales_little_endian_pcm() -> None:
    """Expose the UI volume control and attenuate PCM before LC3 encoding."""
    player = cast("Any", create_player())
    player.update_state = MagicMock()
    pcm = b"\xff\x7f\x00\x80"

    assert PlayerFeature.VOLUME_SET in player.supported_features
    assert _apply_volume(pcm, 50) == b"\x00\x40\x00\xc0"
    assert _apply_volume(pcm, 0) == bytes(4)
    assert _apply_volume(pcm, 100) == pcm

    # The web UI's volume command updates the attenuation applied to subsequent frames.
    await player.volume_set(35)
    assert player.volume_level == 35
    player.mass.config.set_raw_player_config_value.assert_called_once_with(
        player.player_id, "auracast_stored_volume", 35
    )
    player.update_state.assert_called_once()


def test_player_restores_persisted_volume() -> None:
    """Restore the previously selected software volume after player reload."""
    player = create_player()
    cast("Any", player.mass.config.get_raw_player_config_value).return_value = 42

    restored_player = AuracastPlayer(cast("Any", player.provider), player.player_id)

    assert restored_player.volume_level == 42


def test_player_shows_controller_details_from_setup_probe() -> None:
    """Show the HCI identity discovered during setup before broadcasting starts."""
    player = create_player()
    cast("Any", player.provider).get_setup_value.side_effect = lambda key, default=None: (
        "Bluetooth Core 5.4; company ID 0x0059" if key == CONTROLLER_STACK_SETUP_KEY else default
    )

    player_with_details = AuracastPlayer(cast("Any", player.provider), player.player_id)

    assert player_with_details.device_info.model == (
        "Host stack: Bumble; Controller stack: Bluetooth Core 5.4; company ID 0x0059"
    )

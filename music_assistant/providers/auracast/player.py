"""Music Assistant player backed by a Bluetooth LE Audio broadcast."""

from __future__ import annotations

import asyncio
import sys
from array import array
from contextlib import suppress
from typing import TYPE_CHECKING, cast

from music_assistant_models.enums import ContentType, PlaybackState, PlayerFeature, PlayerType
from music_assistant_models.media_items.audio_format import AudioFormat

from music_assistant.helpers.audio import iter_pcm_slices
from music_assistant.models.player import Player, PlayerMedia

from .broadcaster import (
    CONTROLLER_STACK_SETUP_KEY,
    AuracastBroadcaster,
    BroadcastConfig,
    parse_manufacturer_data,
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
    from collections.abc import AsyncGenerator

    from . import AuracastProvider

PCM_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S16LE,
    codec_type=ContentType.PCM_S16LE,
    sample_rate=48_000,
    bit_depth=16,
    channels=2,
)
FRAME_DURATION_MS = 10
CONF_STORED_VOLUME = "auracast_stored_volume"


class AuracastPlayer(Player):
    """Expose the configured Auracast broadcast as an MA player."""

    def __init__(self, provider: AuracastProvider, player_id: str) -> None:
        """Initialize the player and its PCM stream state."""
        super().__init__(provider, player_id)
        self._attr_name = str(
            provider.config.get_value(CONF_BROADCAST_NAME, DEFAULT_BROADCAST_NAME)
        )
        serial_port = str(provider.config.get_value(CONF_SERIAL_PORT, "UART controller"))
        controller_stack = provider.get_setup_value(CONTROLLER_STACK_SETUP_KEY)
        if not isinstance(controller_stack, str) or not controller_stack:
            controller_stack = f"UART HCI ({serial_port})"
        self._attr_device_info.manufacturer = "Bluetooth SIG"
        self._attr_device_info.model = f"Host stack: Bumble; Controller stack: {controller_stack}"
        self._attr_type = PlayerType.PLAYER
        self._attr_supported_features = {
            PlayerFeature.PLAY_MEDIA,
            PlayerFeature.PAUSE,
            PlayerFeature.POWER,
            PlayerFeature.VOLUME_SET,
        }
        self._attr_powered = True
        self._attr_available = True
        self._attr_volume_level = max(
            0,
            min(
                100,
                int(
                    provider.mass.config.get_raw_player_config_value(
                        player_id, CONF_STORED_VOLUME, 100
                    )
                ),
            ),
        )
        self._broadcaster = AuracastBroadcaster(
            BroadcastConfig(
                name=self._attr_name,
                transport_spec=(
                    f"serial:{provider.config.get_value(CONF_SERIAL_PORT, '/dev/cu.usbmodem112301')},"
                    f"{provider.config.get_value(CONF_BAUDRATE, DEFAULT_BAUDRATE)}"
                ),
                manufacturer_data=parse_manufacturer_data(
                    cast(
                        "str | None",
                        provider.config.get_value(CONF_MANUFACTURER_DATA, ""),
                    )
                ),
            )
        )
        self._stream_task: asyncio.Task[None] | None = None
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._generation = 0

    @property
    def needs_poll(self) -> bool:
        """Return whether player state requires polling."""
        return False

    async def play_media(self, media: PlayerMedia) -> None:
        """Start streaming decoded PCM to the Auracast broadcaster."""
        await self._stop_playback()
        if not self._attr_powered:
            self._attr_powered = True
        await self._broadcaster.start()
        if self._broadcaster.controller_stack:
            self._attr_device_info.model = (
                f"Host stack: Bumble; Controller stack: {self._broadcaster.controller_stack}"
            )
            self.update_state()
        audio_source = self.mass.streams.get_stream(media, PCM_FORMAT, self.player_id)
        self._attr_current_media = media
        self._attr_playback_state = PlaybackState.PLAYING
        self._resume_event.set()
        self._generation += 1
        generation = self._generation
        self._stream_task = asyncio.create_task(
            self._stream_audio(audio_source, generation),
            name=f"auracast_stream_{self.player_id}",
        )
        self.update_state()

    async def play(self) -> None:
        """Resume a paused broadcast stream."""
        if self._stream_task and not self._stream_task.done():
            self._resume_event.set()
            self._attr_playback_state = PlaybackState.PLAYING
            self.update_state()

    async def pause(self) -> None:
        """Pause the source stream while keeping the broadcast active."""
        if self._stream_task and not self._stream_task.done():
            self._resume_event.clear()
            self._attr_playback_state = PlaybackState.PAUSED
            self.update_state()

    async def volume_set(self, volume_level: int) -> None:
        """
        Set software playback volume before LC3 encoding.

        :param volume_level: Volume level from 0 to 100.
        """
        self._attr_volume_level = max(0, min(100, volume_level))
        self.mass.config.set_raw_player_config_value(
            self.player_id, CONF_STORED_VOLUME, self._attr_volume_level
        )
        self.update_state()

    async def stop(self) -> None:
        """Stop streaming and release the controller transport."""
        await self._stop_playback()
        await self._broadcaster.stop()
        self._attr_current_media = None
        self._attr_playback_state = PlaybackState.IDLE
        self.update_state()

    async def power(self, powered: bool) -> None:
        """Power the broadcast player on or stop and power it off."""
        if powered:
            self._attr_powered = True
        else:
            await self.stop()
            self._attr_powered = False
        self.update_state()

    async def on_unload(self) -> None:
        """Stop playback and release controller resources when removed."""
        await self.stop()

    async def _stream_audio(self, audio_source: AsyncGenerator[bytes], generation: int) -> None:
        """Forward frame-aligned PCM to the broadcaster until the stream ends."""
        frame_bytes = PCM_FORMAT.sample_rate * FRAME_DURATION_MS // 1000 * 4
        silence = bytes(frame_bytes)
        completed = False
        try:
            async for chunk in audio_source:
                while not self._resume_event.is_set():
                    await self._broadcaster.send_pcm(silence)
                    await asyncio.sleep(FRAME_DURATION_MS / 1000)
                for pcm_slice in iter_pcm_slices(chunk, PCM_FORMAT, FRAME_DURATION_MS):
                    volume_level = self._attr_volume_level
                    await self._broadcaster.send_pcm(
                        _apply_volume(pcm_slice, 100 if volume_level is None else volume_level)
                    )
            completed = True
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Auracast audio stream failed")
        finally:
            with suppress(Exception):
                await audio_source.aclose()
            if generation == self._generation:
                self._stream_task = None
                await self._broadcaster.stop()
                self._attr_current_media = None
                self._attr_playback_state = PlaybackState.IDLE
                self.update_state()
            if completed:
                self.logger.debug("Auracast audio stream completed")

    async def _stop_playback(self) -> None:
        """Cancel the current source stream and wait for its cleanup."""
        self._generation += 1
        self._resume_event.set()
        task, self._stream_task = self._stream_task, None
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await self._broadcaster.stop()


def _apply_volume(pcm: bytes, volume_level: int) -> bytes:
    """Scale little-endian signed 16-bit PCM by a 0..100 volume level."""
    if volume_level >= 100:
        return pcm
    if volume_level <= 0:
        return bytes(len(pcm))
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    for index, sample in enumerate(samples):
        samples[index] = round(sample * volume_level / 100)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()

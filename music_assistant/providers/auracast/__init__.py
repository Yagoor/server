"""Auracast Bluetooth LE Audio player provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType, PlaybackState

from music_assistant.constants import CONF_LOG_LEVEL
from music_assistant.models.player_provider import PlayerProvider

from .constants import (
    CONF_BAUDRATE,
    CONF_BROADCAST_NAME,
    CONF_MANUFACTURER_DATA,
    CONF_SERIAL_PORT,
    DEFAULT_BAUDRATE,
    DEFAULT_BROADCAST_NAME,
)
from .player import AuracastPlayer

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


class AuracastProvider(PlayerProvider):
    """Provider that broadcasts one MA player over an Auracast controller."""

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return the controller and broadcast settings."""
        serial_port = str(
            self.get_setup_value(CONF_SERIAL_PORT, "/dev/cu.usbmodem112301")
            or "/dev/cu.usbmodem112301"
        )
        baudrate = int(cast("int | str", self.get_setup_value(CONF_BAUDRATE, DEFAULT_BAUDRATE)))
        broadcast_name = str(
            self.get_setup_value(CONF_BROADCAST_NAME, DEFAULT_BROADCAST_NAME)
            or DEFAULT_BROADCAST_NAME
        )
        manufacturer_data = str(self.get_setup_value(CONF_MANUFACTURER_DATA, "") or "")
        return (
            ConfigEntry(
                key=CONF_SERIAL_PORT,
                type=ConfigEntryType.STRING,
                required=True,
                default_value=serial_port,
                requires_reload=True,
            ),
            ConfigEntry(
                key=CONF_BAUDRATE,
                type=ConfigEntryType.STRING,
                required=True,
                default_value=str(baudrate),
                requires_reload=True,
            ),
            ConfigEntry(
                key=CONF_BROADCAST_NAME,
                type=ConfigEntryType.STRING,
                required=True,
                default_value=broadcast_name,
                requires_reload=True,
            ),
            ConfigEntry(
                key=CONF_MANUFACTURER_DATA,
                type=ConfigEntryType.STRING,
                required=False,
                default_value=manufacturer_data,
                requires_reload=True,
            ),
        )

    async def discover_players(self) -> None:
        """Register the controller as the provider's single player."""
        player = AuracastPlayer(self, f"auracast_{self.instance_id}")
        await self.mass.players.register(player)

    async def update_config(self, config: ProviderConfig, changed_keys: set[str]) -> None:
        """
        Reload the controller and resume an active queue after settings change.

        :param config: The updated provider configuration.
        :param changed_keys: Config keys changed by the update.
        """
        changed_values = {
            key
            for key in changed_keys
            if key.startswith("values/") and key != f"values/{CONF_LOG_LEVEL}"
        }
        if not changed_values:
            await super().update_config(config, changed_keys)
            return

        player_id = f"auracast_{self.instance_id}"
        player = self.mass.players.get_player(player_id)
        active_queue = self.mass.players.get_active_queue(player) if player else None
        resume_queue = bool(active_queue and active_queue.state == PlaybackState.PLAYING)
        queue_id = active_queue.queue_id if active_queue else player_id
        if resume_queue:
            await self.mass.player_queues.stop(queue_id)

        self.config = config
        await self.mass.load_provider_config(config)

        if resume_queue:
            await self.mass.player_queues.resume(queue_id, False)

    async def unload(self, is_removed: bool = False) -> None:
        """Unregister the player and release any active Bluetooth transport."""
        for player in self.players:
            await self.mass.players.unregister(player.player_id)


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize the Auracast player provider."""
    return AuracastProvider(mass, manifest, config, set())

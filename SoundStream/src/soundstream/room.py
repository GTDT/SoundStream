from __future__ import annotations

import logging

from .signaling.client import SignalingClient

logger = logging.getLogger('soundstream')


class RoomManager:

    def __init__(self, signaling: SignalingClient) -> None:
        self._signaling = signaling

    async def create_room(
        self,
        room_id: str,
        session_key: str,
        is_public: bool = False,
        listener_cap: int = 10,
        title: str | None = None,
    ) -> dict:
        result = await self._signaling.create_room(
            room_id=room_id,
            session_key=session_key,
            is_public=is_public,
            listener_cap=listener_cap,
            title=title,
        )
        if result.get('success'):
            logger.info(f'Room created: {room_id}')
        return result

    async def list_rooms(self) -> list[dict]:
        result = await self._signaling.list_rooms()
        return result.get('rooms', [])

    async def get_room(self, room_id: str) -> dict:
        return await self._signaling.get_room(room_id)

    async def delete_room(self, room_id: str) -> dict:
        result = await self._signaling.delete_room(room_id)
        if result.get('success'):
            logger.info(f'Room deleted: {room_id}')
        return result

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import aiohttp

from .config import DEFAULT_SERVER_URL, LONG_POLL_TIMEOUT

logger = logging.getLogger("soundstream")


class SignalingClient:
    def __init__(self, server_url: str = DEFAULT_SERVER_URL) -> None:
        self.server_url = server_url.rstrip("/") + "/"
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(force_close=True)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def create_room(
        self,
        room_id: str,
        session_key: str,
        is_public: bool = False,
        listener_cap: int = 10,
        title: str | None = None,
    ) -> dict:
        session = await self._get_session()
        logger.debug(f"[signal] POST room.create room={room_id}")
        async with session.post(
            f"{self.server_url}?action=room.create",
            json={
                "room_id": room_id,
                "session_key": session_key,
                "is_public": is_public,
                "listener_cap": listener_cap,
                "title": title,
            },
        ) as resp:
            result = await resp.json()
            logger.debug(f"[signal] POST room.create response: {result}")
            return result

    async def heartbeat(self, room_id: str) -> dict:
        session = await self._get_session()
        async with session.post(
            f"{self.server_url}?action=room.heartbeat&room_id={room_id}",
        ) as resp:
            return await resp.json()

    async def listener_join(self, room_id: str) -> dict:
        session = await self._get_session()
        logger.debug(f"[signal] POST listener.join room={room_id}")
        async with session.post(
            f"{self.server_url}?action=listener.join&room_id={room_id}",
        ) as resp:
            result = await resp.json()
            logger.debug(f"[signal] listener.join response: {result}")
            return result

    async def listener_leave(self, room_id: str) -> dict:
        session = await self._get_session()
        logger.debug(f"[signal] POST listener.leave room={room_id}")
        async with session.post(
            f"{self.server_url}?action=listener.leave&room_id={room_id}",
        ) as resp:
            result = await resp.json()
            logger.debug(f"[signal] listener.leave response: {result}")
            return result

    async def delete_room(self, room_id: str) -> dict:
        session = await self._get_session()
        async with session.delete(
            f"{self.server_url}?action=room.delete&room_id={room_id}",
        ) as resp:
            return await resp.json()

    async def send_signal(
        self,
        room_id: str,
        signal_type: str,
        sdp: str | None = None,
        candidate: str | None = None,
    ) -> dict:
        session = await self._get_session()
        data: dict = {"type": signal_type}
        if sdp is not None:
            data["sdp"] = sdp
        if candidate is not None:
            data["candidate"] = candidate
        sdp_len = len(sdp) if sdp else 0
        logger.debug(
            f"[signal] POST signal type={signal_type} room={room_id} sdp_len={sdp_len}"
        )
        async with session.post(
            f"{self.server_url}?action=signaling&room_id={room_id}",
            json=data,
        ) as resp:
            result = await resp.json()
            logger.debug(f"[signal] POST response: {result}")
            return result

    async def poll_signals(
        self,
        room_id: str,
        since_id: int = 0,
        timeout: float = LONG_POLL_TIMEOUT,
    ) -> dict:
        session = await self._get_session()
        logger.debug(f"[signal] GET poll room={room_id} since_id={since_id}")
        async with session.get(
            f"{self.server_url}?action=signaling&room_id={room_id}&since_id={since_id}",
            timeout=aiohttp.ClientTimeout(total=timeout + 5),
        ) as resp:
            result = await resp.json()
            signal = result.get("signal")
            sig_type = signal["type"] if signal else None
            sig_id = signal["id"] if signal else None
            logger.debug(
                f"[signal] GET poll result: type={sig_type} id={sig_id} timeout={result.get('timeout')}"
            )
            return result

    async def list_rooms(self) -> dict:
        session = await self._get_session()
        async with session.get(f"{self.server_url}?action=rooms") as resp:
            return await resp.json()

    async def get_room(self, room_id: str) -> dict:
        session = await self._get_session()
        async with session.get(
            f"{self.server_url}?action=room&room_id={room_id}",
        ) as resp:
            return await resp.json()

    async def send_chat_message(
        self, room_id: str, message: str, session_id: str
    ) -> dict:
        """
        Gintautas: Send a chat message to the server's chat queue.
        Listeners use this endpoint; the streamer polls and re-broadcasts.
        """
        session = await self._get_session()
        logger.debug(f"[signal] POST chat.send room={room_id}")
        async with session.post(
            f"{self.server_url}?action=chat.send&room_id={room_id}",
            json={
                "message": message,
                "sender_session_id": session_id,
            },
        ) as resp:
            result = await resp.json()
            logger.debug(f"[signal] chat.send response: {result}")
            return result

    async def poll_chat_messages(self, room_id: str, since_id: int = 0) -> dict:
        """
        Gintautas: Poll the server for new chat messages queued by listeners.
        Only the streamer should call this endpoint.
        """
        session = await self._get_session()
        logger.debug(f"[signal] GET chat.poll room={room_id} since_id={since_id}")
        async with session.get(
            f"{self.server_url}?action=chat.poll&room_id={room_id}&since_id={since_id}",
            timeout=aiohttp.ClientTimeout(total=LONG_POLL_TIMEOUT + 5),
        ) as resp:
            result = await resp.json()
            logger.debug(f"[signal] chat.poll response: {result}")
            return result

    async def stream_signals(
        self,
        room_id: str,
        since_id: int = 0,
    ) -> AsyncIterator[dict]:
        session = await self._get_session()
        url = f"{self.server_url}?action=signals.stream&room_id={room_id}&since_id={since_id}"
        logger.debug(f"[signal] SSE stream room={room_id} since_id={since_id}")
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=3605),
        ) as resp:
            async for line in resp.content:
                decoded = line.decode("utf-8")
                if not decoded.startswith("data: "):
                    continue
                payload = decoded[len("data: ") :].strip()
                if not payload:
                    continue
                signal = json.loads(payload)
                logger.debug(
                    f"[signal] SSE received: type={signal.get('type')} id={signal.get('id')}",
                )
                yield signal

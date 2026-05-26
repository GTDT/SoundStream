from __future__ import annotations

import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from typing import Callable

logger = logging.getLogger("soundstream")


class ChatManager(ABC):
    @abstractmethod
    async def start(self, room_id: str) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    async def send(self, message: str) -> None:
        pass

    @abstractmethod
    def on_message(self, callback: Callable[[str, str], None]) -> None:
        pass


class SignalChatManager(ChatManager):
    """
    Gintautas: Chat implementation using the signaling server as transport.

    Listeners POST messages to a server-side queue (chat.send).
    The streamer polls that queue (chat.poll) and re-broadcasts each
    message as a 'chat' type signal so all listeners receive it via
    the existing long-poll / SSE signaling channel.

    A short random session_id is generated per instance so chat lines
    can be attributed as IP:session_id (e.g. 192.168.1.5:516).
    """

    def __init__(self, signaling_client, mode: str = "listener") -> None:
        """
        Gintautas: Initialize the chat manager.

        Args:
            signaling_client: The SignalingClient instance used for HTTP calls.
            mode: Either 'streamer' or 'listener'.
        """
        self._signaling = signaling_client
        self._mode = mode
        self._room_id: str | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._on_message: Callable[[str, str], None] | None = None
        self._last_signal_id = 0
        self._last_chat_id = 0
        # Gintautas: Generate a random 3-digit session identifier for display.
        self._session_id = str(random.randint(100, 999))

    async def start(self, room_id: str) -> None:
        """
        Gintautas: Start polling for chat messages depending on mode.
        """
        self._room_id = room_id
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            f"SignalChatManager started for room: {room_id} (mode={self._mode})"
        )

    async def stop(self) -> None:
        """
        Gintautas: Stop the background polling task.
        """
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("SignalChatManager stopped")

    async def send(self, message: str) -> None:
        """
        Gintautas: Send a chat message.

        Listeners enqueue the message on the server (chat.send).
        Streamers broadcast directly through the signaling table.
        """
        if self._room_id is None:
            raise RuntimeError("Chat not started")
        if self._mode == "streamer":
            payload = json.dumps(
                {
                    "message": message,
                    "sender_ip": "streamer",
                    "sender_session_id": self._session_id,
                }
            )
            await self._signaling.send_signal(
                self._room_id,
                "chat",
                sdp=payload,
            )
            logger.debug(f"Chat broadcast by streamer: {message[:50]}...")
        else:
            await self._signaling.send_chat_message(
                self._room_id,
                message,
                self._session_id,
            )
            logger.debug(f"Chat message sent: {message[:50]}...")

    def on_message(self, callback: Callable[[str, str], None]) -> None:
        """
        Gintautas: Register a callback to receive incoming chat lines.
        """
        self._on_message = callback

    async def _poll_loop(self) -> None:
        """
        Gintautas: Background loop that polls for chat messages.

        - Streamer mode polls the chat_messages queue via chat.poll,
          triggers on_message locally, then re-broadcasts via signaling.
        - Listener mode polls the signaling stream for type='chat'.
        """
        while self._running:
            try:
                if self._room_id is None:
                    await asyncio.sleep(0.5)
                    continue
                if self._mode == "streamer":
                    await self._poll_chat_queue()
                else:
                    await self._poll_signaling_chat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Chat poll error: {e}")
                await asyncio.sleep(1)

    async def _poll_chat_queue(self) -> None:
        """
        Gintautas: Poll the server chat queue (streamer only).
        Messages are deleted on the server after being returned.
        """
        room = self._room_id
        if room is None:
            return
        result = await self._signaling.poll_chat_messages(
            room,
            since_id=self._last_chat_id,
        )
        messages = result.get("messages", [])
        if messages:
            self._last_chat_id = messages[-1].get("id", self._last_chat_id)
            for msg in messages:
                sender = (
                    f"{msg.get('sender_ip', '?')}:{msg.get('sender_session_id', '?')}"
                )
                text = msg.get("message", "")
                room = self._room_id
                if room and self._on_message:
                    self._on_message(room, f"[{sender}] {text}")
                # Gintautas: Re-broadcast to all listeners via signaling table.
                payload = json.dumps(
                    {
                        "message": text,
                        "sender_ip": msg.get("sender_ip", "?"),
                        "sender_session_id": msg.get("sender_session_id", "?"),
                    }
                )
                if room:
                    await self._signaling.send_signal(
                        room,
                        "chat",
                        sdp=payload,
                    )
        else:
            await asyncio.sleep(0.5)

    async def _poll_signaling_chat(self) -> None:
        """
        Gintautas: Poll the signaling stream for chat-type signals (listener only).
        """
        room = self._room_id
        if room is None:
            return
        result = await self._signaling.poll_signals(
            room,
            since_id=self._last_signal_id,
            timeout=5,
        )
        signal = result.get("signal")
        if signal:
            self._last_signal_id = signal.get("id", 0)
            if signal.get("type") == "chat":
                sdp = signal.get("sdp", "{}")
                try:
                    payload = json.loads(sdp)
                    sender_ip = payload.get("sender_ip", "?")
                    sender_session = payload.get("sender_session_id", "?")
                    msg = payload.get("message", "")
                    if msg and self._on_message:
                        display = f"[{sender_ip}:{sender_session}] {msg}"
                        room = self._room_id
                        if room:
                            self._on_message(room, display)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid chat payload: {sdp}")
        else:
            await asyncio.sleep(0.5)

    @property
    def room_id(self) -> str | None:
        return self._room_id

    @property
    def session_id(self) -> str:
        return self._session_id

from __future__ import annotations

import asyncio
import logging
import secrets
from typing import Callable

from .audio import (
    AdaptiveBufferController,
    AudioCapture,
    AudioPlayer,
    AudioProcessor,
    JitterBuffer,
)
from .chat import ChatManager, SignalChatManager
from .room import RoomManager
from .signaling.client import SignalingClient
from .signaling.config import DEFAULT_SERVER_URL, HEARTBEAT_INTERVAL
from .webrtc import AudioSendTrack, WebRTCManager, setup_ice

logger = logging.getLogger("soundstream")


class SoundSession:
    def __init__(self, server_url: str = DEFAULT_SERVER_URL) -> None:
        self._server_url = server_url
        self._signaling = SignalingClient(server_url)
        self._room_manager = RoomManager(self._signaling)

        self._state = "idle"
        self._room_id: str | None = None
        self._session_key: str | None = None

        self._input_device: int | None = None
        self._output_device: int | None = None
        self._sample_rate = 48000
        self._channels = 1
        self._block_size = 480
        self._volume = 1.0

        self._stun_servers = ["stun:stun.l.google.com:19302"]
        self._turn_servers: list[dict] = []

        self._jitter_buffer_enabled = True
        self._jitter_buffer_ms = 20
        self._jitter_buffer_min_ms = 10
        self._jitter_buffer_max_ms = 100

        self._latency = "low"

        self._capture: AudioCapture | None = None
        self._player: AudioPlayer | None = None
        self._jitter: JitterBuffer | None = None
        self._send_track: AudioSendTrack | None = None
        self._webrtc: WebRTCManager | None = None
        self._chat: ChatManager | None = None

        self._heartbeat_task: asyncio.Task | None = None
        self._signal_poll_task: asyncio.Task | None = None
        self._audio_play_task: asyncio.Task | None = None
        self._adaptive_buffer_task: asyncio.Task | None = None

        self._audio_processor = AudioProcessor()

        self.on_state_change: Callable[[str, str], None] | None = None
        self.on_connection_change: Callable[[bool], None] | None = None
        self.on_error: Callable[[Exception], None] | None = None
        self.on_chat_message: Callable[[str, str], None] | None = None

    def _set_state(self, new_state: str) -> None:
        old_state = self._state
        self._state = new_state
        logger.info(f"State: {old_state} -> {new_state}")
        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    @property
    def state(self) -> str:
        return self._state

    @property
    def room_id(self) -> str | None:
        return self._room_id

    @property
    def is_streaming(self) -> bool:
        return self._state == "streaming"

    @property
    def is_listening(self) -> bool:
        return self._state == "listening"

    @property
    def is_connected(self) -> bool:
        return self._webrtc is not None and self._webrtc.is_connected

    @property
    def input_device(self) -> int | None:
        return self._input_device

    @input_device.setter
    def input_device(self, value: int | None) -> None:
        self._input_device = value
        if self._capture is not None:
            self._capture.device = value

    @property
    def output_device(self) -> int | None:
        return self._output_device

    @output_device.setter
    def output_device(self, value: int | None) -> None:
        self._output_device = value
        if self._player is not None:
            self._player.device = value

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        self._sample_rate = value
        if self._capture is not None:
            self._capture.sample_rate = value
        if self._player is not None:
            self._player.sample_rate = value
        if self._jitter is not None:
            self._jitter = JitterBuffer(
                target_ms=self._jitter_buffer_ms,
                min_ms=self._jitter_buffer_min_ms,
                max_ms=self._jitter_buffer_max_ms,
                sample_rate=value,
            )

    @property
    def channels(self) -> int:
        return self._channels

    @channels.setter
    def channels(self, value: int) -> None:
        self._channels = value

    @property
    def block_size(self) -> int:
        return self._block_size

    @block_size.setter
    def block_size(self, value: int) -> None:
        self._block_size = value
        if self._capture is not None:
            self._capture.block_size = value
        if self._player is not None:
            self._player.block_size = value

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = max(0.0, min(10.0, value))
        if self._player is not None:
            self._player.volume = value

    @property
    def server_url(self) -> str:
        return self._server_url

    @property
    def stun_servers(self) -> list[str]:
        return self._stun_servers

    @stun_servers.setter
    def stun_servers(self, value: list[str]) -> None:
        self._stun_servers = value

    @property
    def turn_servers(self) -> list[dict]:
        return self._turn_servers

    @turn_servers.setter
    def turn_servers(self, value: list[dict]) -> None:
        self._turn_servers = value

    @property
    def latency(self) -> str:
        return self._latency

    @latency.setter
    def latency(self, value: str) -> None:
        self._latency = value
        if self._capture is not None:
            self._capture.latency = value
        if self._player is not None:
            self._player.latency = value

    @property
    def jitter_buffer_enabled(self) -> bool:
        return self._jitter_buffer_enabled

    @jitter_buffer_enabled.setter
    def jitter_buffer_enabled(self, value: bool) -> None:
        self._jitter_buffer_enabled = value
        if self._jitter is not None:
            self._jitter.enabled = value

    @property
    def jitter_buffer_ms(self) -> int:
        return self._jitter_buffer_ms

    @jitter_buffer_ms.setter
    def jitter_buffer_ms(self, value: int) -> None:
        self._jitter_buffer_ms = value
        if self._jitter is not None:
            self._jitter.target_ms = value

    @property
    def jitter_buffer_min_ms(self) -> int:
        return self._jitter_buffer_min_ms

    @jitter_buffer_min_ms.setter
    def jitter_buffer_min_ms(self, value: int) -> None:
        self._jitter_buffer_min_ms = value
        if self._jitter is not None:
            self._jitter.min_ms = value

    @property
    def jitter_buffer_max_ms(self) -> int:
        return self._jitter_buffer_max_ms

    @jitter_buffer_max_ms.setter
    def jitter_buffer_max_ms(self, value: int) -> None:
        self._jitter_buffer_max_ms = value
        if self._jitter is not None:
            self._jitter.max_ms = value

    @property
    def audio_processor(self):
        return self._audio_processor

    @audio_processor.setter
    def audio_processor(self, value) -> None:
        self._audio_processor = value
        if self._capture is not None:
            self._capture.processor = value
        if self._send_track is not None:
            self._send_track.processor = value

    @property
    def chat_manager(self) -> ChatManager | None:
        return self._chat

    @chat_manager.setter
    def chat_manager(self, value: ChatManager | None) -> None:
        self._chat = value

    async def create_room(
        self,
        room_id: str,
        session_key: str | None = None,
        is_public: bool = False,
        title: str | None = None,
    ) -> None:
        key = session_key or secrets.token_hex(16)
        self._session_key = key
        result = await self._room_manager.create_room(
            room_id=room_id,
            session_key=key,
            is_public=is_public,
            title=title,
        )
        if result.get("success"):
            self._room_id = room_id
        else:
            raise RuntimeError(result.get("error", "Failed to create room"))

    async def start_streaming(self) -> None:
        if self._state != "idle" or not self._room_id:
            raise RuntimeError("Must create a room before streaming")

        setup_ice()

        self._capture = AudioCapture(
            device=self._input_device,
            sample_rate=self._sample_rate,
            block_size=self._block_size,
            latency=self._latency,
        )
        self._capture.processor = self._audio_processor

        self._send_track = AudioSendTrack(self._capture, webrtc_rate=self._sample_rate)
        self._send_track.processor = self._audio_processor
        self._send_track.start()

        self._webrtc = WebRTCManager()
        self._webrtc.set_ice_servers(self._stun_servers, self._turn_servers)
        self._webrtc.setup_streamer(self._send_track)

        self._webrtc.on_connection_state(self._on_webrtc_state)
        self._webrtc.on_error(self._on_webrtc_error)

        self._set_state("streaming")

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._signal_poll_task = asyncio.create_task(self._streamer_poll_loop())

        if self._chat:
            self._chat.on_message(lambda r, m: self._handle_chat_message(r, m))
            await self._chat.start(self._room_id)

        logger.info(f"Streaming started in room: {self._room_id}")

    async def start_listening(self, room_id: str) -> None:
        if self._state != "idle":
            raise RuntimeError("Session is not idle")

        setup_ice()

        join = await self._signaling.listener_join(room_id)
        if not join.get("success"):
            raise RuntimeError(f"Failed to join room: {join.get('error')}")

        self._room_id = room_id

        self._webrtc = WebRTCManager()
        self._webrtc.set_ice_servers(self._stun_servers, self._turn_servers)
        self._webrtc.setup_listener()
        self._webrtc.on_connection_state(self._on_webrtc_state)
        self._webrtc.on_track(self._on_remote_track)
        self._webrtc.on_error(self._on_webrtc_error)

        self._jitter = JitterBuffer(
            target_ms=self._jitter_buffer_ms,
            min_ms=self._jitter_buffer_min_ms,
            max_ms=self._jitter_buffer_max_ms,
            sample_rate=self._sample_rate,
        )
        self._jitter.enabled = self._jitter_buffer_enabled

        self._player = AudioPlayer(
            device=self._output_device,
            sample_rate=self._sample_rate,
            block_size=self._block_size,
            volume=self._volume,
            latency=self._latency,
        )
        self._player.start()

        self._adaptive_buffer = AdaptiveBufferController(self)
        self._adaptive_buffer_task = asyncio.create_task(self._adaptive_buffer_loop())

        self._set_state("listening")

        offer_sdp = await self._webrtc.create_offer()
        result = await self._signaling.send_signal(room_id, "offer", sdp=offer_sdp)
        since_id = result.get("signal_id", 0)

        answer_received = False
        async for sig in self._signaling.stream_signals(room_id, since_id=since_id):
            if sig["type"] == "answer":
                await self._webrtc.apply_answer(sig["sdp"])
                answer_received = True
                break

        if not answer_received:
            raise RuntimeError("No answer received from streamer")

        self._audio_play_task = asyncio.create_task(self._audio_play_loop())

        if self._chat:
            self._chat.on_message(lambda r, m: self._handle_chat_message(r, m))
            await self._chat.start(room_id)

        logger.info(f"Listening started in room: {room_id}")

    async def list_rooms(self) -> list[dict]:
        return await self._room_manager.list_rooms()

    async def send_chat(self, message: str) -> None:
        if self._chat is None:
            raise RuntimeError("Chat not configured")
        await self._chat.send(message)

    async def stop(self) -> None:
        if self._state == "idle":
            return

        was_streamer = self._state == "streaming"
        was_listener = self._state == "listening"

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._signal_poll_task is not None:
            self._signal_poll_task.cancel()
            try:
                await self._signal_poll_task
            except asyncio.CancelledError:
                pass
            self._signal_poll_task = None

        if self._audio_play_task is not None:
            self._audio_play_task.cancel()
            try:
                await self._audio_play_task
            except asyncio.CancelledError:
                pass
            self._audio_play_task = None

        if self._adaptive_buffer_task is not None:
            self._adaptive_buffer_task.cancel()
            try:
                await self._adaptive_buffer_task
            except asyncio.CancelledError:
                pass
            self._adaptive_buffer_task = None

        if self._chat is not None:
            await self._chat.stop()

        if self._send_track is not None:
            self._send_track.stop()
            self._send_track = None

        if self._webrtc is not None:
            await self._webrtc.close()
            self._webrtc = None

        if self._capture is not None:
            self._capture.close()
            self._capture = None

        if self._player is not None:
            self._player.close()
            self._player = None

        self._jitter = None

        if was_streamer and self._room_id:
            try:
                await self._signaling.delete_room(self._room_id)
            except Exception as e:
                logger.debug(f"Room deletion skipped: {e}")

        if was_listener and self._room_id:
            try:
                await self._signaling.listener_leave(self._room_id)
            except Exception as e:
                logger.debug(f"Listener leave skipped: {e}")

        self._set_state("idle")
        logger.info("Session stopped")

    async def close(self) -> None:
        await self.stop()
        await self._signaling.close()

    async def _heartbeat_loop(self) -> None:
        while self._state == "streaming" and self._room_id:
            try:
                await self._signaling.heartbeat(self._room_id)
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _streamer_poll_loop(self) -> None:
        while self._state == "streaming" and self._room_id:
            try:
                async for sig in self._signaling.stream_signals(self._room_id):
                    if sig["type"] == "offer":
                        if self._webrtc is None:
                            continue
                        state = self._webrtc.connection_state
                        if state in ("connected", "connecting"):
                            logger.debug(f"Ignoring offer (state={state})")
                            continue
                        if state in ("closed", "failed"):
                            logger.info("Peer gone, creating new connection")
                            if self._capture is None:
                                logger.error("Capture device not available")
                                continue
                            capture = self._capture
                            self._send_track = AudioSendTrack(
                                capture,
                                webrtc_rate=self._sample_rate,
                            )
                            self._send_track.start()
                            self._webrtc = WebRTCManager()
                            self._webrtc.set_ice_servers(
                                self._stun_servers, self._turn_servers
                            )
                            self._webrtc.setup_streamer(self._send_track)
                            self._webrtc.on_connection_state(self._on_webrtc_state)
                            self._webrtc.on_error(self._on_webrtc_error)

                        answer_sdp = await self._webrtc.handle_offer(sig["sdp"])
                        await self._signaling.send_signal(
                            self._room_id, "answer", sdp=answer_sdp
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Signal poll error: {e}")
                await asyncio.sleep(1)

    async def _audio_play_loop(self) -> None:
        while self._state == "listening":
            await asyncio.sleep(0.01)
            if self._jitter is None or self._player is None:
                continue
            frame = self._jitter.read()
            if frame is not None:
                self._player.write(frame)

    async def _adaptive_buffer_loop(self) -> None:
        while self._state == "listening":
            await asyncio.sleep(1)
            if self._jitter is None or self._adaptive_buffer is None:
                continue
            self._adaptive_buffer.update_metrics(self._jitter.current_ms)

    def _on_webrtc_state(self, state: str) -> None:
        connected = state == "connected"
        if self.on_connection_change:
            self.on_connection_change(connected)
        logger.info(f"WebRTC state: {state}")

    def _on_webrtc_error(self, error: Exception) -> None:
        logger.error(f"WebRTC error: {error}")
        if self.on_error:
            self.on_error(error)

    def _on_remote_track(self, track) -> None:
        if track.kind == "audio":
            logger.info("Remote audio track received")
            asyncio.create_task(self._handle_remote_audio(track))

    async def _handle_remote_audio(self, track) -> None:
        while self._state == "listening":
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                import numpy as np

                raw = frame.to_ndarray().astype(np.float32) / 32768.0
                ns = frame.samples
                if raw.ndim == 2 and raw.shape[0] == 1 and raw.shape[1] == ns * 2:
                    audio = raw.reshape(ns, 2).mean(axis=1)
                elif raw.ndim > 1 and raw.shape[0] > 1:
                    audio = raw.mean(axis=0)
                else:
                    audio = raw[:ns].flatten()
                if self._jitter is not None:
                    self._jitter.write(audio)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if not isinstance(e, asyncio.CancelledError):
                    logger.error(f"Audio receive error: {e}")
                break

    def _handle_chat_message(self, room_id: str, message: str) -> None:
        logger.info(f"Chat from {room_id}: {message}")
        if self.on_chat_message:
            self.on_chat_message(room_id, message)

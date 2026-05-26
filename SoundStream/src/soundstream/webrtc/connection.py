from __future__ import annotations

import asyncio
import logging
from typing import Callable, Any

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

logger = logging.getLogger('soundstream')


class WebRTCManager:

    def __init__(self) -> None:
        self._pc: RTCPeerConnection | None = None
        self._config: RTCConfiguration | None = None

        self._on_connection_state: Callable[[str], None] | None = None
        self._on_track: Callable[[Any], None] | None = None
        self._on_error: Callable[[Exception], None] | None = None

    def set_ice_servers(self, stun_servers: list[str], turn_servers: list[dict] | None = None) -> None:
        ice_servers = []
        for url in stun_servers:
            ice_servers.append(RTCIceServer(urls=url))
        if turn_servers:
            for turn in turn_servers:
                ice_servers.append(RTCIceServer(
                    urls=turn.get('urls', []),
                    username=turn.get('username'),
                    credential=turn.get('credential'),
                ))
        self._config = RTCConfiguration(iceServers=ice_servers)

    def _create_pc(self) -> RTCPeerConnection:
        if self._config is None:
            self._config = RTCConfiguration(
                iceServers=[RTCIceServer(urls='stun:stun.l.google.com:19302')],
            )
        pc = RTCPeerConnection(configuration=self._config)

        @pc.on('connectionstatechange')
        async def _on_state():
            state = pc.connectionState
            logger.debug(f'WebRTC connection state: {state}')
            if self._on_connection_state:
                self._on_connection_state(state)
            if state == 'failed' and self._on_error:
                self._on_error(Exception('WebRTC connection failed'))

        @pc.on('track')
        def _on_track(event):
            logger.debug(f'WebRTC track received: {event.kind}')
            if self._on_track:
                self._on_track(event)

        return pc

    @property
    def peer_connection(self) -> RTCPeerConnection | None:
        return self._pc

    @property
    def connection_state(self) -> str | None:
        if self._pc is None:
            return None
        return self._pc.connectionState

    @property
    def is_connected(self) -> bool:
        return self._pc is not None and self._pc.connectionState == 'connected'

    def setup_streamer(self, track: Any) -> None:
        self._pc = self._create_pc()
        self._pc.addTrack(track)
        logger.info('WebRTCManager: configured as streamer')

    def setup_listener(self) -> None:
        self._pc = self._create_pc()
        self._pc.addTransceiver('audio', direction='recvonly')
        logger.info('WebRTCManager: configured as listener')

    async def create_offer(self) -> str:
        if self._pc is None:
            raise RuntimeError('Peer connection not initialized')
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        logger.debug('WebRTC offer created')
        return self._pc.localDescription.sdp

    async def handle_offer(self, sdp: str) -> str:
        if self._pc is None:
            raise RuntimeError('Peer connection not initialized')
        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type='offer'),
        )
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        logger.debug('WebRTC answer created')
        return self._pc.localDescription.sdp

    async def apply_answer(self, sdp: str) -> None:
        if self._pc is None:
            raise RuntimeError('Peer connection not initialized')
        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type='answer'),
        )
        logger.debug('WebRTC answer applied')

    async def add_ice_candidate(self, candidate: str) -> None:
        if self._pc is None:
            return
        # aiortc handles ICE candidates automatically from SDP

    async def close(self) -> None:
        if self._pc is not None:
            await self._pc.close()
            self._pc = None
            logger.info('WebRTCManager closed')

    def on_connection_state(self, callback: Callable[[str], None]) -> None:
        self._on_connection_state = callback

    def on_track(self, callback: Callable[[Any], None]) -> None:
        self._on_track = callback

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        self._on_error = callback
from __future__ import annotations

import asyncio
import logging
import time
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from av import AudioFrame
from aiortc import AudioStreamTrack, MediaStreamError

if TYPE_CHECKING:
    from soundstream.audio import AudioCapture, AudioProcessor

logger = logging.getLogger('soundstream')


class AudioSendTrack(AudioStreamTrack):

    kind = 'audio'

    def __init__(
        self,
        capture: 'AudioCapture',
        webrtc_rate: int = 48000,
    ) -> None:
        super().__init__()
        self._capture = capture
        self._webrtc_rate = webrtc_rate
        self._frame_samples = webrtc_rate // 100

        self._processor: 'AudioProcessor | None' = None

        self._timestamp = 0
        self._start = 0.0
        self._count = 0

    @property
    def processor(self) -> 'AudioProcessor | None':
        return self._processor

    @processor.setter
    def processor(self, value: 'AudioProcessor | None') -> None:
        self._processor = value

    def start(self) -> None:
        self._capture.start()

    async def recv(self) -> AudioFrame:
        if self.readyState != 'live':
            raise MediaStreamError()

        n = self._frame_samples

        if hasattr(self, '_timestamp'):
            self._timestamp += n
            wait = self._start + (self._timestamp / self._webrtc_rate) - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.monotonic()
            self._timestamp = 0

        samples = self._capture.read(n)

        if self._processor is not None:
            samples = self._processor.process(samples)

        pcm = (samples * 32767).astype(np.int16)
        stereo = np.empty(2 * n, dtype=np.int16)
        stereo[0::2] = pcm
        stereo[1::2] = pcm

        frame = AudioFrame.from_ndarray(
            stereo.reshape(1, -1), format='s16', layout='stereo',
        )
        frame.sample_rate = self._webrtc_rate
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, self._webrtc_rate)

        self._count += 1
        if self._count <= 5:
            rms = float(np.sqrt(np.mean(samples ** 2)))
            logger.debug(
                f'AudioSendTrack #{self._count}: samples={len(samples)} '
                f'RMS={rms:.4f} frame.samples={frame.samples}',
            )
        return frame

    def stop(self) -> None:
        super().stop()
        self._capture.stop()
from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger('soundstream')


class AudioPlayer:

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = 48000,
        block_size: int = 480,
        volume: float = 1.0,
        latency: str = 'low',
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._volume = volume
        self._latency = latency
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()

    @property
    def device(self) -> int | None:
        return self._device

    @device.setter
    def device(self, value: int | None) -> None:
        if value != self._device:
            self._device = value
            if self._stream is not None:
                self.stop()
                self.start()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        if value != self._sample_rate:
            self._sample_rate = value
            if self._stream is not None:
                self.stop()
                self.start()

    @property
    def block_size(self) -> int:
        return self._block_size

    @block_size.setter
    def block_size(self, value: int) -> None:
        if value != self._block_size:
            self._block_size = value
            if self._stream is not None:
                self.stop()
                self.start()

    @property
    def latency(self) -> str:
        return self._latency

    @latency.setter
    def latency(self, value: str) -> None:
        if value != self._latency:
            self._latency = value
            if self._stream is not None:
                self.stop()
                self.start()

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = max(0.0, min(10.0, value))

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active

    def start(self) -> None:
        with self._lock:
            if self._stream is not None and self._stream.active:
                return

            try:
                device = self._device if self._device is not None else sd.default.device[1]
                dev_info = sd.query_devices(device, kind='output')
            except sd.PortAudioError:
                devices = sd.query_devices()
                for i, d in enumerate(devices):
                    if d['max_output_channels'] > 0:
                        device = i
                        dev_info = d
                        break
                else:
                    raise RuntimeError('No output device available')

            stream_kwargs = {
                'samplerate': self._sample_rate,
                'channels': 1,
                'blocksize': self._block_size,
                'dtype': 'float32',
                'latency': self._latency,
            }
            if self._device is not None:
                stream_kwargs['device'] = self._device

            self._stream = sd.OutputStream(**stream_kwargs)
            self._stream.start()

            logger.info(
                f'AudioPlayer: "{dev_info["name"]}" '
                f'rate={self._sample_rate}Hz block={self._block_size}',
            )

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

    def write(self, samples: np.ndarray) -> None:
        if self._stream is None or not self._stream.active:
            raise RuntimeError('Player not active')

        if self._volume != 1.0:
            samples = samples * self._volume

        self._stream.write(samples.reshape(-1, 1))

    def close(self) -> None:
        self.stop()

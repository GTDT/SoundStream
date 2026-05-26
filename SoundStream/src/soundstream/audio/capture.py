from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger('soundstream')


class AudioCapture:

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = 48000,
        block_size: int = 480,
        latency: str = 'low',
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._latency = latency
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._resample_ratio = 1.0

        self._capture_rate: int = 0
        self._capture_samples: int = 0
        self._needs_resample = False

        self._processor = None

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
    def processor(self):
        return self._processor

    @processor.setter
    def processor(self, value) -> None:
        self._processor = value

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active

    def start(self) -> None:
        with self._lock:
            if self._stream is not None and self._stream.active:
                return

            try:
                device = self._device if self._device is not None else sd.default.device[0]
                dev_info = sd.query_devices(device, kind='input')
            except sd.PortAudioError:
                devices = sd.query_devices()
                for i, d in enumerate(devices):
                    if d['max_input_channels'] > 0:
                        device = i
                        dev_info = d
                        break
                else:
                    raise RuntimeError('No input device available')

            self._capture_rate = int(dev_info['default_samplerate'])

            self._capture_samples = round(
                self._block_size * self._capture_rate / self._sample_rate,
            )
            self._needs_resample = self._capture_rate != self._sample_rate
            self._resample_ratio = self._sample_rate / self._capture_rate

            stream_kwargs = {
                'samplerate': self._capture_rate,
                'channels': 1,
                'blocksize': self._capture_samples,
                'dtype': 'float32',
                'latency': self._latency,
            }
            if self._device is not None:
                stream_kwargs['device'] = self._device

            self._stream = sd.InputStream(**stream_kwargs)
            self._stream.start()

            actual = self._stream.samplerate
            if actual != self._capture_rate:
                self._capture_rate = actual
                self._capture_samples = round(
                    self._block_size * self._capture_rate / self._sample_rate,
                )
                self._needs_resample = self._capture_rate != self._sample_rate
                self._resample_ratio = self._sample_rate / self._capture_rate

            logger.info(
                f'MicCapture: "{dev_info["name"]}" '
                f'capture={self._capture_rate}Hz target={self._sample_rate}Hz '
                f'block={self._capture_samples} resample={self._needs_resample}',
            )

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

    def read(self, frames: int) -> np.ndarray:
        if self._stream is None or not self._stream.active:
            raise RuntimeError('Capture not active')

        data, _ = self._stream.read(self._capture_samples)
        samples = data.flatten()

        if self._needs_resample:
            samples = self._resample(samples, frames)

        if self._processor is not None:
            samples = self._processor.process(samples)

        return samples

    def _resample(self, samples: np.ndarray, target_frames: int) -> np.ndarray:
        if not self._needs_resample:
            return samples

        x_old = np.linspace(0, 1, len(samples))
        x_new = np.linspace(0, 1, target_frames)
        return np.interp(x_new, x_old, samples).astype(np.float32)

    def close(self) -> None:
        self.stop()

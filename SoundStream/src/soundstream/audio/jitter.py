from __future__ import annotations

import collections
import logging
import time
from typing import Callable

import numpy as np

logger = logging.getLogger('soundstream')


class JitterBuffer:

    def __init__(
        self,
        target_ms: int = 60,
        min_ms: int = 20,
        max_ms: int = 200,
        sample_rate: int = 48000,
    ) -> None:
        self._target_ms = target_ms
        self._min_ms = min_ms
        self._max_ms = max_ms
        self._sample_rate = sample_rate

        self._buffer: collections.deque[np.ndarray] = collections.deque()
        self._target_frames = (sample_rate * target_ms) // 1000
        self._min_frames = (sample_rate * min_ms) // 1000
        self._max_frames = (sample_rate * max_ms) // 1000

        self._last_arrival_time: float | None = None
        self._late_count = 0
        self._on_underrun: Callable[[], None] | None = None
        self._enabled = True

    @property
    def target_ms(self) -> int:
        return self._target_ms

    @target_ms.setter
    def target_ms(self, value: int) -> None:
        self._target_ms = value
        self._target_frames = (self._sample_rate * value) // 1000

    @property
    def min_ms(self) -> int:
        return self._min_ms

    @min_ms.setter
    def min_ms(self, value: int) -> None:
        self._min_ms = value
        self._min_frames = (self._sample_rate * value) // 1000

    @property
    def max_ms(self) -> int:
        return self._max_ms

    @max_ms.setter
    def max_ms(self, value: int) -> None:
        self._max_ms = value
        self._max_frames = (self._sample_rate * value) // 1000

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def on_underrun(self) -> Callable[[], None] | None:
        return self._on_underrun

    @on_underrun.setter
    def on_underrun(self, value: Callable[[], None]) -> None:
        self._on_underrun = value

    @property
    def buffer_depth(self) -> int:
        return len(self._buffer)

    @property
    def current_ms(self) -> int:
        if not self._buffer:
            return 0
        total_frames = sum(len(frame) for frame in self._buffer)
        return (total_frames * 1000) // self._sample_rate

    def write(self, frame: np.ndarray) -> None:
        if not self._enabled:
            return

        now = time.monotonic()
        if self._last_arrival_time is not None:
            interval = now - self._last_arrival_time
            expected_interval = (len(frame) / self._sample_rate) * 0.9
            if interval > expected_interval:
                self._late_count += 1

                if self._late_count > 3 and self._target_ms < self._max_ms:
                    new_target = min(self._target_ms + 20, self._max_ms)
                    logger.debug(f'JitterBuffer: increasing target {self._target_ms} -> {new_target}ms')
                    self.target_ms = new_target
                    self._late_count = 0
        else:
            self._late_count = 0

        self._last_arrival_time = now
        self._buffer.append(frame)

        if len(self._buffer) > self._max_frames * 2:
            self._buffer.popleft()

    def read(self) -> np.ndarray | None:
        if not self._enabled:
            return None

        if len(self._buffer) == 0:
            if self._on_underrun is not None:
                self._on_underrun()
            return None

        if len(self._buffer) < self._target_frames // (self._sample_rate // 100):
            if self._on_underrun is not None:
                self._on_underrun()
            return None

        frame = self._buffer.popleft()

        if self._late_count == 0 and self._target_ms > self._min_ms:
            new_target = max(self._target_ms - 5, self._min_ms)
            if new_target != self._target_ms:
                logger.debug(f'JitterBuffer: decreasing target {self._target_ms} -> {new_target}ms')
                self.target_ms = new_target

        return frame

    def reset(self) -> None:
        self._buffer.clear()
        self._last_arrival_time = None
        self._late_count = 0

from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import SoundSession

logger = logging.getLogger("soundstream")


class AudioProcessor:
    """Bazinė klasė visiems garso procesoriams."""

    def process(self, data: np.ndarray) -> np.ndarray:
        return data


class EffectChain(AudioProcessor):
    """Leidžia sujungti kelis efektus į vieną grandinę."""

    def __init__(self, processors: list[AudioProcessor]):
        self._processors = processors

    def process(self, data: np.ndarray) -> np.ndarray:
        for processor in self._processors:
            data = processor.process(data)
        return data


class OpusCompressor(AudioProcessor):
    """
    Paruošia garsą Opus formatui.
    Pastaba: aiortc automatiškai naudoja Opus, tačiau ši klasė užtikrina
    teisingą signalo normalizavimą prieš kodavimą.
    """

    def __init__(self, bitrate: int = 64000):
        self.bitrate = bitrate
        logger.debug(f"Opus compressor initialized with {bitrate} bps")

    def process(self, data: np.ndarray) -> np.ndarray:
        # Užtikriname, kad signalas neviršija ribų (clipping prevention)
        return np.clip(data, -1.0, 1.0)


class NoiseSuppressor(AudioProcessor):
    """Triukšmo slopinimas su sklandžiu perėjimu (Attack / Release)."""

    def __init__(self, threshold: float = 0.015, min_gain: float = 0.05):
        self.threshold = threshold
        self.min_gain = min_gain
        self._current_gain = 1.0

    def process(self, data: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(data**2)) + 1e-6

        # Nustatome, koks turėtų būti garsumas
        if rms < self.threshold:
            target_gain = self.min_gain  # Tyla (ventiliatoriaus nutildymas)
        else:
            target_gain = 1.0  # Kalbėjimas (pilnas garsas)

        # Sklandus perėjimas (Smoothing / Attack & Release)
        if target_gain > self._current_gain:
            # Greitai atidarome vartus, kai pradedi kalbėti (Attack)
            self._current_gain = min(target_gain, self._current_gain + 0.15)
        else:
            # Lėtai uždarome vartus, kai nustoji kalbėti (Release)
            # Tai užkerta kelią traškėjimui
            self._current_gain = max(target_gain, self._current_gain - 0.015)

        return data * self._current_gain


class SpectralNoiseReducer(AudioProcessor):
    """
    Pašalina pastovius triukšmus su dažnių glotninimu laike,
    kad išvengtume "povandeninio" (musical noise) efekto.
    """

    def __init__(self, noise_floor_len=20):
        self.noise_floor_len = noise_floor_len
        self.noise_profile = None
        self._buffers = []
        self.smooth_gain = None  # Išsaugo praeito bloko filtrą

    def process(self, data: np.ndarray) -> np.ndarray:
        fft_data = np.fft.rfft(data)
        magnitude = (
            np.abs(fft_data) + 1e-9
        )  # Pridedame mažą skaičių, kad išvengtume dalybos iš nulio
        phase = np.angle(fft_data)

        # 1. Kaupiame triukšmo profilį (pirmas 0.5-1 sek.)
        if self.noise_profile is None:
            self._buffers.append(magnitude)
            if len(self._buffers) >= self.noise_floor_len:
                self.noise_profile = np.mean(self._buffers, axis=0)
            return data

        # 2. Skaičiuojame, kiek reikia slopinti kiekvieną dažnį
        # (Jei signalas artimas triukšmui, slopiname. Jei garsus balsas - paliekame)
        raw_gain = (magnitude - self.noise_profile * 1.2) / magnitude

        # Neleidžiame stiprinti garso arba nutildyti jo visiškai (paliekame 10% "grindis")
        raw_gain = np.clip(raw_gain, 0.1, 1.0)

        # 3. GLOTNINIMAS LAIKE (Magiška eilutė prieš burbuliavimą)
        if self.smooth_gain is None:
            self.smooth_gain = raw_gain
        else:
            # Maišome 70% seno bloko filtro ir 30% naujo.
            # Tai sušvelnina staigius šuolius ir panaikina vandens efektą.
            self.smooth_gain = 0.7 * self.smooth_gain + 0.3 * raw_gain

        # 4. Pritaikome filtrą ir grįžtame į laiko sritį
        new_magnitude = magnitude * self.smooth_gain
        reconstructed = new_magnitude * np.exp(1j * phase)

        return np.fft.irfft(reconstructed).astype(np.float32)


class AutoGainControl(AudioProcessor):
    """
    Automatinis garsumo adaptavimas (AGC) su integruotu Limiteriu,
    kuris apsaugo nuo garso perdegimo (peakinimo/clipping).
    """

    def __init__(self, target_rms: float = 0.12, max_gain: float = 3.0):
        self.target_rms = target_rms
        self.max_gain = max_gain
        self._current_gain = 1.0

    def process(self, data: np.ndarray) -> np.ndarray:
        # Apskaičiuojame esamą signalo stiprumą
        current_rms = np.sqrt(np.mean(data**2)) + 1e-6

        # Jei signalas visiškai tylus, nedidiname garsumo (kad nekeltume likusio triukšmo)
        if current_rms < 0.005:
            return data

        gain_needed = self.target_rms / current_rms

        # Ribojame maksimalų stiprinimą (kad per daug neužkeltų ramaus kalbėjimo)
        gain_needed = min(self.max_gain, gain_needed)

        # Švelnus perėjimas (Smoothing):
        # Jei reikia garsinti - garsiname lėtai, jei reikia tylinti (nes peakina) - nutildome ŽAIBIŠKAI
        if gain_needed < self._current_gain:
            self._current_gain = (
                0.5 * self._current_gain + 0.5 * gain_needed
            )  # Greitas atsakas į garsų šauksmą
        else:
            self._current_gain = (
                0.95 * self._current_gain + 0.05 * gain_needed
            )  # Lėtas garso kėlimas tyliuose momentuose

        # Pritaikome sušvelnintą stiprinimą
        processed_data = data * self._current_gain

        # --- LIMITERIS (Apsauga nuo peakinimo) ---
        # Jei net ir po visko kurios nors bangos viršūnė viršija saugią 0.95 ribą,
        # mes ją ne aštriai nukertame, o sušvelniname naudojant minkštą ribojimą (soft-clipping)
        max_val = np.max(np.abs(processed_data))
        if max_val > 0.95:
            # Minkštas limitavimas naudojant tanh (hiperbolinį tangentą)
            # Tai suapvalina bangų viršūnes, todėl nelieka skaitmeninio traškėjimo
            processed_data = np.tanh(processed_data / max_val) * 0.95

        return processed_data


class AdaptiveBufferController:
    """
    Stebi vėlavimą ir dinamiškai reguliuoja sesijos buferio dydį.
    """

    def __init__(self, session: SoundSession):
        self._session = session
        self._history = []

    def update_metrics(self, current_latency_ms: float):
        self._history.append(current_latency_ms)
        if len(self._history) > 50:
            self._history.pop(0)

            avg_latency = sum(self._history) / len(self._history)

            # Jei vėlavimas didelis, didiname buferį stabilumui
            if avg_latency > self._session.jitter_buffer_max_ms * 0.8:
                new_size = min(
                    self._session.jitter_buffer_max_ms,
                    self._session.jitter_buffer_ms + 10,
                )
                if new_size != self._session.jitter_buffer_ms:
                    self._session.jitter_buffer_ms = new_size
                    logger.debug(f"Adaptive Buffer: increased to {new_size}ms")

            # Jei ryšys labai stabilus, mažiname buferį mažesniam vėlavimui
            elif avg_latency < self._session.jitter_buffer_ms * 0.4:
                new_size = max(
                    self._session.jitter_buffer_min_ms,
                    self._session.jitter_buffer_ms - 5,
                )
                if new_size != self._session.jitter_buffer_ms:
                    self._session.jitter_buffer_ms = new_size
                    logger.debug(f"Adaptive Buffer: decreased to {new_size}ms")

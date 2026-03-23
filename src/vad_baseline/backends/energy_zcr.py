import wave
from array import array

import torch

from vad_baseline.backends.common import BaseVADBackend


def _load_audio_mono(audio_path):
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), int(sample_rate)
    except Exception:
        with wave.open(str(audio_path), "rb") as handle:
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            if sample_width != 2:
                raise ValueError("wave fallback only supports 16-bit PCM input")

            num_channels = handle.getnchannels()
            raw_samples = array("h")
            raw_samples.frombytes(handle.readframes(handle.getnframes()))
            waveform = torch.tensor(raw_samples, dtype=torch.float32) / 32768.0
            if num_channels > 1:
                waveform = waveform.view(-1, num_channels).mean(dim=1)
            return waveform, int(sample_rate)


def _resample_if_needed(waveform, sample_rate, target_sample_rate):
    if sample_rate == target_sample_rate:
        return waveform

    import torchaudio.functional

    return torchaudio.functional.resample(
        waveform.unsqueeze(0),
        sample_rate,
        target_sample_rate,
    ).squeeze(0)


def _frame_signal(waveform, frame_size):
    if waveform.numel() < frame_size:
        return waveform.new_zeros((0, frame_size))

    usable_samples = (waveform.numel() // frame_size) * frame_size
    framed = waveform[:usable_samples].view(-1, frame_size)
    return framed


def _fill_short_gaps(decisions, max_gap_frames):
    if max_gap_frames <= 0:
        return decisions[:]

    smoothed = decisions[:]
    gap_start = None
    for frame_index, is_speech in enumerate(smoothed):
        if is_speech:
            if gap_start is not None:
                gap_length = frame_index - gap_start
                if gap_start > 0 and gap_length <= max_gap_frames:
                    for gap_index in range(gap_start, frame_index):
                        smoothed[gap_index] = True
                gap_start = None
            continue

        if gap_start is None:
            gap_start = frame_index

    return smoothed


def _apply_hangover(decisions, hangover_frames):
    if hangover_frames <= 0:
        return decisions[:]

    extended = decisions[:]
    for frame_index, is_speech in enumerate(decisions):
        if not is_speech:
            continue
        for offset in range(1, hangover_frames + 1):
            target_index = frame_index + offset
            if target_index >= len(extended):
                break
            extended[target_index] = True
    return extended


def _frames_to_segments(decisions, frame_sec, total_duration_sec, min_speech_frames):
    segments = []
    start_index = None
    for frame_index, is_speech in enumerate(decisions):
        if is_speech and start_index is None:
            start_index = frame_index
            continue

        if is_speech or start_index is None:
            continue

        if frame_index - start_index >= min_speech_frames:
            start_sec = start_index * frame_sec
            end_sec = min(total_duration_sec, frame_index * frame_sec)
            segments.append(
                {
                    "start": round(float(start_sec), 6),
                    "end": round(float(end_sec), 6),
                    "duration": round(float(end_sec - start_sec), 6),
                }
            )
        start_index = None

    if start_index is not None and len(decisions) - start_index >= min_speech_frames:
        start_sec = start_index * frame_sec
        end_sec = min(total_duration_sec, len(decisions) * frame_sec)
        segments.append(
            {
                "start": round(float(start_sec), 6),
                "end": round(float(end_sec), 6),
                "duration": round(float(end_sec - start_sec), 6),
            }
        )

    return segments


class EnergyZCRBackend(BaseVADBackend):
    backend_name = "energy_zcr"
    model_name = "classical/energy_zcr"

    def __init__(
        self,
        sample_rate=16000,
        frame_ms=20,
        energy_ratio=0.5,
        zcr_threshold=0.2,
        min_speech_sec=0.08,
        min_silence_sec=0.08,
        hangover_frames=1,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.energy_ratio = energy_ratio
        self.zcr_threshold = zcr_threshold
        self.min_speech_sec = min_speech_sec
        self.min_silence_sec = min_silence_sec
        self.hangover_frames = hangover_frames

    def load(self):
        return None

    def predict_segments(self, backend_model, audio_path):
        waveform, sample_rate = _load_audio_mono(audio_path)
        waveform = _resample_if_needed(
            waveform,
            sample_rate,
            self.sample_rate,
        )
        waveform = waveform.float()
        frame_size = int(self.sample_rate * (self.frame_ms / 1000.0))
        framed = _frame_signal(waveform, frame_size)
        if framed.shape[0] == 0:
            return []

        frame_energy = framed.pow(2).mean(dim=1)
        energy_threshold = max(
            1e-6,
            float(frame_energy.max().item()) * self.energy_ratio,
        )
        sign_changes = torch.diff(torch.signbit(framed), dim=1).float()
        zcr = sign_changes.mean(dim=1)
        decisions = [
            bool(energy.item() >= energy_threshold and zcr_value.item() <= self.zcr_threshold)
            for energy, zcr_value in zip(frame_energy, zcr)
        ]

        frame_sec = self.frame_ms / 1000.0
        decisions = _fill_short_gaps(
            decisions,
            int(round(self.min_silence_sec / frame_sec)),
        )
        decisions = _apply_hangover(decisions, self.hangover_frames)

        return _frames_to_segments(
            decisions,
            frame_sec,
            waveform.numel() / self.sample_rate,
            max(1, int(round(self.min_speech_sec / frame_sec))),
        )

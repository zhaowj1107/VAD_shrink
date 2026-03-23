import torch

from vad_baseline.backends.common import BaseVADBackend
from vad_baseline.backends.energy_zcr import (
    _apply_hangover,
    _fill_short_gaps,
    _frames_to_segments,
    _load_audio_mono,
    _resample_if_needed,
)


class WebRTCVADBackend(BaseVADBackend):
    backend_name = "webrtc_vad"
    model_name = "classical/webrtc_vad"

    def __init__(
        self,
        sample_rate=16000,
        frame_ms=20,
        aggressiveness=2,
        min_speech_sec=0.08,
        min_silence_sec=0.08,
        hangover_frames=0,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.aggressiveness = aggressiveness
        self.min_speech_sec = min_speech_sec
        self.min_silence_sec = min_silence_sec
        self.hangover_frames = hangover_frames

    def load(self):
        try:
            import webrtcvad
        except ImportError as error:
            raise ImportError(
                "webrtcvad is required for the webrtc_vad backend"
            ) from error

        return webrtcvad.Vad(self.aggressiveness)

    def predict_segments(self, backend_model, audio_path):
        waveform, sample_rate = _load_audio_mono(audio_path)
        waveform = _resample_if_needed(
            waveform,
            sample_rate,
            self.sample_rate,
        )
        waveform = waveform.float().clamp(-1.0, 1.0)
        pcm = (waveform * 32767.0).to(torch.int16)
        frame_size = int(self.sample_rate * (self.frame_ms / 1000.0))
        if frame_size <= 0:
            raise ValueError("frame size must be positive")

        num_frames = pcm.numel() // frame_size
        if num_frames == 0:
            return []

        pcm = pcm[: num_frames * frame_size]
        decisions = []
        for frame_index in range(num_frames):
            frame = pcm[
                frame_index * frame_size : (frame_index + 1) * frame_size
            ]
            decisions.append(
                bool(
                    backend_model.is_speech(
                        frame.contiguous().numpy().tobytes(),
                        self.sample_rate,
                    )
                )
            )

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

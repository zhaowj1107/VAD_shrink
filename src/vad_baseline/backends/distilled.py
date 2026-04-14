# src/vad_baseline/backends/distilled.py
"""Distilled Student VAD Backend."""

from pathlib import Path

import numpy as np
import torch
import torchaudio

from vad_baseline.backends.common import BaseVADBackend
from vad_baseline.distillation.student_model import SimplifiedCRDNN
from vad_baseline.inference import normalize_segments


class DistilledBackend(BaseVADBackend):
    """Backend for distilled Student VAD model."""

    backend_name = "distilled"
    model_name = "distilled-student"
    supports_frame_probabilities = True

    def __init__(self, checkpoint_path=None, device=None):
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load(self):
        """Load the distilled Student model."""
        if self.model is None:
            self.model = SimplifiedCRDNN()
            if self.checkpoint_path:
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                )
                self.model.load_state_dict(checkpoint["student_state"])
            self.model.to(self.device)
        self.model.eval()
        return self.model

    def predict_segments(self, backend_model, audio_path):
        """Predict speech segments for an audio file."""
        frame_probs = self._get_frame_probs(backend_model, audio_path)
        segments = self._probs_to_segments(frame_probs)
        return normalize_segments(segments)

    def predict_frame_probabilities(self, backend_model, audio_path):
        """Get frame-level probabilities."""
        frame_probs = self._get_frame_probs(backend_model, audio_path)
        from vad_baseline.inference import normalize_frame_probabilities
        return normalize_frame_probabilities(frame_probs)

    def _get_frame_probs(self, model, audio_path):
        """Get frame probabilities from audio."""
        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        audio = waveform.mean(dim=0).to(self.device)

        # Simple energy-based feature (placeholder)
        # In production, should use same feature extraction as training
        frame_length = 400  # 25ms at 16kHz
        hop_length = 160  # 10ms at 16kHz

        frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]
            energy = (frame**2).mean().sqrt()
            frames.append(energy.item())

        if not frames:
            return np.array([])

        frames = torch.tensor(frames).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(frames / frames.std()).cpu().numpy()

        return probs

    def _probs_to_segments(self, probs, threshold=0.5, min_duration_frames=10):
        """Convert frame probabilities to segments."""
        probs = np.array(probs)
        is_speech = probs > threshold

        segments = []
        start = None

        for i, speech in enumerate(is_speech):
            if speech and start is None:
                start = i
            elif not speech and start is not None:
                if i - start >= min_duration_frames:
                    segments.append({
                        "start": start * 0.01,
                        "end": i * 0.01,
                    })
                start = None

        if start is not None:
            if len(probs) - start >= min_duration_frames:
                segments.append({
                    "start": start * 0.01,
                    "end": len(probs) * 0.01,
                })

        return segments

    def summarize_model_tensors(self, backend_model):
        """Get model tensor summary."""
        from vad_baseline.backends.common import summarize_module_tensors
        return summarize_module_tensors(backend_model)
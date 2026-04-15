# src/vad_baseline/distillation/dataset.py
"""Dataset classes for VAD distillation."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm


# Mel filterbank parameters matching SpeechBrain VAD
FBANK_N_MELS = 80  # Standard mel filterbank bins
FBANK_N_FFT = 512
FBANK_HOP_LENGTH = 160  # 10ms at 16kHz
FBANK_WIN_LENGTH = 400  # 25ms at 16kHz


class FbankExtractor:
    """Fbank feature extractor using torchaudio transforms."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=FBANK_N_FFT,
            hop_length=FBANK_HOP_LENGTH,
            win_length=FBANK_WIN_LENGTH,
            n_mels=FBANK_N_MELS,
        )

    def __call__(self, audio):
        """
        Extract fbank features from audio.

        Args:
            audio: 1D tensor of audio samples

        Returns:
            Tensor of shape (time, n_mels)
        """
        mel_spec = self.mel_transform(audio)
        log_mel = torch.log(mel_spec + 1e-8)
        return log_mel.T  # (time, n_mels)


# Global extractor instance for efficiency
_fbank_extractor = None


def get_fbank_extractor():
    global _fbank_extractor
    if _fbank_extractor is None:
        _fbank_extractor = FbankExtractor()
    return _fbank_extractor


def extract_fbank_features(audio, sample_rate=16000):
    """
    Extract mel filterbank features from audio.

    Args:
        audio: 1D tensor of audio samples
        sample_rate: audio sample rate

    Returns:
        Tensor of shape (time, n_mels)
    """
    extractor = get_fbank_extractor()
    return extractor(audio)


class LibriPartyDistillationDataset:
    """
    Dataset for LibriParty distillation training.

    Loads audio, soft labels from Teacher, and hard labels from annotations.
    """

    def __init__(
        self,
        session_dirs: list[str],
        soft_labels_dir: str,
        feature_type: str = "fbank",
        frame_shift_sec: float = 0.01,
        sample_rate: int = 16000,
    ):
        self.session_dirs = [Path(d) for d in session_dirs]
        self.soft_labels_dir = Path(soft_labels_dir)
        self.feature_type = feature_type
        self.frame_shift_sec = frame_shift_sec
        self.sample_rate = sample_rate

        # Collect all sessions
        self.samples = []
        for session_dir in tqdm(self.session_dirs, desc="Loading dataset"):
            session_name = session_dir.name
            annotation_path = session_dir / f"{session_name}.json"
            soft_label_path = self.soft_labels_dir / f"{session_name}.npy"

            if not annotation_path.exists():
                print(f"Warning: No annotation {annotation_path}")
                continue

            if not soft_label_path.exists():
                print(f"Warning: No soft labels {soft_label_path}")
                continue

            # Find audio
            audio_files = list(session_dir.glob("*_mixture.wav"))
            if not audio_files:
                audio_files = list(session_dir.glob("*.wav"))
            if not audio_files:
                print(f"Warning: No audio in {session_dir}")
                continue

            self.samples.append({
                "session_dir": session_dir,
                "audio_path": str(audio_files[0]),
                "annotation_path": str(annotation_path),
                "soft_label_path": str(soft_label_path),
                "session_name": session_name,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        waveform, sr = torchaudio.load(sample["audio_path"])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        audio = waveform.mean(dim=0)  # Mono

        # Extract fbank features
        fbank_features = extract_fbank_features(audio, self.sample_rate)

        # Load soft labels
        teacher_probs = np.load(sample["soft_label_path"])
        if teacher_probs.ndim > 1:
            teacher_probs = teacher_probs.flatten()
        teacher_probs = torch.from_numpy(teacher_probs).float()

        # Load annotation and create hard labels
        with open(sample["annotation_path"]) as f:
            annotation = json.load(f)

        from vad_baseline.distillation.soft_label_generator import (
            extract_speech_segments_from_annotation,
            segments_to_frame_labels,
        )

        segments = extract_speech_segments_from_annotation(annotation)
        num_frames = fbank_features.size(0)
        duration_sec = num_frames * self.frame_shift_sec
        hard_labels = segments_to_frame_labels(
            segments,
            duration_sec=duration_sec,
            frame_shift_sec=self.frame_shift_sec,
        )
        hard_labels = torch.from_numpy(hard_labels).float()

        # Align lengths
        min_len = min(len(fbank_features), len(teacher_probs), len(hard_labels))
        fbank_features = fbank_features[:min_len]
        teacher_probs = teacher_probs[:min_len]
        hard_labels = hard_labels[:min_len]

        return {
            "fbank": fbank_features,
            "teacher_probs": teacher_probs,
            "hard_labels": hard_labels,
        }


def collate_distillation_batch(batch):
    """Collate function for DataLoader.

    Pads sequences to same length within batch.
    """
    batch_size = len(batch)
    max_frame_len = max(item["fbank"].size(0) for item in batch)
    n_mels = batch[0]["fbank"].size(1)

    # Pad fbank features
    fbank_padded = torch.zeros(batch_size, max_frame_len, n_mels)
    for i, item in enumerate(batch):
        fbank_padded[i, : item["fbank"].size(0), :] = item["fbank"]

    # Pad teacher probs
    teacher_probs_padded = torch.zeros(batch_size, max_frame_len)
    for i, item in enumerate(batch):
        teacher_probs_padded[i, : item["teacher_probs"].size(0)] = item["teacher_probs"]

    # Pad hard labels
    hard_labels_padded = torch.zeros(batch_size, max_frame_len)
    for i, item in enumerate(batch):
        hard_labels_padded[i, : item["hard_labels"].size(0)] = item["hard_labels"]

    return {
        "fbank": fbank_padded,
        "teacher_probs": teacher_probs_padded,
        "hard_labels": hard_labels_padded,
    }
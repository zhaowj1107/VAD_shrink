# src/vad_baseline/distillation/dataset.py
"""Dataset classes for VAD distillation."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import torchaudio
from tqdm import tqdm


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
        duration_sec = len(audio) / self.sample_rate
        hard_labels = segments_to_frame_labels(
            segments,
            duration_sec=duration_sec,
            frame_shift_sec=self.frame_shift_sec,
        )
        hard_labels = torch.from_numpy(hard_labels).float()

        # Align lengths (audio might have different length than soft labels)
        min_len = min(len(audio) // 160, len(teacher_probs), len(hard_labels))
        audio = audio[: min_len * 160]
        teacher_probs = teacher_probs[:min_len]
        hard_labels = hard_labels[:min_len]

        return {
            "audio": audio,
            "teacher_probs": teacher_probs,
            "hard_labels": hard_labels,
        }


def collate_distillation_batch(batch):
    """Collate function for DataLoader.

    Pads sequences to same length within batch.
    """
    batch_size = len(batch)
    max_audio_len = max(item["audio"].size(0) for item in batch)
    max_frame_len = max(item["teacher_probs"].size(0) for item in batch)

    # Pad audio
    audio_padded = torch.zeros(batch_size, max_audio_len)
    for i, item in enumerate(batch):
        audio_padded[i, : item["audio"].size(0)] = item["audio"]

    # Pad teacher probs
    teacher_probs_padded = torch.zeros(batch_size, max_frame_len)
    for i, item in enumerate(batch):
        teacher_probs_padded[i, : item["teacher_probs"].size(0)] = item["teacher_probs"]

    # Pad hard labels
    hard_labels_padded = torch.zeros(batch_size, max_frame_len)
    for i, item in enumerate(batch):
        hard_labels_padded[i, : item["hard_labels"].size(0)] = item["hard_labels"]

    return {
        "audio": audio_padded,
        "teacher_probs": teacher_probs_padded,
        "hard_labels": hard_labels_padded,
    }
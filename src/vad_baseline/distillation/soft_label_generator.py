# src/vad_baseline/distillation/soft_label_generator.py
"""Generate soft labels from Teacher model for distillation."""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm


def extract_speech_segments_from_annotation(annotation: dict) -> list[dict]:
    """
    Extract speech segments from LibriParty annotation format.

    LibriParty annotations are dict of speaker_id -> list of utterances.
    Each utterance has start/stop times.

    Returns merged list of unique speech segments.
    """
    all_segments = []

    for speaker_id, utterances in annotation.items():
        for utt in utterances:
            if "start" in utt and "stop" in utt:
                all_segments.append({
                    "start": float(utt["start"]),
                    "stop": float(utt["stop"]),
                })

    # Sort by start time
    all_segments.sort(key=lambda x: x["start"])

    # Merge overlapping segments
    merged = []
    for seg in all_segments:
        if merged and seg["start"] <= merged[-1]["stop"]:
            merged[-1]["stop"] = max(merged[-1]["stop"], seg["stop"])
        else:
            merged.append(seg)

    return merged


def segments_to_frame_labels(
    segments: Sequence[dict],
    duration_sec: float,
    frame_shift_sec: float = 0.01,
) -> np.ndarray:
    """
    Convert speech segments to frame-level binary labels.

    Args:
        segments: List of {start, stop} in seconds
        duration_sec: Total audio duration
        frame_shift_sec: Frame shift (default 10ms)

    Returns:
        Binary array of shape (num_frames,) where 1 = speech
    """
    num_frames = int(duration_sec / frame_shift_sec)
    labels = np.zeros(num_frames, dtype=np.float32)

    for seg in segments:
        start_frame = max(0, int(seg["start"] / frame_shift_sec))
        end_frame = min(num_frames, int(seg["stop"] / frame_shift_sec))
        labels[start_frame:end_frame] = 1.0

    return labels


class SoftLabelGenerator:
    """Generate soft labels from Teacher (FP32) VAD model."""

    def __init__(
        self,
        teacher_model,  # SpeechBrain VAD model
        output_dir: str,
        frame_shift_sec: float = 0.01,
    ):
        self.teacher_model = teacher_model
        self.output_dir = Path(output_dir)
        self.frame_shift_sec = frame_shift_sec
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_for_session(self, audio_path: str, annotation_path: str) -> str:
        """
        Generate soft labels for a single session.

        Args:
            audio_path: Path to session mixture WAV
            annotation_path: Path to session JSON annotation

        Returns:
            Path to saved soft label .npy file
        """
        # Get frame probabilities from Teacher
        frame_probs = self.teacher_model.get_speech_prob_file(str(audio_path))

        # Convert to numpy
        if hasattr(frame_probs, "numpy"):
            frame_probs = frame_probs.numpy()
        elif not isinstance(frame_probs, np.ndarray):
            frame_probs = np.array(frame_probs)

        # Save soft labels
        session_id = Path(audio_path).stem.replace("_mixture", "")
        output_path = self.output_dir / f"{session_id}.npy"
        np.save(output_path, frame_probs)

        return str(output_path)

    def generate_for_sessions(
        self,
        session_dirs: list[str],
        annotation_suffix: str = "_session.json",
    ) -> list[str]:
        """
        Generate soft labels for multiple sessions.

        Args:
            session_dirs: List of paths to session directories
            annotation_suffix: Suffix for annotation file

        Returns:
            List of paths to generated .npy files
        """
        output_paths = []

        for session_dir in tqdm(session_dirs, desc="Generating soft labels"):
            session_dir = Path(session_dir)

            # Find mixture audio
            audio_files = list(session_dir.glob("*_mixture.wav"))
            if not audio_files:
                audio_files = list(session_dir.glob("*.wav"))
            if not audio_files:
                print(f"Warning: No audio found in {session_dir}")
                continue

            audio_path = audio_files[0]

            # Find annotation
            session_name = session_dir.name
            annotation_path = session_dir / f"{session_name}.json"

            if not annotation_path.exists():
                print(f"Warning: No annotation at {annotation_path}")
                continue

            try:
                output_path = self.generate_for_session(
                    str(audio_path),
                    str(annotation_path),
                )
                output_paths.append(output_path)
            except Exception as e:
                print(f"Error processing {session_dir}: {e}")

        return output_paths
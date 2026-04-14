# src/vad_baseline/distillation/__init__.py
"""VAD distillation module."""

from vad_baseline.distillation.config import DistillationConfig
from vad_baseline.distillation.student_model import SimplifiedCRDNN

__all__ = ["DistillationConfig", "SimplifiedCRDNN"]

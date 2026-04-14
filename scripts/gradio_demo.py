"""Gradio demo for VAD (ONNX + numpy fbank backend)."""

import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

ONNX_MODEL_PATH = REPO_ROOT / "outputs" / "onnx_export" / "model.onnx"

from vad_baseline.onnx_runtime import load_onnx_vad_runtime, read_wav_mono

# Load once at startup.
runtime = load_onnx_vad_runtime(str(ONNX_MODEL_PATH))


def run_vad(audio_path):
    if audio_path is None:
        return "请上传一个 WAV 文件。", None

    sample_rate, samples = read_wav_mono(audio_path)
    duration_sec = len(samples) / sample_rate

    if sample_rate != runtime.sample_rate:
        return (
            f"采样率不匹配：需要 {runtime.sample_rate} Hz，"
            f"上传的文件是 {sample_rate} Hz。",
            None,
        )

    t0 = time.perf_counter()
    segments = runtime.predict_segments(audio_path)
    elapsed = time.perf_counter() - t0
    rtf = elapsed / duration_sec

    # --- 文字结果 ---
    lines = [
        f"时长：{duration_sec:.2f}s　　推理时间：{elapsed:.3f}s　　RTF：{rtf:.4f}",
        f"检测到 {len(segments)} 段语音：",
        "",
    ]
    for i, seg in enumerate(segments, 1):
        lines.append(
            f"  [{i:02d}]  {seg['start']:.2f}s → {seg['end']:.2f}s  "
            f"（{seg['duration']:.2f}s）"
        )
    if not segments:
        lines.append("  （未检测到语音）")
    result_text = "\n".join(lines)

    # --- 波形图 ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    times = np.linspace(0, duration_sec, len(samples))

    # 上图：波形
    axes[0].plot(times, samples, color="#555", linewidth=0.4, rasterized=True)
    for seg in segments:
        axes[0].axvspan(seg["start"], seg["end"], alpha=0.25, color="#2196F3")
    axes[0].set_ylabel("振幅")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_xlim(0, duration_sec)

    # 下图：语音概率（frame-level）
    try:
        probs = runtime.get_speech_prob_file(audio_path)  # (1, T, 1)
        probs_1d = probs[0, :, 0]
        frame_times = np.arange(len(probs_1d)) * runtime.time_resolution
        axes[1].fill_between(frame_times, probs_1d, alpha=0.6, color="#4CAF50")
        axes[1].axhline(runtime.activation_th, color="red", linewidth=0.8,
                        linestyle="--", label=f"阈值 {runtime.activation_th}")
        axes[1].set_ylabel("语音概率")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=8, loc="upper right")
    except Exception:
        axes[1].text(0.5, 0.5, "概率图不可用", ha="center", va="center",
                     transform=axes[1].transAxes)

    axes[1].set_xlabel("时间 (s)")
    fig.suptitle("VAD 结果  （蓝色 = 语音段）", fontsize=11)

    return result_text, fig


import gradio as gr

demo = gr.Interface(
    fn=run_vad,
    inputs=gr.Audio(type="filepath", label="上传音频（WAV，16kHz）"),
    outputs=[
        gr.Textbox(label="检测结果", lines=12),
        gr.Plot(label="波形 & 语音概率"),
    ],
    title="VAD Demo — ONNX Runtime + NumPy Fbank",
    description=(
        "基于 SpeechBrain CRDNN 模型，"
        "推理阶段仅依赖 NumPy + ONNX Runtime，无需 PyTorch。"
    ),
    examples=[],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=False)

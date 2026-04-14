"""Gradio demo: compare FP32 / Dynamic INT8 / Static INT8 / ONNX on LibriParty."""

import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# --- paths ---
MANIFEST_PATH = REPO_ROOT / "outputs" / "libriparty_dev_manifest" / "manifest.csv"
ONNX_MODEL_PATH = REPO_ROOT / "outputs" / "onnx_export" / "model.onnx"
NUM_SESSIONS = 10
SESSION_WAVS = [
    REPO_ROOT / "data/external/LibriParty/dataset/dev"
    / f"session_{i}" / f"session_{i}_mixture.wav"
    for i in range(NUM_SESSIONS)
]
SESSION_ANNOTATIONS = [
    REPO_ROOT / "outputs/libriparty_dev_manifest/annotations"
    / f"dev_session_{i}.json"
    for i in range(NUM_SESSIONS)
]

BACKEND_LABELS = {
    "fp32":         "FP32",
    "dynamic_int8": "Dynamic INT8",
    "static_int8":  "Static INT8",
    "onnx":         "ONNX (numpy fbank)",
}

# --- lazy backend cache ---
_cache: dict = {}
_load_times: dict = {}   # name -> load time in seconds
_locks: dict = {k: threading.Lock() for k in BACKEND_LABELS}


def _get_backend(name: str):
    """Load backend on first call; record load time; return cached thereafter."""
    if name in _cache:
        return _cache[name]

    with _locks[name]:
        if name in _cache:
            return _cache[name]

        t0 = time.perf_counter()

        from vad_baseline.backends import get_backend

        if name == "fp32":
            b = get_backend("speechbrain_fp32")
            model = b.load()
        elif name == "dynamic_int8":
            b = get_backend("speechbrain_dynamic_int8")
            model = b.load()
        elif name == "static_int8":
            b = get_backend(
                "speechbrain_static_int8",
                calibration_manifest_path=str(MANIFEST_PATH),
            )
            model = b.load()
        elif name == "onnx":
            from vad_baseline.onnx_runtime import load_onnx_vad_runtime
            b = None
            model = load_onnx_vad_runtime(str(ONNX_MODEL_PATH))
        else:
            raise ValueError(name)

        _load_times[name] = time.perf_counter() - t0
        _cache[name] = (b, model)
    return _cache[name]


# --- audio trimming ---
def _trim_wav(src_path: Path, duration_sec: float, dst_path: str):
    with wave.open(str(src_path), "rb") as r:
        sr = r.getframerate()
        nch = r.getnchannels()
        sw = r.getsampwidth()
        n = min(r.getnframes(), int(sr * duration_sec))
        frames = r.readframes(n)
    with wave.open(dst_path, "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(frames)
    return n / sr


def _trim_annotations(ann_path: Path, duration_sec: float):
    import json
    data = json.loads(ann_path.read_text())
    trimmed = []
    for seg in data:
        start = float(seg["start"])
        end = float(seg["end"])
        if start >= duration_sec:
            continue
        trimmed.append({"start": start, "end": min(end, duration_sec)})
    return trimmed


# --- per-backend inference ---
def _run_one(name: str, wav_path: str, ann_segs: list, duration_sec: float):
    from vad_baseline.metrics import compute_segment_metrics

    backend, model = _get_backend(name)

    t0 = time.perf_counter()
    if name == "onnx":
        segs = model.predict_segments(wav_path)
    else:
        segs = backend.predict_segments(model, wav_path)
    elapsed = time.perf_counter() - t0

    rtf = elapsed / duration_sec
    metrics = compute_segment_metrics(ann_segs, segs) if ann_segs else {}
    return {
        "name": name,
        "inference_sec": elapsed,
        "rtf": rtf,
        "f1": metrics.get("f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "tp_sec": metrics.get("tp_sec"),
        "fp_sec": metrics.get("fp_sec"),
        "fn_sec": metrics.get("fn_sec"),
    }


# --- aggregate results across sessions ---
def _aggregate(session_results: list[dict]):
    """Aggregate per-session results into one row per backend."""
    from vad_baseline.metrics import compute_segment_metrics

    by_backend: dict[str, list] = {}
    for sr in session_results:
        name = sr["name"]
        by_backend.setdefault(name, []).append(sr)

    agg = []
    for name, runs in by_backend.items():
        ok = [r for r in runs if r.get("rtf") is not None]
        if not ok:
            agg.append({"name": name, "rtf": None, "f1": None,
                        "precision": None, "recall": None,
                        "total_sec": None, "error": "all sessions failed"})
            continue

        mean_rtf = sum(r["rtf"] for r in ok) / len(ok)
        total_sec = sum(r["inference_sec"] for r in ok)

        # Aggregate F1: pool all TP/FP/FN across sessions
        tp = sum(r["tp_sec"] for r in ok if r.get("tp_sec") is not None)
        fp = sum(r["fp_sec"] for r in ok if r.get("fp_sec") is not None)
        fn = sum(r["fn_sec"] for r in ok if r.get("fn_sec") is not None)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        agg.append({
            "name": name,
            "rtf": mean_rtf,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "total_sec": total_sec,
        })
    return agg


# --- main callback ---
def run_comparison(n_sessions: int, duration_sec: float, selected_backends: list):
    if not selected_backends:
        return "请至少选择一个 backend。", None

    n_sessions = int(n_sessions)
    session_results = []

    for idx in range(n_sessions):
        wav_src = SESSION_WAVS[idx]
        ann_src = SESSION_ANNOTATIONS[idx]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name

        actual_dur = _trim_wav(wav_src, duration_sec, tmp_wav)
        ann_segs = _trim_annotations(ann_src, actual_dur)

        for name in BACKEND_LABELS:
            if name not in selected_backends:
                continue
            try:
                r = _run_one(name, tmp_wav, ann_segs, actual_dur)
                session_results.append(r)
            except Exception as e:
                session_results.append({
                    "name": name, "inference_sec": None,
                    "rtf": None, "f1": None,
                    "precision": None, "recall": None,
                    "tp_sec": None, "fp_sec": None, "fn_sec": None,
                    "error": str(e),
                })

    rows = _aggregate(session_results)

    total_audio = n_sessions * duration_sec

    # --- text table ---
    lines = [
        f"Sessions: {n_sessions}  |  每段时长: {duration_sec:.0f}s  "
        f"|  总音频: {total_audio:.0f}s",
        "",
        f"{'Backend':<22} {'F1':>6}  {'Precision':>9}  {'Recall':>7}  "
        f"{'Mean RTF':>9}  {'推理总耗时':>10}  {'首次加载':>9}",
        "-" * 82,
    ]
    for r in rows:
        label = BACKEND_LABELS[r["name"]]
        load_t = _load_times.get(r["name"])
        load_str = f"{load_t:.1f}s" if load_t is not None else "(cached)"
        if r.get("error"):
            lines.append(f"{label:<22}  ERROR: {r['error']}")
        else:
            f1  = f"{r['f1']:.4f}"        if r["f1"]        is not None else "  N/A"
            pre = f"{r['precision']:.4f}" if r["precision"]  is not None else "  N/A"
            rec = f"{r['recall']:.4f}"    if r["recall"]     is not None else "  N/A"
            rtf = f"{r['rtf']:.4f}"
            sec = f"{r['total_sec']:.2f}s"
            lines.append(
                f"{label:<22} {f1:>6}  {pre:>9}  {rec:>7}  {rtf:>9}  "
                f"{sec:>10}  {load_str:>9}"
            )
    text = "\n".join(lines)

    # --- bar chart ---
    ok = [r for r in rows if r.get("rtf") is not None]
    if not ok:
        return text, None

    labels = [BACKEND_LABELS[r["name"]] for r in ok]
    rtfs   = [r["rtf"] for r in ok]
    f1s    = [r["f1"] if r["f1"] is not None else 0 for r in ok]

    colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50"]
    x = np.arange(len(ok))
    load_secs = [_load_times.get(r["name"], 0) for r in ok]
    has_load = any(t > 0 for t in load_secs)

    ncols = 3 if has_load else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig.subplots_adjust(wspace=0.38)
    for ax in axes:
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax1, ax2 = axes[0], axes[1]

    bars1 = ax1.bar(x, rtfs, 0.5, color=colors[:len(ok)], alpha=0.85)
    ax1.set_title("平均 RTF（越低越快）")
    ax1.set_ylabel("Real-Time Factor")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    ax1.bar_label(bars1, fmt="%.4f", padding=2, fontsize=8)
    ax1.set_ylim(0, max(rtfs) * 1.35)

    bars2 = ax2.bar(x, f1s, 0.5, color=colors[:len(ok)], alpha=0.85)
    ax2.set_title("F1 Score（越高越好）")
    ax2.set_ylabel("F1")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    ax2.bar_label(bars2, fmt="%.4f", padding=2, fontsize=8)
    ax2.set_ylim(
        max(0, min(f1s) - 0.05) if any(f > 0 for f in f1s) else 0,
        1.05,
    )

    if has_load:
        ax3 = axes[2]
        bars3 = ax3.bar(x, load_secs, 0.5, color=colors[:len(ok)], alpha=0.85)
        ax3.set_title("首次加载时间（越低越快）")
        ax3.set_ylabel("秒")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
        ax3.bar_label(bars3, fmt="%.1fs", padding=2, fontsize=8)
        ax3.set_ylim(0, max(load_secs) * 1.35)

    fig.suptitle(
        f"LibriParty Dev — {n_sessions} sessions × 前 {duration_sec:.0f}s",
        fontsize=11,
    )

    return text, fig


# --- Gradio UI ---
import gradio as gr

with gr.Blocks(title="VAD 模型对比") as demo:
    gr.Markdown("## VAD 模型对比 — LibriParty Dev Set（10 sessions）")
    gr.Markdown(
        "选择使用多少个 session 和每段截取时长，勾选要对比的 backend，点击运行。  \n"
        "首次运行每个 backend 需要加载模型（约 10–30s），之后缓存复用。"
    )

    with gr.Row():
        session_slider = gr.Slider(
            minimum=1, maximum=NUM_SESSIONS, step=1, value=2,
            label=f"Session 数量（共 {NUM_SESSIONS} 个）", scale=1,
        )
        duration_slider = gr.Slider(
            minimum=10, maximum=294, step=10, value=60,
            label="每段截取时长（秒）", scale=2,
        )

    backend_checkboxes = gr.CheckboxGroup(
        choices=list(BACKEND_LABELS.keys()),
        value=list(BACKEND_LABELS.keys()),
        label="对比 Backend",
        type="value",
    )

    run_btn = gr.Button("运行对比", variant="primary")

    result_text = gr.Textbox(label="结果", lines=10, max_lines=15)
    result_plot = gr.Plot(label="对比图")

    run_btn.click(
        fn=run_comparison,
        inputs=[session_slider, duration_slider, backend_checkboxes],
        outputs=[result_text, result_plot],
    )


if __name__ == "__main__":
    demo.launch(share=False)

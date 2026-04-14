import json
import wave
from pathlib import Path

import numpy as np

from vad_baseline.onnx_export import metadata_path_for_model


def _stable_mb(value):
    return round(float(value), 6)


def _stable_float(value):
    return round(float(value), 6)


def _empty_boundaries():
    return np.empty((0, 2), dtype=np.float64)


def _read_audio_with_torchcodec(audio_path):
    try:
        from torchcodec.decoders import AudioDecoder
    except Exception:
        return None

    try:
        decoded = AudioDecoder(str(audio_path)).get_all_samples()
    except Exception:
        return None

    samples = decoded.data
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        mono_samples = samples
    elif samples.ndim == 2:
        if samples.shape[0] == 1:
            mono_samples = samples[0]
        elif samples.shape[1] == 1:
            mono_samples = samples[:, 0]
        elif samples.shape[0] <= 8 and samples.shape[1] > samples.shape[0]:
            mono_samples = samples.mean(axis=0)
        elif samples.shape[1] <= 8 and samples.shape[0] > samples.shape[1]:
            mono_samples = samples.mean(axis=1)
        else:
            mono_samples = samples.mean(axis=0)
    else:
        raise ValueError(f"unsupported decoded audio shape: {samples.shape}")

    return int(decoded.sample_rate), mono_samples.astype(np.float32, copy=False)


def read_wav_mono(audio_path):
    torchcodec_result = _read_audio_with_torchcodec(audio_path)
    if torchcodec_result is not None:
        return torchcodec_result

    with wave.open(str(audio_path), "rb") as handle:
        sample_rate = handle.getframerate()
        num_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        num_frames = handle.getnframes()
        raw_bytes = handle.readframes(num_frames)

    if sample_width == 1:
        samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sample_width == 3:
        raw_uint8 = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        samples = (
            raw_uint8[:, 0].astype(np.int32)
            | (raw_uint8[:, 1].astype(np.int32) << 8)
            | (raw_uint8[:, 2].astype(np.int32) << 16)
        )
        sign_mask = 1 << 23
        samples = ((samples ^ sign_mask) - sign_mask).astype(np.float32)
        samples = samples / 8388608.0
    elif sample_width == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
        samples = samples / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if num_channels > 1:
        samples = samples.reshape(-1, num_channels).mean(axis=1)

    return sample_rate, samples.astype(np.float32)


class ONNXVADRuntime:
    def __init__(
        self,
        session,
        model_path,
        sample_rate,
        time_resolution,
        input_names,
        output_names,
        feature_extractor,
        large_chunk_size=30.0,
        small_chunk_size=10.0,
        close_th=0.250,
        len_th=0.250,
        activation_th=0.5,
        deactivation_th=0.25,
        speech_th=0.5,
        double_check=True,
    ):
        self.session = session
        self.model_path = str(Path(model_path).resolve())
        self.sample_rate = int(sample_rate)
        self.time_resolution = float(time_resolution)
        self.input_names = list(input_names)
        self.output_names = list(output_names)
        self.feature_extractor = feature_extractor
        self.large_chunk_size = float(large_chunk_size)
        self.small_chunk_size = float(small_chunk_size)
        self.close_th = float(close_th)
        self.len_th = float(len_th)
        self.activation_th = float(activation_th)
        self.deactivation_th = float(deactivation_th)
        self.speech_th = float(speech_th)
        self.double_check = bool(double_check)

    def get_speech_prob_chunk(self, wavs):
        feats = self.feature_extractor(wavs)
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 2:
            feats = feats[np.newaxis, ...]
        outputs = self.session.run(
            self.output_names,
            {
                self.input_names[0]: feats,
            },
        )[0]
        return np.asarray(outputs, dtype=np.float32)

    def get_speech_prob_file(self, audio_path):
        sample_rate, samples = read_wav_mono(audio_path)
        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from the exported model metadata"
            )

        long_chunk_len = int(sample_rate * self.large_chunk_size)
        small_chunk_len = int(sample_rate * self.small_chunk_size)
        if long_chunk_len <= 0 or small_chunk_len <= 0:
            raise ValueError("chunk sizes must be positive")

        prob_chunks = []
        total_samples = samples.shape[0]
        for begin_sample in range(0, total_samples, long_chunk_len):
            large_chunk = samples[begin_sample : begin_sample + long_chunk_len]
            small_chunks = []
            for small_begin in range(0, max(len(large_chunk), 1), small_chunk_len):
                small_chunk = large_chunk[small_begin : small_begin + small_chunk_len]
                if small_chunk.shape[0] < small_chunk_len:
                    padded = np.zeros((small_chunk_len,), dtype=np.float32)
                    padded[: small_chunk.shape[0]] = small_chunk
                    small_chunk = padded
                small_chunks.append(small_chunk)

            small_chunks = np.stack(small_chunks, axis=0)
            small_chunks_prob = self.get_speech_prob_chunk(small_chunks)
            if small_chunks_prob.shape[1] == 0:
                continue
            small_chunks_prob = small_chunks_prob[:, :-1, :]
            prob_chunks.append(small_chunks_prob.reshape(-1, small_chunks_prob.shape[-1]))

        if not prob_chunks:
            return np.zeros((1, 0, 1), dtype=np.float32)

        prob_vad = np.concatenate(prob_chunks, axis=0)
        last_elem = int(total_samples / (self.time_resolution * sample_rate))
        prob_vad = prob_vad[:last_elem]
        return prob_vad[np.newaxis, :, :]

    def apply_threshold(self, vad_prob):
        frame_does_not_deactivate = vad_prob >= self.deactivation_th
        vad_th = vad_prob >= self.activation_th

        for index in range(1, vad_prob.shape[1]):
            vad_th[:, index, ...] |= vad_th[:, index - 1, ...]
            vad_th[:, index, ...] &= frame_does_not_deactivate[:, index, ...]

        return vad_th.astype(np.int32)

    def get_boundaries(self, prob_th):
        if prob_th.shape[1] == 0:
            return _empty_boundaries()

        prob_th = prob_th.astype(np.int32, copy=True)
        prob_th_shifted = np.roll(prob_th, shift=1, axis=1)
        prob_th_shifted[:, 0, :] = 0
        prob_th = prob_th + prob_th_shifted
        prob_th[:, 0, :] = (prob_th[:, 0, :] >= 1).astype(np.int32)
        prob_th[:, -1, :] = (prob_th[:, -1, :] >= 1).astype(np.int32)

        if np.argwhere(prob_th == 1).shape[0] % 2 == 1:
            prob_th = np.concatenate(
                [prob_th, np.ones((1, 1, 1), dtype=np.int32)],
                axis=1,
            )

        indexes = np.argwhere(prob_th == 1)
        if indexes.shape[0] == 0:
            return _empty_boundaries()

        indexes = indexes[:, 1].reshape(-1, 2)
        indexes[:, -1] = indexes[:, -1] - 1
        return indexes * self.time_resolution

    def merge_close_segments(self, boundaries):
        if boundaries.shape[0] == 0:
            return boundaries

        new_boundaries = []
        prev_beg_seg = float(boundaries[0, 0])
        prev_end_seg = float(boundaries[0, 1])

        for index in range(1, boundaries.shape[0]):
            beg_seg = float(boundaries[index, 0])
            segment_distance = beg_seg - prev_end_seg
            if segment_distance <= self.close_th:
                prev_end_seg = float(boundaries[index, 1])
            else:
                new_boundaries.append([prev_beg_seg, prev_end_seg])
                prev_beg_seg = beg_seg
                prev_end_seg = float(boundaries[index, 1])

        new_boundaries.append([prev_beg_seg, prev_end_seg])
        return np.asarray(new_boundaries, dtype=np.float64)

    def remove_short_segments(self, boundaries):
        if boundaries.shape[0] == 0:
            return boundaries

        kept = []
        for boundary in boundaries:
            seg_len = float(boundary[1] - boundary[0])
            if seg_len > self.len_th:
                kept.append([float(boundary[0]), float(boundary[1])])
        if not kept:
            return _empty_boundaries()
        return np.asarray(kept, dtype=np.float64)

    def double_check_speech_segments(self, boundaries, samples):
        if boundaries.shape[0] == 0:
            return boundaries

        kept = []
        for beg_sec, end_sec in boundaries:
            beg_sample = int(beg_sec * self.sample_rate)
            end_sample = int(end_sec * self.sample_rate)
            segment = samples[beg_sample:end_sample]
            if segment.shape[0] == 0:
                continue
            speech_prob = self.get_speech_prob_chunk(segment)
            if float(speech_prob.mean()) > self.speech_th:
                kept.append([float(beg_sec), float(end_sec)])

        if not kept:
            return _empty_boundaries()
        return np.asarray(kept, dtype=np.float64)

    def predict_segments(self, audio_path):
        sample_rate, samples = read_wav_mono(audio_path)
        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from the exported model metadata"
            )

        prob_chunks = self.get_speech_prob_file(audio_path)
        prob_th = self.apply_threshold(prob_chunks)
        boundaries = self.get_boundaries(prob_th)
        boundaries = self.merge_close_segments(boundaries)
        boundaries = self.remove_short_segments(boundaries)
        if self.double_check:
            boundaries = self.double_check_speech_segments(boundaries, samples)

        return [
            {
                "start": _stable_float(start),
                "end": _stable_float(end),
                "duration": _stable_float(end - start),
            }
            for start, end in boundaries.tolist()
        ]


def _numpy_fbank(wavs, mel_matrix, window, hop_length=160):
    """Compute log-mel fbank features using only NumPy (no PyTorch/SpeechBrain).

    Matches SpeechBrain's Fbank pipeline:
      zero-pad → hamming window → rfft → power spectrum → mel filterbank
      → 10*log10(clamp(x, 1e-10)) → top-80-dB clip

    Args:
        wavs:       (batch, samples) or (samples,) float32 array
        mel_matrix: (n_fft//2+1, n_mels) float32 — linear mel weights
        window:     (n_fft,) float32 — hamming analysis window
        hop_length: int, frame hop in samples

    Returns:
        (batch, num_frames, n_mels) float32
    """
    wavs = np.asarray(wavs, dtype=np.float32)
    if wavs.ndim == 1:
        wavs = wavs[np.newaxis, :]

    n_fft = len(window)
    pad = n_fft // 2
    results = []

    for wav in wavs:
        padded = np.pad(wav, (pad, pad), mode="constant")
        n_frames = (len(padded) - n_fft) // hop_length + 1
        # Use stride tricks to create overlapping frames without copying.
        frames = np.lib.stride_tricks.as_strided(
            padded,
            shape=(n_frames, n_fft),
            strides=(padded.strides[0] * hop_length, padded.strides[0]),
        ).copy()
        windowed = frames * window  # (n_frames, n_fft)
        stft = np.fft.rfft(windowed, n=n_fft, axis=-1)  # (n_frames, n_fft//2+1)
        power = stft.real**2 + stft.imag**2  # power spectrum
        mel = power @ mel_matrix  # (n_frames, n_mels)
        log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, log_mel.max() - 80.0)  # top-80-dB clip
        results.append(log_mel)

    return np.stack(results, axis=0).astype(np.float32)


def build_feature_extractor_from_metadata(metadata, model_path=None):
    frontend_name = metadata.get("frontend")

    if frontend_name == "numpy_fbank":
        if model_path is None:
            raise ValueError(
                "model_path is required to locate the .fbank.npz sidecar "
                "for a numpy_fbank model"
            )
        fbank_path = Path(model_path).with_name(
            f"{Path(model_path).stem}.fbank.npz"
        )
        if not fbank_path.exists():
            raise FileNotFoundError(f"missing numpy fbank sidecar: {fbank_path}")
        fbank_data = np.load(str(fbank_path))
        mel_matrix = fbank_data["mel_matrix"].astype(np.float32)
        window = fbank_data["window"].astype(np.float32)
        hop_length = int(metadata.get("hop_length", 160))

        def extract_features(wavs):
            return _numpy_fbank(wavs, mel_matrix, window, hop_length)

        return extract_features

    if frontend_name == "speechbrain_fbank":
        from vad_baseline.model import _ensure_torchaudio_backend_compat

        _ensure_torchaudio_backend_compat()
        import torch
        from speechbrain.lobes.features import Fbank

        frontend = Fbank(
            deltas=False,
            context=False,
            requires_grad=False,
            sample_rate=int(metadata["sample_rate"]),
        )
        frontend.eval()

        def extract_features(wavs):
            wavs = np.asarray(wavs, dtype=np.float32)
            wavs_tensor = torch.as_tensor(wavs, dtype=torch.float32)
            if wavs_tensor.ndim == 1:
                wavs_tensor = wavs_tensor.unsqueeze(0)
            with torch.no_grad():
                feats = frontend(wavs_tensor)
            return feats.detach().cpu().numpy().astype(np.float32)

        return extract_features

    raise ValueError(f"unsupported frontend: {frontend_name}")


def load_onnx_vad_runtime(
    onnx_model_path,
    session_factory=None,
    feature_extractor_factory=None,
):
    model_path = Path(onnx_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"missing ONNX model: {model_path}")

    metadata_path = metadata_path_for_model(model_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata sidecar: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())

    if session_factory is None:
        import onnxruntime

        session_factory = onnxruntime.InferenceSession
    if feature_extractor_factory is None:
        feature_extractor_factory = build_feature_extractor_from_metadata

    session = session_factory(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    return ONNXVADRuntime(
        session=session,
        model_path=str(model_path.resolve()),
        sample_rate=int(metadata["sample_rate"]),
        time_resolution=float(metadata["time_resolution"]),
        input_names=metadata["input_names"],
        output_names=metadata["output_names"],
        feature_extractor=feature_extractor_factory(metadata, model_path=str(model_path)),
    )

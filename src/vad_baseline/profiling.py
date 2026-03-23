import resource
from pathlib import Path
from time import perf_counter

from vad_baseline.batch import (
    process_manifest_entry,
    read_manifest,
    summarize_results,
)
from vad_baseline.io_utils import write_json, write_jsonl
from vad_baseline.model import load_vad_model, model_source_name


def _stable_float(value):
    return round(float(value), 12)


def _stable_mb(value):
    return round(float(value), 6)


def parse_rss_mb(status_text):
    for line in status_text.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError("VmRSS line missing numeric value")
        return _stable_mb(int(fields[1]) / 1024)
    raise ValueError("VmRSS line not found")


def read_current_rss_mb(status_path="/proc/self/status"):
    return parse_rss_mb(Path(status_path).read_text())


def read_peak_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return _stable_mb(usage.ru_maxrss / 1024)


def read_cpu_times_sec():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "cpu_user_sec": _stable_float(usage.ru_utime),
        "cpu_system_sec": _stable_float(usage.ru_stime),
    }


def delta_cpu_times(before, after):
    return {
        "cpu_user_sec": _stable_float(
            after["cpu_user_sec"] - before["cpu_user_sec"]
        ),
        "cpu_system_sec": _stable_float(
            after["cpu_system_sec"] - before["cpu_system_sec"]
        ),
    }


def summarize_model_tensors(vad_model):
    modules = getattr(vad_model, "mods", None)
    if modules is None:
        parameter_count = 0
        parameter_bytes = 0
        buffer_bytes = 0
    else:
        parameters = list(modules.parameters())
        buffers = list(modules.buffers())
        parameter_count = sum(parameter.numel() for parameter in parameters)
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in parameters
        )
        buffer_bytes = sum(
            buffer.numel() * buffer.element_size()
            for buffer in buffers
        )

    total_tensor_bytes = parameter_bytes + buffer_bytes
    return {
        "model_parameter_count": int(parameter_count),
        "model_parameter_bytes": int(parameter_bytes),
        "model_buffer_bytes": int(buffer_bytes),
        "model_total_tensor_bytes": int(total_tensor_bytes),
        "model_parameter_mb": _stable_mb(parameter_bytes / (1024 * 1024)),
        "model_buffer_mb": _stable_mb(buffer_bytes / (1024 * 1024)),
        "model_total_tensor_mb": _stable_mb(total_tensor_bytes / (1024 * 1024)),
    }


def _with_backend_metadata(payload, backend):
    if backend is None:
        return payload
    return {
        "backend_name": backend.backend_name,
        "model_name": backend.model_name,
        **payload,
    }


def _should_record_memory_stages(backend):
    return backend is not None and backend.backend_name == "speechbrain_fp32"


def _build_memory_stage(
    stage_name,
    rss_mb,
    peak_rss_mb,
    cpu_times,
    start_rss_mb,
    previous_rss_mb,
    start_cpu_times,
):
    cpu_delta = delta_cpu_times(start_cpu_times, cpu_times)
    return {
        "stage_name": stage_name,
        "rss_mb": rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "cpu_user_sec": cpu_delta["cpu_user_sec"],
        "cpu_system_sec": cpu_delta["cpu_system_sec"],
        "delta_rss_from_previous_mb": _stable_mb(rss_mb - previous_rss_mb),
        "delta_rss_from_start_mb": _stable_mb(rss_mb - start_rss_mb),
    }


def _append_memory_stage(
    stages,
    stage_name,
    current_rss_reader,
    peak_rss_reader,
    cpu_times_reader,
    start_rss_mb,
    start_cpu_times,
):
    stage = _build_memory_stage(
        stage_name=stage_name,
        rss_mb=current_rss_reader(),
        peak_rss_mb=peak_rss_reader(),
        cpu_times=cpu_times_reader(),
        start_rss_mb=start_rss_mb,
        previous_rss_mb=stages[-1]["rss_mb"],
        start_cpu_times=start_cpu_times,
    )
    stages.append(stage)


def profile_batch_manifest(
    manifest_path,
    output_dir,
    save_frame_probs=False,
    backend=None,
    load_model_fn=None,
    timer=perf_counter,
    current_rss_reader=read_current_rss_mb,
    peak_rss_reader=read_peak_rss_mb,
    cpu_times_reader=read_cpu_times_sec,
    process_manifest_entry_fn=None,
):
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir)
    entries = read_manifest(manifest_path)
    num_entries = len(entries)
    load_model_fn = load_model_fn or load_vad_model
    process_manifest_entry_fn = (
        process_manifest_entry_fn or process_manifest_entry
    )

    rss_before_load_mb = current_rss_reader()
    peak_rss_before_load_mb = peak_rss_reader()
    cpu_before_load = cpu_times_reader()
    memory_stages = None
    if _should_record_memory_stages(backend):
        memory_stages = [
            _build_memory_stage(
                stage_name="process_start",
                rss_mb=rss_before_load_mb,
                peak_rss_mb=peak_rss_before_load_mb,
                cpu_times=cpu_before_load,
                start_rss_mb=rss_before_load_mb,
                previous_rss_mb=rss_before_load_mb,
                start_cpu_times=cpu_before_load,
            )
        ]

    load_started_at = timer()
    vad_model = load_model_fn() if backend is None else backend.load()
    load_time_sec = timer() - load_started_at

    rss_after_load_mb = current_rss_reader()
    peak_rss_after_load_mb = peak_rss_reader()
    cpu_after_load = cpu_times_reader()
    if memory_stages is not None:
        memory_stages.append(
            _build_memory_stage(
                stage_name="after_backend_load",
                rss_mb=rss_after_load_mb,
                peak_rss_mb=peak_rss_after_load_mb,
                cpu_times=cpu_after_load,
                start_rss_mb=rss_before_load_mb,
                previous_rss_mb=memory_stages[-1]["rss_mb"],
                start_cpu_times=cpu_before_load,
            )
        )

    run_started_at = timer()
    results = []
    if memory_stages is not None and entries:
        first_entry = entries[0]
        stage_callbacks = {
            "after_metadata": lambda: _append_memory_stage(
                memory_stages,
                "after_first_entry_metadata",
                current_rss_reader,
                peak_rss_reader,
                cpu_times_reader,
                rss_before_load_mb,
                cpu_before_load,
            ),
            "after_inference": lambda: _append_memory_stage(
                memory_stages,
                "after_first_entry_inference",
                current_rss_reader,
                peak_rss_reader,
                cpu_times_reader,
                rss_before_load_mb,
                cpu_before_load,
            ),
            "after_scoring": lambda: _append_memory_stage(
                memory_stages,
                "after_first_entry_scoring",
                current_rss_reader,
                peak_rss_reader,
                cpu_times_reader,
                rss_before_load_mb,
                cpu_before_load,
            ),
        }
        results.append(
            process_manifest_entry_fn(
                first_entry,
                vad_model,
                output_dir,
                save_frame_probs=save_frame_probs,
                backend=backend,
                stage_callbacks=stage_callbacks,
            )
        )
        entries = entries[1:]

    results.extend(
        [
            (
                process_manifest_entry_fn(
                    entry,
                    vad_model,
                    output_dir,
                    save_frame_probs=save_frame_probs,
                )
                if backend is None
                else process_manifest_entry_fn(
                    entry,
                    vad_model,
                    output_dir,
                    save_frame_probs=save_frame_probs,
                    backend=backend,
                )
            )
            for entry in entries
        ]
    )
    run_wall_time_sec = timer() - run_started_at

    rss_after_run_mb = current_rss_reader()
    peak_rss_after_run_mb = peak_rss_reader()
    cpu_after_run = cpu_times_reader()
    if memory_stages is not None:
        memory_stages.append(
            _build_memory_stage(
                stage_name="after_full_run",
                rss_mb=rss_after_run_mb,
                peak_rss_mb=peak_rss_after_run_mb,
                cpu_times=cpu_after_run,
                start_rss_mb=rss_before_load_mb,
                previous_rss_mb=memory_stages[-1]["rss_mb"],
                start_cpu_times=cpu_before_load,
            )
        )

    batch_summary = summarize_results(results)
    batch_summary = _with_backend_metadata(batch_summary, backend)
    profile_summary = {
        "manifest_path": str(manifest_path),
        "model_name": (
            model_source_name() if backend is None else backend.model_name
        ),
        "num_entries": num_entries,
        "load_time_sec": _stable_float(load_time_sec),
        "run_wall_time_sec": _stable_float(run_wall_time_sec),
        "total_wall_time_sec": _stable_float(load_time_sec + run_wall_time_sec),
        "rss_before_load_mb": rss_before_load_mb,
        "rss_after_load_mb": rss_after_load_mb,
        "rss_after_run_mb": rss_after_run_mb,
        "peak_rss_before_load_mb": peak_rss_before_load_mb,
        "peak_rss_after_load_mb": peak_rss_after_load_mb,
        "peak_rss_after_run_mb": peak_rss_after_run_mb,
        **{
            f"cpu_user_load_sec": delta_cpu_times(
                cpu_before_load,
                cpu_after_load,
            )["cpu_user_sec"],
            f"cpu_system_load_sec": delta_cpu_times(
                cpu_before_load,
                cpu_after_load,
            )["cpu_system_sec"],
            f"cpu_user_run_sec": delta_cpu_times(
                cpu_after_load,
                cpu_after_run,
            )["cpu_user_sec"],
            f"cpu_system_run_sec": delta_cpu_times(
                cpu_after_load,
                cpu_after_run,
            )["cpu_system_sec"],
        },
        **(
            summarize_model_tensors(vad_model)
            if backend is None
            else backend.summarize_model_tensors(vad_model)
        ),
        "batch_summary": batch_summary,
    }
    if memory_stages is not None:
        profile_summary["memory_stages"] = memory_stages
    profile_summary = _with_backend_metadata(profile_summary, backend)

    write_jsonl(output_dir / "per_file.jsonl", results)
    write_json(output_dir / "summary.json", batch_summary)
    write_json(output_dir / "profile.json", profile_summary)
    return profile_summary

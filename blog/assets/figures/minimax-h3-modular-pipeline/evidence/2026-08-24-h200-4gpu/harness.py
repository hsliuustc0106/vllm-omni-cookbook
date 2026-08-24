#!/usr/bin/env python3
"""Four-GPU MiniMax-H3 startup and complete-response timing harness.

This is an external measurement harness. It does not patch repository source,
drop filesystem caches, or launch unless canhazgpu supplies exactly four GPUs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


PINNED_SHA = "072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65"
DEFAULT_OUTPUT = Path("/home/hsliu/tmp/minimax-h3-e2e-4gpu-20260824/artifacts")
DEFAULT_REPO = Path("/root/.codex/worktrees/561c/vllm-omni")
DEFAULT_VENV = Path("/home/hsliu/tmp/venvs/vllm-0.27.0")
DEFAULT_MODEL = Path(
    "/data/models/hub/models--MiniMaxAI--MiniMax-H3/"
    "snapshots/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA"
)
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
PROMPT = (
    "At night, three cats march into a bedroom playing tiny brass instruments, "
    "then abruptly file out, with synchronized room ambience."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=18093)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command_output(command: list[str], timeout: float = 60.0) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:  # Preserve environmental failures in the artifact.
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def require_pinned_checkout(repo: Path) -> None:
    resolved = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != PINNED_SHA:
        raise RuntimeError(f"expected pinned checkout {PINNED_SHA}, got {resolved}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"pinned checkout is dirty:\n{dirty}")


def physical_gpu_ids() -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        ids = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must contain physical integer IDs, got {raw!r}") from exc
    if len(ids) != 4 or len(set(ids)) != 4:
        raise RuntimeError(
            "this experiment requires exactly four GPUs supplied by `gpu run`; "
            f"CUDA_VISIBLE_DEVICES={raw!r}"
        )
    return ids


def gpu_query(selected: set[int]) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 7:
            continue
        index = int(fields[0])
        if index not in selected:
            continue
        rows.append(
            {
                "index": index,
                "uuid": fields[1],
                "name": fields[2],
                "driver_version": fields[3],
                "memory_total_mib": int(fields[4]),
                "memory_used_mib": int(fields[5]),
                "utilization_pct": float(fields[6]),
            }
        )
    if {row["index"] for row in rows} != selected:
        raise RuntimeError(f"nvidia-smi did not return every reserved GPU: {rows}")
    return sorted(rows, key=lambda row: row["index"])


def process_tree(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            status = (item / "status").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        match = re.search(r"^PPid:\s+(\d+)\s*$", status, re.MULTILINE)
        if match:
            parents[int(item.name)] = int(match.group(1))
    found = {root_pid}
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        for pid, ppid in parents.items():
            if ppid == parent and pid not in found:
                found.add(pid)
                frontier.append(pid)
    return found


def smaps_value(pid: int, key: str) -> int:
    try:
        text = Path(f"/proc/{pid}/smaps_rollup").read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return 0
    match = re.search(rf"^{re.escape(key)}:\s+(\d+)\s+kB\s*$", text, re.MULTILINE)
    return int(match.group(1)) if match else 0


class ResourceSampler(threading.Thread):
    def __init__(self, server_pid: int, selected: set[int], output_dir: Path) -> None:
        super().__init__(daemon=True)
        self.server_pid = server_pid
        self.selected = selected
        self.output_dir = output_dir
        self.phase = "startup"
        self.stop_event = threading.Event()
        self.error: str | None = None

    def run(self) -> None:
        gpu_path = self.output_dir / "gpu_samples.csv"
        host_path = self.output_dir / "host_samples.jsonl"
        try:
            with gpu_path.open("w", newline="") as gpu_file, host_path.open("w") as host_file:
                writer = csv.DictWriter(
                    gpu_file,
                    fieldnames=["wall_epoch", "phase", "gpu", "memory_used_mib", "utilization_pct"],
                )
                writer.writeheader()
                while not self.stop_event.is_set():
                    now = time.time()
                    pids = process_tree(self.server_pid)
                    host = {
                        "wall_epoch": now,
                        "phase": self.phase,
                        "process_count": len(pids),
                        "rss_kb": sum(smaps_value(pid, "Rss") for pid in pids),
                        "pss_kb": sum(smaps_value(pid, "Pss") for pid in pids),
                        "private_dirty_kb": sum(smaps_value(pid, "Private_Dirty") for pid in pids),
                        "shared_clean_kb": sum(smaps_value(pid, "Shared_Clean") for pid in pids),
                        "locked_kb": sum(smaps_value(pid, "Locked") for pid in pids),
                    }
                    host_file.write(json.dumps(host, separators=(",", ":")) + "\n")
                    host_file.flush()
                    for row in gpu_query(self.selected):
                        writer.writerow(
                            {
                                "wall_epoch": now,
                                "phase": self.phase,
                                "gpu": row["index"],
                                "memory_used_mib": row["memory_used_mib"],
                                "utilization_pct": row["utilization_pct"],
                            }
                        )
                    gpu_file.flush()
                    self.stop_event.wait(1.0)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"


def assert_port_free(port: int) -> None:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    finally:
        sock.close()


def wait_for_health(
    server: subprocess.Popen[bytes],
    base_url: str,
    started_mono: float,
    output_dir: Path,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    polls = output_dir / "health_poll.jsonl"
    with polls.open("w") as handle:
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"server exited during startup with code {server.returncode}")
            record: dict[str, Any] = {
                "wall_epoch": time.time(),
                "elapsed_s": time.monotonic() - started_mono,
            }
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                record["status"] = response.status_code
                if response.status_code == 200:
                    record["observed_health_wall_epoch"] = time.time()
                    record["startup_to_health_s"] = time.monotonic() - started_mono
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                    handle.flush()
                    return record
            except requests.RequestException as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            time.sleep(0.5)
    raise TimeoutError(f"server did not return /health=200 within {timeout_s:.1f}s")


def media_probe(path: Path) -> dict[str, Any]:
    probe = command_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            (
                "stream=index,codec_type,codec_name,width,height,r_frame_rate,"
                "sample_rate,channels,duration,nb_read_frames"
            ),
            "-of",
            "json",
            str(path),
        ],
        timeout=60,
    )
    parsed = json.loads(probe["stdout"]) if probe["returncode"] == 0 else {}
    streams = parsed.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    errors: list[str] = []
    if video is None:
        errors.append("missing video stream")
    else:
        expected_video = {
            "codec_name": "h264",
            "width": 1344,
            "height": 768,
            "r_frame_rate": "24/1",
            "nb_read_frames": "124",
        }
        for key, expected in expected_video.items():
            if video.get(key) != expected:
                errors.append(f"video {key}={video.get(key)!r}, expected {expected!r}")
    if audio is None:
        errors.append("missing audio stream")
    else:
        expected_audio = {"codec_name": "aac", "sample_rate": "32000", "channels": 2}
        for key, expected in expected_audio.items():
            if audio.get(key) != expected:
                errors.append(f"audio {key}={audio.get(key)!r}, expected {expected!r}")
    hashes = {}
    for kind, mapping in (("video", "0:v:0"), ("audio", "0:a:0")):
        result = command_output(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", mapping, "-f", "hash", "-hash", "sha256", "-"],
            timeout=180,
        )
        hashes[kind] = result
    return {
        "valid": not errors and probe["returncode"] == 0,
        "errors": errors,
        "ffprobe": probe,
        "streams": streams,
        "decoded_stream_hashes": hashes,
    }


def request_payload(sigma_points: int) -> dict[str, str]:
    return {
        "prompt": PROMPT,
        "width": "1344",
        "height": "768",
        "aspect_ratio": "16:9",
        "fps": "24",
        "num_inference_steps": str(sigma_points),
        "flow_shift": "12",
        "seed": "1101",
        "extra_params": json.dumps(
            {"task": "t2va", "duration": 5.0, "audio_flow_shift": 3.0},
            separators=(",", ":"),
        ),
    }


def run_request(
    session: requests.Session,
    base_url: str,
    output_dir: Path,
    label: str,
    sigma_points: int,
    measured: bool,
    timeout_s: float,
) -> dict[str, Any]:
    output_path = output_dir / f"{label}.mp4"
    started_wall = time.time()
    started = time.perf_counter()
    response = session.post(
        f"{base_url}/v1/videos/sync",
        data=request_payload(sigma_points),
        stream=True,
        timeout=timeout_s,
    )
    headers_received = time.perf_counter()
    response.raise_for_status()
    digest = hashlib.sha256()
    byte_count = 0
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
    body_received = time.perf_counter()
    headers = {key.lower(): value for key, value in response.headers.items()}
    json_dump(output_dir / f"{label}.headers.json", headers)
    try:
        stage_durations = json.loads(headers.get("x-stage-durations", "{}"))
    except json.JSONDecodeError:
        stage_durations = {"parse_error": headers.get("x-stage-durations")}
    return {
        "label": label,
        "measured": measured,
        "sigma_points": sigma_points,
        "denoiser_evaluations": sigma_points - 1,
        "started_wall_epoch": started_wall,
        "output": str(output_path),
        "status": response.status_code,
        "bytes": byte_count,
        "container_sha256": digest.hexdigest(),
        "client_total_s": body_received - started,
        "client_ttfb_s": headers_received - started,
        "response_body_s": body_received - headers_received,
        "server_inference_s": float(headers.get("x-inference-time-s", 0.0)),
        "peak_memory_mb": float(headers.get("x-peak-memory-mb", 0.0)),
        "stage_durations": stage_durations,
        "server_request_id": headers.get("x-request-id"),
        "media": media_probe(output_path),
    }


def parse_cpu_encode_spans(log_text: str, request_count: int) -> list[float | None]:
    chunks = re.split(r"Video sampling params:", log_text)[1:]
    pattern = re.compile(r"Video response encoding \(MP4 bytes\): ([0-9.]+) ms")
    spans: list[float | None] = []
    for chunk in chunks[:request_count]:
        match = pattern.search(chunk)
        spans.append(float(match.group(1)) / 1000.0 if match else None)
    while len(spans) < request_count:
        spans.append(None)
    return spans


def parse_timestamp(line: str, reference_epoch: float) -> float | None:
    match = re.search(r"(?:INFO|WARNING|ERROR) (\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2})", line)
    if not match:
        return None
    reference = datetime.fromtimestamp(reference_epoch).astimezone()
    month, day = int(match.group(1)), int(match.group(2))
    hour, minute, second = (int(value) for value in match.group(3).split(":"))
    parsed = datetime(reference.year, month, day, hour, minute, second, tzinfo=reference.tzinfo)
    return parsed.timestamp()


def matching_events(log_text: str, pattern: str, reference_epoch: float) -> list[dict[str, Any]]:
    regex = re.compile(pattern)
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(log_text.splitlines(), start=1):
        if not regex.search(line):
            continue
        events.append(
            {
                "line_number": line_number,
                "line": line[-1000:],
                "wall_epoch": parse_timestamp(line, reference_epoch),
            }
        )
    return events


def startup_breakdown(log_text: str, started_wall: float, health_wall: float) -> dict[str, Any]:
    definitions = {
        "server_start": r"Starting server\.\.\.",
        "worker_initialized": r"Worker \d+: Initialized device and distributed environment\.",
        "model_loaded": r"Model runner: Model loaded successfully\.",
        "orchestrator_ready": r"Orchestrator ready with \d+ stages",
    }
    events = {key: matching_events(log_text, pattern, started_wall) for key, pattern in definitions.items()}
    missing = [key for key, values in events.items() if not values]
    timestamp_missing = [
        key for key, values in events.items() if values and not any(value["wall_epoch"] is not None for value in values)
    ]
    if missing or timestamp_missing:
        return {"valid": False, "missing": missing, "timestamp_missing": timestamp_missing, "events": events}
    server_start = min(value["wall_epoch"] for value in events["server_start"] if value["wall_epoch"] is not None)
    worker_latest = max(
        value["wall_epoch"] for value in events["worker_initialized"] if value["wall_epoch"] is not None
    )
    model_latest = max(value["wall_epoch"] for value in events["model_loaded"] if value["wall_epoch"] is not None)
    phases = {
        "imports_cli_config_s": server_start - started_wall,
        "workers_distributed_s": worker_latest - server_start,
        "model_init_load_placement_s": model_latest - worker_latest,
        "orchestrator_api_health_s": health_wall - model_latest,
    }
    total = health_wall - started_wall
    phase_sum = sum(phases.values())
    valid = all(value >= -1.0 for value in phases.values()) and abs(phase_sum - total) <= 0.01
    return {
        "valid": valid,
        "definition": "log-derived, one-second log timestamp resolution; final boundary is first observed /health 200",
        "phases": phases,
        "phase_sum_s": phase_sum,
        "wall_launch_to_health_s": total,
        "closure_error_s": phase_sum - total,
        "events": events,
    }


def summarize_resources(output_dir: Path) -> dict[str, Any]:
    gpu_peak: dict[int, int] = {}
    phase_util: dict[str, list[float]] = {}
    with (output_dir / "gpu_samples.csv").open() as handle:
        for row in csv.DictReader(handle):
            gpu = int(row["gpu"])
            gpu_peak[gpu] = max(gpu_peak.get(gpu, 0), int(row["memory_used_mib"]))
            phase_util.setdefault(row["phase"], []).append(float(row["utilization_pct"]))
    host_peak: dict[str, int] = {}
    with (output_dir / "host_samples.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            for key in ("rss_kb", "pss_kb", "private_dirty_kb", "shared_clean_kb", "locked_kb"):
                host_peak[key] = max(host_peak.get(key, 0), int(row[key]))
    return {
        "gpu_peak_mib": gpu_peak,
        "gpu_utilization_mean_pct_by_phase": {
            phase: statistics.fmean(values) for phase, values in phase_util.items() if values
        },
        "host_peak_kb": host_peak,
    }


def server_command(args: argparse.Namespace, profiled: bool) -> list[str]:
    command = [
        str(args.venv / "bin/python"),
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--omni",
        "--trust-remote-code",
        "--task-type",
        "fl2va",
        "--num-gpus",
        "4",
        "--usp",
        "4",
        "--ring",
        "1",
        "--text-encoder-tp-size",
        "4",
        "--vae-patch-parallel-size",
        "4",
        "--vae-parallel-mode",
        "tile",
        "--vae-use-tiling",
        "--diffusion-attention-backend",
        "CUDNN_ATTN",
        "--stage-init-timeout",
        "1800",
        "--init-timeout",
        "1800",
    ]
    if profiled:
        command.append("--enable-diffusion-pipeline-profiler")
    return command


def server_environment(args: argparse.Namespace, output_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(args.repo),
            "HF_HOME": "/data/models",
            "HF_HUB_OFFLINE": "1",
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": "1800",
            "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
            "TORCH_NCCL_ENABLE_MONITORING": "0",
            "VLLM_OMNI_STORAGE_PATH": str(output_dir / "storage"),
        }
    )
    return env


def stop_server(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=30)


def wait_gpu_baseline(selected: set[int], baseline: dict[int, int], timeout_s: float = 180.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        last = gpu_query(selected)
        if all(row["memory_used_mib"] <= baseline[row["index"]] + 100 for row in last):
            return {"cleared": True, "samples": last, "tolerance_mib": 100}
        time.sleep(2)
    return {"cleared": False, "samples": last, "tolerance_mib": 100}


def request_plan(profiled: bool) -> list[tuple[str, int, bool]]:
    if profiled:
        return [
            ("canonical_cold", 50, False),
            ("canonical_warm_1", 50, True),
            ("canonical_warm_2", 50, True),
            ("canonical_warm_3", 50, True),
            ("diagnostic_warmup", 4, False),
            ("diagnostic_warm_1", 4, True),
            ("diagnostic_warm_2", 4, True),
            ("diagnostic_warm_3", 4, True),
        ]
    return [
        ("control_cold", 50, False),
        ("control_warm_1", 50, True),
        ("control_warm_2", 50, True),
        ("control_warm_3", 50, True),
    ]


def attach_decomposition(request: dict[str, Any], cpu_encode_s: float | None, profiled: bool) -> None:
    request["cpu_encode_mux_s"] = cpu_encode_s
    if not profiled:
        request["decomposition"] = None
        return
    stages = request["stage_durations"]
    names = {
        "prompt": "MiniMaxH3Pipeline.encode_prompt",
        "denoise": "MiniMaxH3Pipeline.diffuse",
        "decode": "MiniMaxH3Pipeline.decode",
    }
    missing = [source for source in names.values() if source not in stages]
    if cpu_encode_s is None:
        missing.append("server-log CPU MP4 encode span")
    if missing:
        request["decomposition"] = {"valid": False, "missing": missing}
        return
    prompt = float(stages[names["prompt"]])
    denoise = float(stages[names["denoise"]])
    decode = float(stages[names["decode"]])
    server = float(request["server_inference_s"])
    client = float(request["client_total_s"])
    model = prompt + denoise + decode
    engine = server - model - float(cpu_encode_s)
    http = client - server
    accounted = prompt + denoise + decode + engine + float(cpu_encode_s) + http
    closure = accounted - client
    tolerance = max(0.05, client * 0.01)
    buckets = {
        "prompt_text_encoder_s": prompt,
        "denoise_s": denoise,
        "video_audio_vae_decode_s": decode,
        "engine_ipc_output_residual_s": engine,
        "cpu_encode_mux_s": float(cpu_encode_s),
        "http_transport_residual_s": http,
    }
    request["decomposition"] = {
        "valid": engine >= -tolerance and http >= -tolerance and abs(closure) <= tolerance,
        "buckets": buckets,
        "model_direct_s": model,
        "accounted_s": accounted,
        "client_total_s": client,
        "closure_error_s": closure,
        "tolerance_s": tolerance,
        "direct_buckets": ["prompt_text_encoder_s", "denoise_s", "video_audio_vae_decode_s", "cpu_encode_mux_s"],
        "residual_buckets": ["engine_ipc_output_residual_s", "http_transport_residual_s"],
    }


def run_condition(
    args: argparse.Namespace,
    condition: str,
    profiled: bool,
    selected: set[int],
    baseline: dict[int, int],
) -> dict[str, Any]:
    output_dir = args.output / "runs" / condition
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    assert_port_free(args.port)
    command = server_command(args, profiled)
    env = server_environment(args, output_dir)
    json_dump(
        output_dir / "launch.json",
        {
            "condition": condition,
            "profiled": profiled,
            "command": command,
            "cwd": str(args.repo),
            "environment": {
                key: env.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONPATH",
                    "HF_HOME",
                    "HF_HUB_OFFLINE",
                    "VLLM_WORKER_MULTIPROC_METHOD",
                    "VLLM_OMNI_VIDEO_SYNC_TIMEOUT",
                    "VLLM_ENGINE_READY_TIMEOUT_S",
                    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
                    "TORCH_NCCL_ENABLE_MONITORING",
                    "VLLM_OMNI_STORAGE_PATH",
                )
            },
            "request_plan": [
                {"label": label, "sigma_points": points, "denoiser_evaluations": points - 1, "measured": measured}
                for label, points, measured in request_plan(profiled)
            ],
        },
    )
    server_log_path = output_dir / "server.log"
    server_log = server_log_path.open("wb")
    started_wall = time.time()
    started_mono = time.monotonic()
    server = subprocess.Popen(
        command,
        cwd=args.repo,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    sampler = ResourceSampler(server.pid, selected, output_dir)
    sampler.start()
    requests_out: list[dict[str, Any]] = []
    failure: str | None = None
    health: dict[str, Any] | None = None
    cleanup: dict[str, Any] | None = None
    try:
        health = wait_for_health(server, f"http://127.0.0.1:{args.port}", started_mono, output_dir, args.startup_timeout)
        with requests.Session() as session:
            for label, sigma_points, measured in request_plan(profiled):
                sampler.phase = label
                print(f"[{condition}] request {label} ({sigma_points} sigma points) starting", flush=True)
                result = run_request(
                    session,
                    f"http://127.0.0.1:{args.port}",
                    output_dir,
                    label,
                    sigma_points,
                    measured,
                    args.request_timeout,
                )
                requests_out.append(result)
                print(
                    f"[{condition}] request {label} complete client={result['client_total_s']:.3f}s "
                    f"server={result['server_inference_s']:.3f}s valid_media={result['media']['valid']}",
                    flush=True,
                )
                sampler.phase = "between_requests"
                time.sleep(2)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sampler.phase = "shutdown"
        stop_server(server)
        sampler.stop_event.set()
        sampler.join(timeout=20)
        server_log.close()
        cleanup = wait_gpu_baseline(selected, baseline)
        log_text = server_log_path.read_text(errors="replace")
        cpu_spans = parse_cpu_encode_spans(log_text, len(requests_out))
        for request, cpu_span in zip(requests_out, cpu_spans, strict=True):
            attach_decomposition(request, cpu_span, profiled)
        started_health_wall = float(health["observed_health_wall_epoch"]) if health else time.time()
        result = {
            "condition": condition,
            "profiled": profiled,
            "started_wall_epoch": started_wall,
            "health": health,
            "startup_breakdown": startup_breakdown(log_text, started_wall, started_health_wall),
            "requests": requests_out,
            "resources": summarize_resources(output_dir),
            "sampler_error": sampler.error,
            "gpu_cleanup": cleanup,
            "resolved_attention_backends": sorted(
                set(re.findall(r"Resolved diffusion attention backend '([^']+)'", log_text))
            ),
            "failure": failure,
            "server_returncode": server.returncode,
        }
        json_dump(output_dir / "result.json", result)
    return result


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def measured_requests(condition: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    return [
        request
        for request in condition["requests"]
        if request["measured"] and request["label"].startswith(prefix)
    ]


def build_summary(profiled: dict[str, Any], unprofiled: dict[str, Any]) -> dict[str, Any]:
    canonical = measured_requests(profiled, "canonical_warm_")
    diagnostic = measured_requests(profiled, "diagnostic_warm_")
    control = measured_requests(unprofiled, "control_warm_")
    profiled_client = stats([float(request["client_total_s"]) for request in canonical])
    control_client = stats([float(request["client_total_s"]) for request in control])
    observer_overhead_pct = (
        (profiled_client["median"] / control_client["median"] - 1.0) * 100.0
        if control_client["median"]
        else None
    )
    checks = {
        "canonical_three_warm": len(canonical) == 3,
        "diagnostic_three_warm": len(diagnostic) == 3,
        "control_three_warm": len(control) == 3,
        "all_http_200": all(
            request["status"] == 200 for condition in (profiled, unprofiled) for request in condition["requests"]
        ),
        "all_media_valid": all(
            request["media"]["valid"] for condition in (profiled, unprofiled) for request in condition["requests"]
        ),
        "all_profiled_decompositions_valid": all(
            (request.get("decomposition") or {}).get("valid", False) for request in profiled["requests"]
        ),
        "startup_breakdowns_valid": all(
            condition["startup_breakdown"].get("valid", False) for condition in (profiled, unprofiled)
        ),
        "gpu_cleanup": all(condition["gpu_cleanup"].get("cleared", False) for condition in (profiled, unprofiled)),
        "samplers_clean": all(condition.get("sampler_error") is None for condition in (profiled, unprofiled)),
        "requested_backend_resolved": all(
            condition["resolved_attention_backends"] == ["CUDNN_ATTN"] for condition in (profiled, unprofiled)
        ),
    }
    return {
        "canonical_profiled_client_total_s": profiled_client,
        "canonical_profiled_server_inference_s": stats(
            [float(request["server_inference_s"]) for request in canonical]
        ),
        "canonical_profiled_buckets_s": {
            key: stats([float(request["decomposition"]["buckets"][key]) for request in canonical])
            for key in canonical[0]["decomposition"]["buckets"]
        },
        "diagnostic_profiled_client_total_s": stats(
            [float(request["client_total_s"]) for request in diagnostic]
        ),
        "diagnostic_profiled_buckets_s": {
            key: stats([float(request["decomposition"]["buckets"][key]) for request in diagnostic])
            for key in diagnostic[0]["decomposition"]["buckets"]
        },
        "canonical_unprofiled_client_total_s": control_client,
        "profiled_observer_overhead_pct_at_client_median": observer_overhead_pct,
        "profiled_observer_effect_exceeds_5pct": (
            observer_overhead_pct is not None and abs(observer_overhead_pct) > 5.0
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def environment_manifest(args: argparse.Namespace, selected: set[int]) -> dict[str, Any]:
    torch_probe = command_output(
        [
            str(args.venv / "bin/python"),
            "-c",
            (
                "import json,torch; print(json.dumps({"
                "'python':__import__('sys').version,'torch':torch.__version__,"
                "'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}))"
            ),
        ]
    )
    torch_environment = json.loads(torch_probe["stdout"]) if torch_probe["returncode"] == 0 else None
    return {
        "experiment": "MiniMax-H3 Blog 1 four-GPU startup and complete-response decomposition",
        "pinned_upstream_sha": PINNED_SHA,
        "model_path": str(args.model),
        "model_revision": MODEL_REVISION,
        "repo": str(args.repo),
        "venv": str(args.venv),
        "versions": {
            "vllm": importlib.metadata.version("vllm"),
            "vllm_omni": importlib.metadata.version("vllm-omni"),
            "harness_python": sys.version,
            "server_python_torch": torch_environment,
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "reserved_gpus": gpu_query(selected),
        "nvidia_topology": command_output(["nvidia-smi", "topo", "-m"]),
        "topology": {
            "task_partition": "FL2VA-only T2VA",
            "dit": "Ulysses 4, Ring 1, TP 1",
            "text_encoder_tp": 4,
            "vae_patch_parallel": 4,
            "vae_parallel_mode": "tile",
            "precision": "BF16 weights / shipped stage autocast behavior; no quantization",
            "attention_backend_requested": "CUDNN_ATTN",
            "execution": "regional torch.compile default; enforce_eager=false",
            "excluded_features": ["LoRA/Turbo", "quantization", "caching", "step execution", "offload"],
        },
        "workloads": {
            "canonical": {
                "task": "t2va",
                "prompt": PROMPT,
                "resolution": [1344, 768],
                "requested_duration_s": 5.0,
                "fps": 24,
                "expected_aligned_frames": 124,
                "sigma_points": 50,
                "denoiser_evaluations": 49,
                "seed": 1101,
            },
            "diagnostic": {
                "same_as": "canonical except schedule",
                "sigma_points": 4,
                "denoiser_evaluations": 3,
                "status": "non-canonical ordinary-path sensitivity control; not Turbo",
            },
        },
        "measurement": {
            "health_poll_interval_s": 0.5,
            "resource_sample_interval_s": 1.0,
            "page_cache_policy": "ambient/warm; caches are not dropped",
            "profiled": "one cold + three warm canonical; one diagnostic warmup + three measured diagnostic",
            "unprofiled": "one cold + three warm canonical controls",
            "startup": "one observation per server condition; illustrative, not a repeated headline metric",
        },
    }


def main() -> int:
    args = parse_args()
    require_pinned_checkout(args.repo)
    if not args.model.is_dir():
        raise RuntimeError(f"model path does not exist: {args.model}")
    if not (args.venv / "bin/python").is_file():
        raise RuntimeError(f"venv python does not exist: {args.venv / 'bin/python'}")
    if args.dry_run:
        print(json.dumps({"profiled": server_command(args, True), "unprofiled": server_command(args, False)}, indent=2))
        return 0
    selected_list = physical_gpu_ids()
    selected = set(selected_list)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact directory: {args.output}")
    args.output.mkdir(parents=True)
    baseline_rows = gpu_query(selected)
    baseline = {row["index"]: row["memory_used_mib"] for row in baseline_rows}
    manifest = environment_manifest(args, selected)
    manifest["started_wall_epoch"] = time.time()
    json_dump(args.output / "manifest.json", manifest)
    json_dump(
        args.output / "commands.json",
        {"profiled": server_command(args, True), "unprofiled": server_command(args, False)},
    )
    failure: str | None = None
    results: dict[str, Any] = {"manifest": str(args.output / "manifest.json")}
    try:
        profiled = run_condition(args, "profiled", True, selected, baseline)
        unprofiled = run_condition(args, "unprofiled", False, selected, baseline)
        results["profiled"] = profiled
        results["unprofiled"] = unprofiled
        results["summary"] = build_summary(profiled, unprofiled)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        results["failure"] = failure
        raise
    finally:
        results["completed_wall_epoch"] = time.time()
        results["failure"] = failure
        json_dump(args.output / "results.json", results)
    print(json.dumps(results["summary"], indent=2, sort_keys=True), flush=True)
    return 0 if results["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

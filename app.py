from __future__ import annotations

import asyncio
import json
import ipaddress
import socket
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse
from urllib.request import Request as URLRequest, urlopen
from urllib.error import HTTPError, URLError

import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import Field


try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

VERSION = "49.0.0-throughput-route"
DATA_ROOT = Path(os.getenv("CLIPSHORTENER_DATA", "/tmp/clipshortener"))
JOBS_ROOT = DATA_ROOT / "jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)
MAX_VIDEO_SIZE = int(os.getenv("CLIPSHORTENER_MAX_VIDEO_GB", "2")) * 1024 * 1024 * 1024
UPLOAD_CHUNK_SIZE = int(os.getenv("CLIPSHORTENER_UPLOAD_CHUNK_MB", "16")) * 1024 * 1024
MAX_UPLOAD_CHUNK_SIZE = 128 * 1024 * 1024
UPLOAD_FSYNC = os.getenv("CLIPSHORTENER_UPLOAD_FSYNC", "0").strip().lower() in {"1", "true", "yes", "on"}
MAX_BATCH_FILES = 20
MAX_CLIP_SECONDS = 900
JOB_TTL = 60 * 60 * 6
MAX_WORKERS = max(1, min(1, int(os.getenv("CLIPSHORTENER_WORKERS", "1"))))  # One active CPU-bound job at a time: the active job gets the whole CPU budget.
RATE_LIMIT = 20
RATE_WINDOW = 600
request_times: dict[str, list[float]] = {}
jobs: dict[str, dict[str, Any]] = {}
job_state_lock = threading.RLock()
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="clip-worker")
ffmpeg_slot = threading.Lock()
whisper_model_lock = threading.Lock()
upload_locks: dict[str, asyncio.Lock] = {}
whisper_model = None
# CPU-bound processing is deliberately single-flight. A single active encode gets
# the complete CPU budget instead of two encoders fighting over the same low-cost
# Render instance. Uploads, acquisition and status polling remain concurrent.

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("clipshortener")

app = FastAPI(title="ClipShortener API", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
if not FFMPEG or not FFPROBE:
    raise RuntimeError("FFmpeg and FFprobe are required.")

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".3gp", ".ts"}
PLATFORM_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com",
    "tiktok.com", "www.tiktok.com", "vm.tiktok.com",
    "instagram.com", "www.instagram.com", "facebook.com", "www.facebook.com", "fb.watch",
    "x.com", "www.x.com", "twitter.com", "www.twitter.com",
    "reddit.com", "www.reddit.com", "old.reddit.com",
    "pinterest.com", "www.pinterest.com", "pin.it",
    "snapchat.com", "www.snapchat.com",
    "soundcloud.com", "www.soundcloud.com", "vk.com", "www.vk.com",
}
MAX_URL_REDIRECTS = 5
URL_READ_CHUNK = 4 * 1024 * 1024
ACQUISITION_TIMEOUT = int(os.getenv("CLIPSHORTENER_ACQUISITION_TIMEOUT", "180"))
ACQUISITION_API_URL = os.getenv("ACQUISITION_API_URL", "").strip().rstrip("/")
def _source_fps(info: dict[str, Any]) -> float | None:
    """Return a reliable constant frame rate when the source exposes one."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = str(info.get(key) or "").strip()
        if not raw or raw in {"0/0", "0"}:
            continue
        try:
            if "/" in raw:
                n, d = raw.split("/", 1)
                value = float(n) / float(d)
            else:
                value = float(raw)
            if 1.0 <= value <= 240.0:
                return value
        except (ValueError, ZeroDivisionError):
            continue
    return None

def _fixed_gop_args(info: dict[str, Any], clip_length: float, total_duration: float) -> list[str]:
    """Use deterministic clip-aligned GOPs when the source is CFR.

    This removes the per-frame force-keyframe expression from the hot path.
    The segment muxer can then cut directly on the encoder's known keyframes.
    VFR/unknown-rate inputs retain the older force-keyframe route for safety.
    """
    fps = _source_fps(info)
    if fps is None or clip_length <= 0.05:
        return []
    frames = max(1, int(round(fps * clip_length)))
    # Do not create a pathological GOP for tiny clip lengths.
    if frames < 2:
        return []
    return ["-g", str(frames), "-keyint_min", str(frames), "-sc_threshold", "0"]

def _effective_cpu_threads() -> int:
    """Choose FFmpeg threads from Render's assigned CPU count first.

    Render exposes the service CPU allocation as RENDER_CPU_COUNT.  That is a
    better signal than the host/cgroup CPU view for deciding how much parallel
    work a single x264 encode should request.
    """
    raw = os.getenv("CLIPSHORTENER_FFMPEG_THREADS", "auto").strip().lower()
    if raw not in {"", "auto", "0"}:
        try:
            return max(1, min(8, int(raw)))
        except ValueError:
            pass
    render_cpu = os.getenv("RENDER_CPU_COUNT", "").strip()
    if render_cpu:
        try:
            cpu = float(render_cpu)
            if cpu > 0:
                return max(1, min(8, int(math.ceil(cpu))))
        except ValueError:
            pass
    quota = None
    try:
        text = Path("/sys/fs/cgroup/cpu.max").read_text().strip().split()
        if len(text) >= 2 and text[0] != "max":
            quota = float(text[0]) / max(1.0, float(text[1]))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    if quota is None:
        try:
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
            if q > 0 and period > 0:
                quota = q / period
        except (OSError, ValueError, ZeroDivisionError):
            pass
    if quota is None:
        return 0
    return max(1, min(8, int(math.ceil(quota))))

FFMPEG_THREADS = _effective_cpu_threads()
FFMPEG_FILTER_THREADS = max(1, min(4, int(os.getenv("CLIPSHORTENER_FILTER_THREADS", str(FFMPEG_THREADS or 1)))))
FFMPEG_PRESET = os.getenv("CLIPSHORTENER_FFMPEG_PRESET", "ultrafast").strip() or "ultrafast"
# Do not force hand-written x264 micro-settings by default. Benchmarking the
# real workload showed that libx264's own ultrafast preset is faster than the
# previous custom parameter bundle on this workload. Users can still override
# the advanced settings explicitly through the environment if needed.
FFMPEG_TUNE = os.getenv("CLIPSHORTENER_FFMPEG_TUNE", "").strip()
FFMPEG_X264_PARAMS = os.getenv("CLIPSHORTENER_X264_PARAMS", "").strip()
CRF_ORIGINAL = int(os.getenv("CLIPSHORTENER_CRF_ORIGINAL", "21"))
CRF_STANDARD = int(os.getenv("CLIPSHORTENER_CRF_STANDARD", "23"))
FFMPEG_PROGRESS_PERIOD = os.getenv("CLIPSHORTENER_PROGRESS_PERIOD", "2")
VALIDATE_OUTPUT_DURATIONS = os.getenv("CLIPSHORTENER_VALIDATE_OUTPUT_DURATIONS", "0").strip().lower() in {"1", "true", "yes", "on"}
NO_UPSCALE = os.getenv("CLIPSHORTENER_NO_UPSCALE", "1").strip().lower() in {"1", "true", "yes", "on"}

def _host_is_platform(host: str) -> bool:
    host = (host or "").lower().rstrip(".")
    return any(host == item or host.endswith("." + item) for item in PLATFORM_HOSTS)

def _host_is_public(host: str) -> bool:
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if not ip.is_global:
            return False
    return True

def _validate_source_url(value: str) -> tuple[str, bool]:
    value = (value or "").strip()
    if not value:
        raise HTTPException(400, "Video URL is required.")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(400, "Enter a valid HTTP or HTTPS video URL.")
    if _host_is_platform(parsed.hostname):
        return value, True
    if not _host_is_public(parsed.hostname):
        raise HTTPException(400, "The video URL must point to a public internet host.")
    return value, False


def _validate_public_url(value: str) -> str:
    value, is_platform = _validate_source_url(value)
    if is_platform:
        raise HTTPException(400, "Platform acquisition is handled separately. Use a configured supported platform link.")
    return value


def _acquisition_request(url: str) -> dict[str, Any]:
    """Ask the separate acquisition service to fetch a direct public media URL."""
    if not ACQUISITION_API_URL:
        raise RuntimeError(
            "URL acquisition is not configured. Set ACQUISITION_API_URL on the processing server."
        )
    endpoint = ACQUISITION_API_URL + "/acquire"
    payload = json.dumps({"url": url}).encode("utf-8")
    request = URLRequest(
        endpoint,
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ClipShortener/27.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=ACQUISITION_TIMEOUT) as response:
            raw = response.read(1024 * 1024)
            status = getattr(response, "status", 200)
    except HTTPError as exc:
        try:
            raw = exc.read(1024 * 1024)
            detail = raw.decode("utf-8", "replace")
        finally:
            exc.close()
        raise RuntimeError(
            f"Acquisition service returned HTTP {exc.code}: {detail[:300]}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Could not reach the URL acquisition service.") from exc

    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Acquisition service returned invalid JSON (HTTP {status})."
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError("Acquisition service returned an invalid response.")
    if not data.get("download_url"):
        raise RuntimeError(data.get("error") or "Acquisition service returned no media file.")
    return data


def _acquire_via_service(job_id: str, url: str) -> tuple[Path, str]:
    parsed = urlparse(url)
    if _host_is_platform(parsed.hostname or ""):
        raise RuntimeError(
            "This link is a platform page. ClipShortener currently accepts direct public video-file URLs here."
        )

    result = _acquisition_request(url)
    download_url = str(result["download_url"])
    if download_url.startswith("/"):
        download_url = urljoin(ACQUISITION_API_URL + "/", download_url.lstrip("/"))
    filename = safe_filename(str(result.get("filename") or "imported-video.mp4"))

    download_parsed = urlparse(download_url)
    if download_parsed.scheme not in {"http", "https"} or not download_parsed.hostname:
        raise RuntimeError("The acquisition service returned an invalid download URL.")
    if not _host_is_public(download_parsed.hostname):
        raise RuntimeError("The acquisition service returned an unsafe download URL.")

    dest = JOBS_ROOT / job_id
    dest.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        suffix = ".mp4"
        filename = Path(filename).stem + suffix
    source = dest / f"input{suffix}"

    req = URLRequest(
        download_url,
        headers={
            "User-Agent": "ClipShortener/27.0",
            "Accept": "video/*,application/octet-stream;q=0.9,*/*;q=0.1",
        },
    )
    try:
        with urlopen(req, timeout=ACQUISITION_TIMEOUT) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > MAX_VIDEO_SIZE:
                        raise RuntimeError("The acquired video exceeds the configured upload limit.")
                except ValueError:
                    pass
            total = 0
            with source.open("wb") as target:
                while True:
                    chunk = response.read(URL_READ_CHUNK)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_VIDEO_SIZE:
                        source.unlink(missing_ok=True)
                        raise RuntimeError("The acquired video exceeds the configured upload limit.")
                    target.write(chunk)
    except HTTPError as exc:
        source.unlink(missing_ok=True)
        raise RuntimeError(f"The acquisition download returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        source.unlink(missing_ok=True)
        raise RuntimeError("Could not download the acquired video.") from exc

    if total <= 0:
        source.unlink(missing_ok=True)
        raise RuntimeError("The acquisition service returned an empty video file.")

    log.info(
        "DIRECT_URL_IMPORT_SUCCESS job=%s service=%s size=%s filename=%s",
        job_id,
        urlparse(ACQUISITION_API_URL).hostname,
        total,
        filename,
    )
    return source, filename


def _acquire_video(job_id: str, url: str) -> tuple[Path, str]:
    value, is_platform = _validate_source_url(url)
    if is_platform:
        raise RuntimeError("Platform-page links are not supported. Use a direct public video-file URL.")
    if ACQUISITION_API_URL:
        return _acquire_via_service(job_id, value)
    return _acquire_public_video(job_id, value)


def _acquire_public_video(job_id: str, url: str) -> tuple[Path, str]:
    current = _validate_public_url(url)
    dest = JOBS_ROOT / job_id
    dest.mkdir(parents=True, exist_ok=True)
    for _ in range(MAX_URL_REDIRECTS + 1):
        parsed = urlparse(current)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or not _host_is_public(parsed.hostname):
            raise RuntimeError("The video URL or redirect target is not a public HTTP(S) host.")
        req = URLRequest(current, headers={"User-Agent": "ClipShortener/27.0", "Accept": "video/*,application/octet-stream;q=0.9,*/*;q=0.1"})
        try:
            response = urlopen(req, timeout=30)
        except HTTPError as exc:
            if exc.code in {301,302,303,307,308}:
                location = exc.headers.get("Location")
                exc.close()
                if not location:
                    raise RuntimeError("The video server returned an invalid redirect.")
                current = urljoin(current, location)
                continue
            raise RuntimeError(f"The video server returned HTTP {exc.code}.") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("Could not connect to the video URL.") from exc
        break
    else:
        raise RuntimeError("Too many redirects while opening the video URL.")

    with response:
        content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].lower()
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > MAX_VIDEO_SIZE:
                    raise RuntimeError("The video exceeds the configured upload limit.")
            except ValueError:
                pass
        suffix = Path(urlparse(current).path).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            suffix = ".mp4" if (content_type.startswith("video/") or "octet-stream" in content_type) else ".mp4"
        source = dest / f"input{suffix}"
        total = 0
        with source.open("wb") as target:
            while True:
                chunk = response.read(URL_READ_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_VIDEO_SIZE:
                    source.unlink(missing_ok=True)
                    raise RuntimeError("The video exceeds the configured upload limit.")
                target.write(chunk)
        if total <= 0:
            source.unlink(missing_ok=True)
            raise RuntimeError("The video URL returned an empty file.")
    name = safe_filename(Path(urlparse(current).path).name or "imported-video.mp4")
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        name = Path(name).stem + suffix
    log.info("PUBLIC_URL_IMPORT_SUCCESS job=%s size=%s filename=%s", job_id, total, name)
    return source, name


def now() -> float:
    return time.time()


def safe_filename(name: str) -> str:
    clean = Path(name or "video.mp4").name
    return clean if clean not in {"", ".", ".."} else "video.mp4"


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    t = now()
    values = [x for x in request_times.get(ip, []) if t - x < RATE_WINDOW]
    if len(values) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests. Please try again later.")
    values.append(t)
    request_times[ip] = values


def _job_state_path(job_id: str) -> Path:
    return JOBS_ROOT / job_id / "job.json"


def _persist_job(job_id: str, job: dict[str, Any]) -> None:
    path = _job_state_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def update_job(job_id: str, **updates: Any) -> None:
    with job_state_lock:
        job = jobs.setdefault(job_id, {})
        job.update(updates)
        job["updated_at"] = now()
        _persist_job(job_id, job)


def load_persisted_jobs() -> None:
    with job_state_lock:
        for path in JOBS_ROOT.glob("*/job.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job_id = path.parent.name
                if re.fullmatch(r"[0-9a-fA-F]{32}", job_id) and isinstance(data, dict):
                    jobs[job_id] = data
            except Exception:
                log.warning("Could not restore job state from %s", path)


def cleanup_old_jobs() -> None:
    # Never remove active jobs.  Also use the job-state file timestamp rather
    # than the directory mtime: updating job.json does not reliably update the
    # parent directory mtime, which could otherwise make a long-running job
    # look stale.
    cutoff = now() - JOB_TTL
    for path in list(JOBS_ROOT.iterdir()):
        try:
            if not path.is_dir():
                continue
            state_path = path / "job.json"
            meta_path = path / "upload.json"
            if state_path.is_file():
                mtime = state_path.stat().st_mtime
            elif meta_path.is_file():
                mtime = meta_path.stat().st_mtime
            else:
                mtime = path.stat().st_mtime
            if mtime >= cutoff:
                continue
            job_id = path.name
            state = jobs.get(job_id)
            # Queued/processing jobs are never garbage-collected merely because
            # a request happened to arrive after the TTL boundary.
            if state and state.get("status") in {"queued", "processing"}:
                continue
            shutil.rmtree(path, ignore_errors=True)
            jobs.pop(job_id, None)
        except OSError:
            pass


def run_checked(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    log.info("CMD %s", " ".join(map(str, cmd)))
    # Every FFmpeg invocation shares the same CPU slot. This prevents a lazy
    # thumbnail request (or another FFmpeg utility call) from stealing CPU
    # cycles from the active clip job. Non-FFmpeg commands remain unconstrained.
    lock = ffmpeg_slot if cmd and str(cmd[0]) == str(FFMPEG) else nullcontext()
    try:
        with lock:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Processing timed out. Try a shorter or smaller video.") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "FFmpeg failed")[-5000:]
        log.error("COMMAND_FAILED %s", detail)
        raise RuntimeError("Video processing failed. The file may use an unsupported codec or container.")
    return result


def run_ffmpeg_progress(
    cmd: list[str],
    job_id: str,
    total_seconds: float,
    progress_start: int = 20,
    progress_span: int = 55,
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    """Run FFmpeg while publishing real encode progress to the job state."""
    log.info("CMD %s", " ".join(map(str, cmd)))
    full_cmd = [*cmd, "-stats_period", FFMPEG_PROGRESS_PERIOD, "-progress", "pipe:1", "-nostats"]
    started = time.monotonic()
    try:
        with ffmpeg_slot:
            try:
                proc = subprocess.Popen(
                    full_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                raise RuntimeError("FFmpeg could not be started.") from exc

            last_progress = -1
            stdout_lines: list[str] = []
            try:
                assert proc.stdout is not None
                assert proc.stderr is not None
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    stdout_lines.append(line)
                    if line.startswith("out_time_ms="):
                        try:
                            out_seconds = max(0.0, int(line.split("=", 1)[1]) / 1_000_000)
                            ratio = min(1.0, out_seconds / max(0.05, total_seconds))
                            current = min(
                                progress_start + progress_span - 1,
                                progress_start + int(progress_span * ratio),
                            )
                            # Progress is intentionally coarse. Persisting job.json on every
                            # percentage point can become surprisingly expensive on a fast
                            # encoder, especially with many short videos. The UI still gets
                            # smooth-enough progress while disk I/O stays out of the encode path.
                            if current != last_progress and (current - last_progress >= 2 or current >= progress_start + progress_span - 1):
                                last_progress = current
                                update_job(job_id, progress=current, message=f"Encoding clips... {current}%")
                        except (ValueError, TypeError):
                            pass
                    if time.monotonic() - started > timeout:
                        proc.kill()
                        proc.wait()
                        raise RuntimeError("Processing timed out. Try a shorter or smaller video.")

                stderr = proc.stderr.read()
                returncode = proc.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                proc.wait()
                raise RuntimeError("Processing timed out. Try a shorter or smaller video.") from exc
            except Exception:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                raise

            if returncode != 0:
                detail = (stderr or "\n".join(stdout_lines) or "FFmpeg failed")[-5000:]
                log.error("COMMAND_FAILED %s", detail)
                raise RuntimeError("Video processing failed. The file may use an unsupported codec or container.")
            return subprocess.CompletedProcess(full_cmd, returncode, "\n".join(stdout_lines), stderr)
    except RuntimeError:
        raise

def probe(path: Path) -> dict[str, Any]:
    result = run_checked([
        FFPROBE, "-v", "error", "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,codec_long_name,pix_fmt,r_frame_rate,avg_frame_rate",
        "-of", "json", str(path),
    ], timeout=60)
    data = json.loads(result.stdout or "{}")
    fmt = data.get("format", {})
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    duration = float(fmt.get("duration") or 0)
    size = int(fmt.get("size") or path.stat().st_size)
    if duration <= 0.05 or size <= 0:
        raise RuntimeError("The video could not be read.")
    return {
        "duration": duration,
        "size": size,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "codec": video.get("codec_name") or "unknown",
        "audio_codec": audio.get("codec_name") or "unknown",
        "pix_fmt": video.get("pix_fmt") or "unknown",
        "r_frame_rate": video.get("r_frame_rate") or "0/0",
        "avg_frame_rate": video.get("avg_frame_rate") or "0/0",
    }


def parse_number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_export_options(export_quality: Any = "original", export_ratio: Any = "source", legacy_export: Any = "original") -> tuple[str, str]:
    """Normalize the new independent quality/ratio model while accepting the old export_format field."""
    q = str(export_quality or "original").strip().lower()
    r = str(export_ratio or "source").strip().lower()
    legacy = str(legacy_export or "original").strip().lower()
    legacy_map = {
        "original": ("original", "source"),
        "720p": ("720", "16:9"),
        "1080p": ("1080", "16:9"),
        "vertical": ("1080", "9:16"),
        "square": ("1080", "1:1"),
        "landscape": ("1080", "16:9"),
    }
    if q == "original" and r == "source" and legacy in legacy_map and legacy != "original":
        q, r = legacy_map[legacy]
    if q in {"720p", "720"}: q = "720"
    elif q in {"1080p", "1080"}: q = "1080"
    elif q in {"original", "source"}: q = "original"
    else: raise HTTPException(400, "Unsupported export quality.")
    ratio_alias = {"original":"source", "source":"source", "16/9":"16:9", "9/16":"9:16", "square":"1:1", "1/1":"1:1"}
    r = ratio_alias.get(r, r)
    if r not in {"source", "16:9", "9:16", "1:1"}:
        raise HTTPException(400, "Unsupported export ratio.")
    return q, r

def validate_options(clip_length: Any, export_format: str = "original", start: Any = 0, end: Any = 0, export_quality: Any = "original", export_ratio: Any = "source") -> tuple[float, str, float, float | None]:
    length = parse_number(clip_length, 30)
    if not 1 <= length <= MAX_CLIP_SECONDS:
        raise HTTPException(400, "Clip length must be between 1 and 900 seconds.")
    quality, ratio = _normalize_export_options(export_quality, export_ratio, export_format)
    export = f"{quality}:{ratio}"
    start_value = max(0.0, parse_number(start, 0))
    end_num = parse_number(end, 0)
    end_value = None if end_num <= 0 else max(0.1, end_num)
    return length, export, start_value, end_value

def split_export_spec(export_format: str) -> tuple[str, str]:
    value = str(export_format or "original").strip().lower()
    if ":" in value:
        q, r = value.split(":", 1)
        return _normalize_export_options(q, r, "original")
    return _normalize_export_options("original", "source", value)

def _target_dimensions(info: dict[str, Any], export_format: str) -> tuple[int, int] | None:
    """Return the cheapest valid output dimensions for the requested preset.

    The important optimization here is *source-class aware output routing*. A
    1280x720 source selected for a 1080/9:16 export is not promoted to a
    1080x1920 encode. It is routed to the 720-class 720x1280 path instead,
    avoiding a roughly 4x output-pixel increase while retaining a useful
    standard vertical canvas.
    """
    quality, ratio = split_export_spec(export_format)
    sw, sh = int(info.get("width") or 0), int(info.get("height") or 0)
    if sw <= 0 or sh <= 0:
        return None
    sw_even, sh_even = sw - sw % 2, sh - sh % 2

    if quality == "original" and ratio == "source":
        return None

    # Work out the largest even-sized crop that matches the requested ratio.
    if ratio == "source":
        crop_w, crop_h = sw_even, sh_even
    else:
        target_aspect = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}[ratio]
        source_aspect = sw_even / max(1, sh_even)
        if abs(source_aspect - target_aspect) < 0.001:
            crop_w, crop_h = sw_even, sh_even
        elif source_aspect > target_aspect:
            crop_h = sh_even
            crop_w = max(2, int(crop_h * target_aspect) // 2 * 2)
        else:
            crop_w = sw_even
            crop_h = max(2, int(crop_w / target_aspect) // 2 * 2)

    # Original quality means preserve source resolution and only crop.
    if quality == "original":
        return max(2, crop_w), max(2, crop_h)

    base = 720 if quality == "720" else 1080
    if ratio == "16:9":
        requested = (1280, 720) if base == 720 else (1920, 1080)
    elif ratio == "9:16":
        requested = (720, 1280) if base == 720 else (1080, 1920)
    else:
        requested = (720, 720) if base == 720 else (1080, 1080)

    if NO_UPSCALE:
        # Treat the quality selector as a maximum output class. A 1280x720
        # landscape source therefore becomes a 720-class vertical export when
        # 9:16 is requested, rather than being promoted to a 1080-class encode.
        # This is the useful speed/quality boundary for low-resolution sources:
        # it avoids a 4x jump in encoded pixels while still producing a standard
        # 720x1280 vertical canvas instead of an awkward 404x720 crop.
        source_short_edge = min(sw_even, sh_even)
        if base == 1080 and source_short_edge < 1000:
            base = 720
            if ratio == "16:9":
                requested = (1280, 720)
            elif ratio == "9:16":
                requested = (720, 1280)
            else:
                requested = (720, 720)
    return requested

def video_filter(export_format: str, info: dict[str, Any] | None = None) -> str | None:
    """Build the cheapest correct resize/crop filter.

    The old filter scaled a landscape source UP to the vertical canvas and
    then threw most of those pixels away. For example, 1920x1080 -> 720x1280
    could first create roughly 2276x1280 pixels before cropping. Crop to the
    requested aspect ratio at source resolution first, then scale DOWN. This
    materially reduces decoder/filter/encoder work while producing the same
    requested canvas.
    """
    info = info or {}
    quality, ratio = split_export_spec(export_format)
    sw, sh = int(info.get("width") or 0), int(info.get("height") or 0)
    target = _target_dimensions(info, export_format)
    if target is None or sw <= 0 or sh <= 0:
        return None
    tw, th = target
    sw_even, sh_even = sw - sw % 2, sh - sh % 2
    if ratio == "source":
        if (tw, th) == (sw_even, sh_even):
            return None
        return f"scale={tw}:{th}:flags=fast_bilinear"

    target_aspect = tw / th
    source_aspect = sw / sh
    if abs(source_aspect - target_aspect) < 0.001:
        crop_w, crop_h = sw_even, sh_even
    elif source_aspect > target_aspect:
        crop_h = sh_even
        crop_w = max(2, int(crop_h * target_aspect) // 2 * 2)
    else:
        crop_w = sw_even
        crop_h = max(2, int(crop_w / target_aspect) // 2 * 2)

    crop_needed = crop_w != sw_even or crop_h != sh_even
    parts = []
    if crop_needed:
        parts.append(f"crop={crop_w}:{crop_h}:(iw-{crop_w})/2:(ih-{crop_h})/2")
    if crop_w != tw or crop_h != th:
        parts.append(f"scale={tw}:{th}:flags=fast_bilinear")
    return ",".join(parts) or None


def make_thumbnail(video: Path, destination: Path) -> None:
    # Fast preview generation: seek before input so long clips do not require
    # decoding from the beginning. If the source does not support the seek
    # cleanly, fall back to the first frame.
    try:
        run_checked([
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", "0.05",
            "-i", str(video), "-frames:v", "1", "-vf", "scale=640:-2:flags=fast_bilinear",
            "-q:v", "5", str(destination),
        ], timeout=30)
    except Exception:
        destination.unlink(missing_ok=True)
        run_checked([
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-frames:v", "1", "-vf", "scale=640:-2:flags=fast_bilinear",
            "-q:v", "5", str(destination),
        ], timeout=60)


def ass_escape(text: str) -> str:
    return (text or "").replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def caption_style(style: str, font: str, size: str, position: str, animation: str, canvas_w: int, canvas_h: int) -> dict[str, Any]:
    base = max(76, min(120, round(min(canvas_w, canvas_h) * 0.115)))
    sizes = {"medium": round(base * 0.82), "large": base, "xlarge": round(base * 1.22)}
    pos = {"bottom": (2, max(70, round(canvas_h * 0.11))), "middle": (5, 0), "top": (8, max(70, round(canvas_h * 0.11)))}
    align, margin_v = pos.get(position, pos["bottom"])
    boxed = style == "boxed"
    return {
        "font": font if font in {"DejaVu Sans", "Liberation Sans", "DejaVu Sans Mono"} else "DejaVu Sans",
        "size": sizes.get(size, sizes["large"]),
        "bold": 1 if style in {"bold", "classic", "boxed"} else 0,
        "outline": max(3, round(sizes.get(size, sizes["large"]) * 0.075)),
        "shadow": 2 if not boxed else 0,
        "back": "&HCC000000" if boxed else "&H00000000",
        "border_style": 3 if boxed else 1,
        "align": align,
        "margin_v": margin_v,
        "margin_lr": max(50, round(canvas_w * 0.075)),
        "tags": r"\fad(120,120)" if animation == "fade" else (r"\fscx92\fscy92\t(0,120,\fscx100\fscy100)" if animation == "pop" else ""),
    }

def _caption_segments(video: Path, language: str, clip_start: float = 0.0, clip_end: float | None = None) -> list[tuple[float, float, str]]:
    """Transcribe only the requested export window and cache one tiny model."""
    global whisper_model
    if WhisperModel is None:
        raise RuntimeError("Automatic captions are unavailable in this deployment.")

    requested_language = (language or "auto").strip().lower()
    model_name = "tiny.en" if requested_language == "en" else "tiny"
    if whisper_model is None or getattr(whisper_model, "_clipshortener_model_name", None) != model_name:
        with whisper_model_lock:
            if whisper_model is None or getattr(whisper_model, "_clipshortener_model_name", None) != model_name:
                log.info("Loading faster-whisper %s model (cached worker model)", model_name)
                whisper_model = WhisperModel(
                    model_name, device="cpu", compute_type="int8",
                    download_root=os.getenv("HF_HOME", "/opt/huggingface"),
                )
                try:
                    setattr(whisper_model, "_clipshortener_model_name", model_name)
                except Exception:
                    pass

    kwargs: dict[str, Any] = {
        "beam_size": 1,
        "vad_filter": True,
        "language": None if requested_language in {"", "auto"} else requested_language,
        "condition_on_previous_text": False,
        "word_timestamps": False,
    }
    if clip_end is not None and clip_end > clip_start + 0.05:
        kwargs["clip_timestamps"] = [float(clip_start), float(clip_end)]

    try:
        segments, _ = whisper_model.transcribe(str(video), **kwargs)
    except TypeError:
        kwargs.pop("clip_timestamps", None)
        segments, _ = whisper_model.transcribe(str(video), **kwargs)

    out: list[tuple[float, float, str]] = []
    for seg in segments:
        text = " ".join((seg.text or "").strip().split())
        if not text or float(seg.end) <= float(seg.start):
            continue
        st, en = float(seg.start), float(seg.end)
        if clip_end is not None and clip_start > 0.05 and en <= (clip_end - clip_start + 0.5):
            st += clip_start
            en += clip_start
        out.append((st, en, text))
    if not out:
        raise RuntimeError("No speech was detected for automatic captions.")
    return out

def _wrap_caption(text: str, max_chars: int = 34) -> str:
    words = text.split(); lines=[]; current=""
    for word in words:
        if not current: current=word
        elif len(current)+1+len(word) <= max_chars: current += " " + word
        else: lines.append(current); current=word
    if current: lines.append(current)
    return "\n".join(lines[:2])

def caption_tracks_from_segments(segments, clip_start, clip_duration, style, font, size, animation, position, ass, vtt, canvas_w=1920, canvas_h=1080) -> None:
    settings = caption_style(style, font, size, position, animation, canvas_w, canvas_h)
    def ass_time(seconds):
        cs=int(round(max(0, seconds)*100)); h,cs=divmod(cs,360000); m,cs=divmod(cs,6000); sec,cs=divmod(cs,100); return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"
    def vtt_time(seconds):
        total=max(0,seconds); h=int(total//3600); total-=h*3600; m=int(total//60); total-=m*60; sec=int(total); ms=int(round((total-sec)*1000));
        if ms>=1000: sec+=1; ms-=1000
        return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"
    count=0; clip_end=clip_start+clip_duration
    with ass.open("w",encoding="utf-8") as af, vtt.open("w",encoding="utf-8") as vfout:
        af.write(f"[Script Info]\nScriptType: v4.00+\nPlayResX: {canvas_w}\nPlayResY: {canvas_h}\nWrapStyle: 2\n\n")
        af.write("[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n")
        af.write(f"Style,Default,{settings['font']},{settings['size']},&H00FFFFFF,&H00FFFFFF,&H00000000,{settings['back']},{settings['bold']},0,0,0,100,100,0,0,{settings['border_style']},{settings['outline']},{settings['shadow']},{settings['align']},{settings['margin_lr']},{settings['margin_lr']},{settings['margin_v']},1\n\n")
        af.write("[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
        # Re-open VTT header correctly; ASS and VTT are separate files, so write VTT independently below.
    with ass.open("a",encoding="utf-8") as af, vtt.open("w",encoding="utf-8") as vfout:
        vfout.write("WEBVTT\n\n")
        for st,en,text in segments:
            if en<=clip_start or st>=clip_end: continue
            ls=max(0.0,st-clip_start); le=min(clip_duration,en-clip_start)
            if le<=ls: continue
            wrapped=_wrap_caption(text, 34 if canvas_w>=1000 else 24)
            if position == "top":
                placement = f"\\an8\\pos({canvas_w//2},{settings['margin_v']})"
            elif position == "middle":
                placement = f"\\an5\\pos({canvas_w//2},{canvas_h//2})"
            else:
                placement = f"\\an2\\pos({canvas_w//2},{canvas_h-settings['margin_v']})"
            tags = placement + settings['tags']
            af.write(f"Dialogue: 0,{ass_time(ls)},{ass_time(le)},Default,,0,0,0,,{{{tags}}}{ass_escape(wrapped)}\n")
            vfout.write(f"{vtt_time(ls)} --> {vtt_time(le)}\n{text}\n\n")
            count+=1
    if count==0: ass.unlink(missing_ok=True); vtt.unlink(missing_ok=True)

def burn_captions(video: Path, ass_path: Path, output: Path, job_id: str | None=None, duration: float | None=None) -> None:
    escaped=str(ass_path).replace("\\","/").replace(":",r"\:")
    cmd=[FFMPEG,"-nostdin","-hide_banner","-loglevel","error","-y","-i",str(video),"-vf",f"subtitles='{escaped}'","-map","0:v:0","-map","0:a?","-c:v","libx264","-preset","ultrafast","-crf","21","-threads",str(FFMPEG_THREADS),"-c:a","aac","-b:a","128k",str(output)]
    if job_id and duration: run_ffmpeg_progress(cmd,job_id,duration,70,25,max(900,int(duration*8)))
    else: run_checked(cmd,timeout=1800)


def _durations_are_consistent(clips: list[Path], clip_length: float, total_duration: float, tolerance: float = 0.20) -> bool:
    """Validate output cheaply; optionally sample timestamps when explicitly enabled.

    The encoder already forces segment boundaries. Running ffprobe three extra times
    after every export adds latency without changing the generated media. The default
    path therefore checks count/non-empty files only. Set
    CLIPSHORTENER_VALIDATE_OUTPUT_DURATIONS=1 when diagnostic timestamp probing is wanted.
    """
    if not clips:
        return False
    expected = max(1, math.ceil((total_duration - 1e-6) / clip_length))
    if len(clips) != expected:
        return False
    if any(not p.is_file() or p.stat().st_size <= 0 for p in clips):
        return False
    if not VALIDATE_OUTPUT_DURATIONS:
        return True

    sample_indexes = sorted({0, len(clips) // 2, len(clips) - 1})
    for index in sample_indexes:
        try:
            d = probe(clips[index])["duration"]
        except Exception:
            return False
        if index < len(clips) - 1 and abs(d - clip_length) > tolerance:
            return False
        if index == len(clips) - 1:
            expected_last = clip_length if total_duration >= expected * clip_length - tolerance else total_duration - clip_length * (expected - 1)
            if expected_last > tolerance and abs(d - expected_last) > max(tolerance, 0.35):
                return False
    return True


def export_matches_source(info: dict[str, Any], export_format: str) -> bool:
    return _target_dimensions(info, export_format) in {None, (int(info.get("width") or 0), int(info.get("height") or 0))}

def _vtt_timestamp(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    total -= hours * 3600
    minutes = int(total // 60)
    total -= minutes * 60
    secs = int(total)
    millis = int(round((total - secs) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _clip_ranges(total_duration: float, clip_length: float) -> list[tuple[float, float]]:
    """Return full requested-duration clips plus one exact remainder when needed."""
    if total_duration <= 0 or clip_length <= 0:
        return []
    full_count = int(total_duration // clip_length)
    remainder = total_duration - (full_count * clip_length)
    ranges = [(i * clip_length, clip_length) for i in range(full_count)]
    if remainder > 0.05:
        ranges.append((full_count * clip_length, remainder))
    if not ranges:
        ranges.append((0.0, total_duration))
    return ranges


def create_clips(job_id, source, clip_length, export_format, start, end, captions, info=None):
    """Create correctly timed clips with the fewest possible encode passes.

    Fast path: stream-copy when no video transformation/captions are required.
    Accurate path: one FFmpeg pass that performs any crop/scale and segmentation.
    Caption path: transcribe once, burn captions in that same single FFmpeg pass,
    and segment the already-rendered frames directly. Never re-encode each clip.
    """
    info = info or probe(source)
    duration = float(info["duration"])
    start = min(max(0.0, float(start)), max(0.0, duration - 0.05))
    end_value = duration if end is None else min(max(start + 0.05, float(end)), duration)
    total_duration = max(0.05, end_value - start)
    ranges = _clip_ranges(total_duration, clip_length)
    expected = len(ranges)
    job_dir = JOBS_ROOT / job_id
    clip_dir = job_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    pattern = clip_dir / "clip_%03d.mp4"
    for old in clip_dir.glob("clip_*.mp4"):
        old.unlink(missing_ok=True)

    captions_enabled = bool(captions and captions.get("enabled"))
    quality, ratio = split_export_spec(export_format)
    transform = video_filter(export_format, info)

    # ---------------------------------------------------------------
    # 1) ZERO-TRANSCODE FAST PATH
    # ---------------------------------------------------------------
    # This is the path used for Original + Source ratio + no captions.
    # It avoids decoding and encoding the video entirely.
    copy_allowed = transform is None and not captions_enabled
    if copy_allowed:
        copy_cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        if start > 0.001:
            copy_cmd += ["-ss", f"{start:.3f}"]
        copy_cmd += [
            "-i", str(source), "-t", f"{total_duration:.3f}",
            "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-f", "segment", "-segment_time", f"{clip_length:.6f}",
            "-segment_time_delta", "0.02", "-break_non_keyframes", "1",
            "-reset_timestamps", "1", str(pattern),
        ]
        try:
            run_checked(copy_cmd, timeout=max(600, int(total_duration * 2) + 120))
            copy_clips = sorted(p for p in clip_dir.glob("clip_*.mp4") if p.is_file() and p.stat().st_size > 0)
            if _durations_are_consistent(copy_clips, clip_length, total_duration, tolerance=0.35):
                return copy_clips
        except Exception as exc:
            log.warning("COPY_PATH_FAILED job=%s: %s", job_id, exc)
        for old in clip_dir.glob("clip_*.mp4"):
            old.unlink(missing_ok=True)

    # ---------------------------------------------------------------
    # 2) CAPTIONS: TRANSCRIBE ONCE + ONE FFmpeg PASS
    # ---------------------------------------------------------------
    # The previous implementation encoded the complete video and then encoded
    # every individual clip again to burn subtitles. That multiplies CPU work.
    # Here subtitles, crop/scale, encoding and segmentation happen together.
    if captions_enabled:
        update_job(job_id, progress=10, message="Generating captions...")
        segments = _caption_segments(source, captions.get("language", "auto"), start, start + total_duration)

        relative_segments: list[tuple[float, float, str]] = []
        for seg_start, seg_end, text in segments:
            rs = max(0.0, float(seg_start) - start)
            re_ = min(total_duration, float(seg_end) - start)
            if re_ > rs + 0.01:
                relative_segments.append((rs, re_, text))

        target = _target_dimensions(info, export_format)
        if target:
            canvas_w, canvas_h = target
        else:
            canvas_w = max(2, int(info.get("width") or 1920))
            canvas_h = max(2, int(info.get("height") or 1080))

        full_ass = job_dir / "captions_full.ass"
        full_vtt = job_dir / "captions_full.vtt"
        caption_tracks_from_segments(
            relative_segments, 0.0, total_duration,
            captions.get("style", "classic"), captions.get("font", "DejaVu Sans"),
            captions.get("size", "large"), captions.get("animation", "none"),
            captions.get("position", "bottom"), full_ass, full_vtt, canvas_w, canvas_h,
        )

        try:
            filters: list[str] = []
            if transform:
                filters.append(transform)
            if full_ass.exists():
                escaped = str(full_ass).replace("\\", "/").replace(":", r"\:")
                filters.append(f"subtitles='{escaped}'")

            gop_args = _fixed_gop_args(info, clip_length, total_duration)
            last_boundary = max(0.0, total_duration - 0.05)
            force_expr = f"expr:if(lt(t,{last_boundary:.3f}),gte(t,n_forced*{clip_length:.6f}),0)"
            cmd = [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-sws_flags", "fast_bilinear",
                "-ss", f"{start:.3f}", "-i", str(source),
                "-t", f"{total_duration:.3f}",
                "-map", "0:v:0", "-map", "0:a?",
                "-fps_mode", "passthrough",
                "-c:v", "libx264", "-preset", FFMPEG_PRESET,
                "-crf", str(CRF_ORIGINAL if quality == "original" else CRF_STANDARD),
                "-threads", str(FFMPEG_THREADS),
                "-filter_threads", str(FFMPEG_FILTER_THREADS),
                "-filter_complex_threads", str(FFMPEG_FILTER_THREADS),
                *(gop_args if gop_args else ["-force_key_frames", force_expr]),
                # Audio does not need to be re-encoded for captions/cropping.
                # AAC keeps the generated MP4 broadly compatible.
                *( ["-c:a", "copy"] if str(info.get("audio_codec", "")).lower() == "aac" else ["-c:a", "aac", "-b:a", "128k"] ),
                "-f", "segment", "-segment_time", f"{clip_length:.6f}",
                "-segment_time_delta", "0.02", "-reset_timestamps", "1",
            ]
            if filters:
                cmd += ["-vf", ",".join(filters)]
            if str(info.get("pix_fmt", "")).lower() != "yuv420p":
                cmd += ["-pix_fmt", "yuv420p"]
            if FFMPEG_TUNE:
                cmd += ["-tune", FFMPEG_TUNE]
            if FFMPEG_X264_PARAMS:
                cmd += ["-x264-params", FFMPEG_X264_PARAMS]
            cmd += [str(pattern)]

            run_ffmpeg_progress(
                cmd, job_id, total_duration, 15, 70,
                max(900, int(total_duration * 6)),
            )

            clips = sorted(p for p in clip_dir.glob("clip_*.mp4") if p.is_file() and p.stat().st_size > 0)
            if len(clips) > expected:
                for extra in clips[expected:]:
                    extra.unlink(missing_ok=True)
                clips = clips[:expected]
            if len(clips) != expected or not _durations_are_consistent(
                clips, clip_length, total_duration, tolerance=0.35
            ):
                raise RuntimeError("The caption/export engine could not produce evenly timed clips.")

            # Create lightweight downloadable VTT sidecars from the same
            # transcription. These do not trigger another video encode.
            for index, (relative_start, relative_duration) in enumerate(ranges):
                clip_start = start + relative_start
                clip_end = clip_start + relative_duration
                vtt = job_dir / f"clip_{index:03d}.vtt"
                count = 0
                with vtt.open("w", encoding="utf-8") as vfout:
                    vfout.write("WEBVTT\n\n")
                    for seg_start, seg_end, text in segments:
                        if seg_end <= clip_start or seg_start >= clip_end:
                            continue
                        ls = max(0.0, seg_start - clip_start)
                        le = min(relative_duration, seg_end - clip_start)
                        if le <= ls + 0.01:
                            continue
                        vfout.write(f"{_vtt_timestamp(ls)} --> {_vtt_timestamp(le)}\n{text}\n\n")
                        count += 1
                if count == 0:
                    vtt.unlink(missing_ok=True)
            return clips
        finally:
            full_ass.unlink(missing_ok=True)
            full_vtt.unlink(missing_ok=True)

    # ---------------------------------------------------------------
    # 3) SINGLE-CLIP DIRECT OUTPUT
    # ---------------------------------------------------------------
    # When the requested window produces exactly one clip, avoid the segment
    # muxer and keyframe-forcing machinery entirely. This is the cheapest
    # transcoding route while preserving the requested output dimensions.
    if expected == 1:
        output = clip_dir / "clip_000.mp4"
        single_cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-sws_flags", "fast_bilinear",
            "-ss", f"{start:.3f}", "-i", str(source),
            "-t", f"{total_duration:.3f}",
            "-map", "0:v:0", "-map", "0:a?",
            "-fps_mode", "passthrough",
            "-c:v", "libx264", "-preset", FFMPEG_PRESET,
            "-crf", str(CRF_ORIGINAL if quality == "original" else CRF_STANDARD),
            "-threads", str(FFMPEG_THREADS),
            "-filter_threads", str(FFMPEG_FILTER_THREADS),
            "-filter_complex_threads", str(FFMPEG_FILTER_THREADS),
            *( ["-c:a", "copy"] if str(info.get("audio_codec", "")).lower() == "aac" else ["-c:a", "aac", "-b:a", "128k"] ),
        ]
        if filters:
            single_cmd += ["-vf", ",".join(filters)]
        if str(info.get("pix_fmt", "")).lower() != "yuv420p":
            single_cmd += ["-pix_fmt", "yuv420p"]
        if FFMPEG_TUNE:
            single_cmd += ["-tune", FFMPEG_TUNE]
        if FFMPEG_X264_PARAMS:
            single_cmd += ["-x264-params", FFMPEG_X264_PARAMS]
        single_cmd += [str(output)]
        run_ffmpeg_progress(single_cmd, job_id, total_duration, 20, 72, max(900, int(total_duration * 6)))
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("The export engine produced no clip.")
        return [output]

    # ---------------------------------------------------------------
    # 3) TRANSFORMED EXPORT
    # ---------------------------------------------------------------
    # Encode once directly into the individual MP4 segments. The segment muxer
    # only copies the encoded packets into each file; there is no second full-
    # video disk write/read cycle through an intermediate encoded.mp4.
    gop_args = _fixed_gop_args(info, clip_length, total_duration)
    last_boundary = max(0.0, total_duration - 0.05)
    force_expr = f"expr:if(lt(t,{last_boundary:.3f}),gte(t,n_forced*{clip_length:.6f}),0)"
    filters = [transform] if transform else []
    encode_cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-sws_flags", "fast_bilinear",
        "-ss", f"{start:.3f}", "-i", str(source),
        "-t", f"{total_duration:.3f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-fps_mode", "passthrough",
        "-c:v", "libx264", "-preset", FFMPEG_PRESET,
        "-crf", str(CRF_ORIGINAL if quality == "original" else CRF_STANDARD),
        "-threads", str(FFMPEG_THREADS),
        "-filter_threads", str(FFMPEG_FILTER_THREADS),
        "-filter_complex_threads", str(FFMPEG_FILTER_THREADS),
        *(gop_args if gop_args else ["-force_key_frames", force_expr]),
        *( ["-c:a", "copy"] if str(info.get("audio_codec", "")).lower() == "aac" else ["-c:a", "aac", "-b:a", "128k"] ),
        "-f", "segment", "-segment_time", f"{clip_length:.6f}",
        "-segment_time_delta", "0.02", "-reset_timestamps", "1",
    ]
    if filters:
        encode_cmd += ["-vf", ",".join(filters)]
    if str(info.get("pix_fmt", "")).lower() != "yuv420p":
        encode_cmd += ["-pix_fmt", "yuv420p"]
    if FFMPEG_TUNE:
        encode_cmd += ["-tune", FFMPEG_TUNE]
    if FFMPEG_X264_PARAMS:
        encode_cmd += ["-x264-params", FFMPEG_X264_PARAMS]
    encode_cmd += [str(pattern)]

    run_ffmpeg_progress(
        encode_cmd, job_id, total_duration, 20, 72,
        max(900, int(total_duration * 6)),
    )

    clips = sorted(p for p in clip_dir.glob("clip_*.mp4") if p.is_file() and p.stat().st_size > 0)
    if len(clips) > expected:
        for extra in clips[expected:]:
            extra.unlink(missing_ok=True)
        clips = clips[:expected]
    if len(clips) != expected or not _durations_are_consistent(
        clips, clip_length, total_duration, tolerance=0.35
    ):
        raise RuntimeError("Clip timing could not be made consistent. No partial result was returned.")
    return clips

def render_result(job_id: str, source_info: dict[str, Any], source_name: str, clips: list[Path], thumb: Path, clip_length: float, start: float, total_duration: float) -> dict[str, Any]:
    items=[]
    for index, clip in enumerate(clips):
        duration = clip_length if index < len(clips)-1 else max(0.001, total_duration - clip_length*(len(clips)-1))
        vtt_name=f"{clip.stem}.vtt"
        items.append({
            "name":clip.name,"duration":round(duration,3),"size":clip.stat().st_size,
            "url":f"/api/jobs/{job_id}/download/{clip.name}",
            "thumbnail":f"/api/jobs/{job_id}/thumbnail/{clip.name}",
            "vtt":f"/api/jobs/{job_id}/captions/{vtt_name}",
            "has_vtt":(JOBS_ROOT/job_id/vtt_name).is_file(),
        })
    return {"success":True,"job_id":job_id,"source_name":source_name,"source_duration":round(source_info["duration"],2),"source_size":source_info["size"],"clip_count":len(items),"clips":items,"thumbnail":f"/api/jobs/{job_id}/source-thumb"}

def complete_job(job_id: str, source: Path, source_name: str, clip_length: float, export_format: str, start: float, end: float | None, captions: dict[str, Any] | None) -> None:
    try:
        update_job(job_id, status="processing", progress=10, message="Reading your video...")
        info = probe(source)
        update_job(job_id, progress=20, message="Creating clips...")
        clips = create_clips(job_id, source, clip_length, export_format, start, end, captions, info=info)
        update_job(job_id, progress=92, message=f"Clips created. Finalising {len(clips)} clips...")
        job_dir = JOBS_ROOT / job_id
        thumb = job_dir / "source-thumb.jpg"
        result = render_result(job_id, info, source_name, clips, thumb, clip_length, start, max(0.05, (end if end is not None else info["duration"]) - start))
        update_job(job_id, status="complete", progress=100, message="Your clips are ready.", result=result)
    except Exception as exc:
        log.exception("JOB_FAILED id=%s", job_id)
        update_job(job_id, status="error", progress=0, message=str(exc)[:600])
    finally:
        source.unlink(missing_ok=True)


def run_job_from_upload(job_id: str, source: Path, source_name: str, clip_length: float, export_format: str, start: float, end: float | None, captions: dict[str, Any] | None) -> None:
    complete_job(job_id, source, source_name, clip_length, export_format, start, end, captions)


def recover_jobs_after_restart() -> None:
    """Requeue jobs whose source file survived a process restart."""
    load_persisted_jobs()
    for job_id, job in list(jobs.items()):
        if job.get("status") not in {"queued", "processing"}:
            continue
        raw_source = job.get("source_path")
        source = Path(raw_source) if raw_source else next(iter((JOBS_ROOT / job_id).glob("input.*")), None)
        if not source or not source.is_file():
            update_job(job_id, status="error", progress=0, message="Processing was interrupted before the source could be recovered.")
            continue
        try:
            update_job(job_id, status="queued", progress=0, message="Recovered after server restart.")
            executor.submit(
                run_job_from_upload, job_id, source, job.get("source_name", source.name),
                float(job.get("clip_length", 30)), job.get("export_format", "original"),
                float(job.get("start", 0)), None if job.get("end") in (None, "", 0) else float(job.get("end")),
                job.get("captions"),
            )
        except Exception as exc:
            update_job(job_id, status="error", progress=0, message=f"Recovery failed: {exc}")


def captions_from_form(enabled: bool, language: str, style: str, font: str, size: str, animation: str, position: str) -> dict[str, Any] | None:
    if not enabled: return None
    return {"enabled": True, "language": language, "style": style, "font": font, "size": size, "animation": animation, "position": position}


recover_jobs_after_restart()


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse("ClipShortener API is running. Use /health for diagnostics.")


def _detected_cpu_quota() -> float | None:
    try:
        text = Path("/sys/fs/cgroup/cpu.max").read_text().strip().split()
        if len(text) >= 2 and text[0] != "max":
            return float(text[0]) / max(1.0, float(text[1]))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return None

def _render_cpu_count() -> float | None:
    try:
        value = os.getenv("RENDER_CPU_COUNT", "").strip()
        return float(value) if value else None
    except ValueError:
        return None

def runtime_diagnostics() -> dict[str, Any]:
    return {
        "processing": "local FFmpeg clip engine",
        "input_mode": "uploads + direct public media",
        "url_acquisition": "direct-media service" if ACQUISITION_API_URL else "local_fallback",
        "ffmpeg_threads": FFMPEG_THREADS,
        "ffmpeg_filter_threads": FFMPEG_FILTER_THREADS,
        "render_cpu_count": _render_cpu_count(),
        "container_cpu_quota": _detected_cpu_quota(),
        "ffmpeg_preset": FFMPEG_PRESET,
        "ffmpeg_tune": FFMPEG_TUNE,
        "ffmpeg_progress_period": FFMPEG_PROGRESS_PERIOD,
        "x264_params": FFMPEG_X264_PARAMS,
        "no_upscale": NO_UPSCALE,
        "keyframe_route": "fixed-cfr-gop-with-safe-vfr-fallback",
        "crf_original": CRF_ORIGINAL,
        "crf_standard": CRF_STANDARD,
        "max_video_size": MAX_VIDEO_SIZE,
        "upload_chunk_size": UPLOAD_CHUNK_SIZE,
        "upload_fsync": UPLOAD_FSYNC,
        "validate_output_durations": VALIDATE_OUTPUT_DURATIONS,
        "max_batch_files": MAX_BATCH_FILES,
        "max_clip_seconds": MAX_CLIP_SECONDS,
        "queue_model": "FIFO single-flight CPU scheduler; one active processing job receives the full FFmpeg CPU budget while uploads/acquisition remain concurrent",
        "export_model": "independent quality + aspect ratio",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": VERSION,
        "ffmpeg": bool(FFMPEG),
        "ffprobe": bool(FFPROBE),
        "whisper": bool(WhisperModel),
        "workers": MAX_WORKERS,
        **runtime_diagnostics(),
    }


def run_url_import_job(
    job_id: str,
    url: str,
    length: float,
    export: str,
    start: float,
    end: float | None,
    captions_cfg: dict[str, Any] | None,
) -> None:
    job_dir = JOBS_ROOT / job_id
    try:
        update_job(job_id, status="processing", progress=3, message="Identifying video source...")
        source, filename = _acquire_video(job_id, url)
        update_job(
            job_id,
            status="queued",
            progress=8,
            message="Video acquired. Waiting for your turn in the processing queue.",
            source_path=str(source),
            source_name=filename,
            clip_length=length,
            export_format=export,
            start=start,
            end=end,
            captions=captions_cfg,
            source_url=url,
        )
        # CPU-heavy work goes through the single-flight FIFO executor. Acquisition
        # may happen concurrently, but it never bypasses the processing queue.
        executor.submit(run_job_from_upload, job_id, source, filename, length, export, start, end, captions_cfg)
    except Exception as exc:
        log.exception("URL_IMPORT_FAILED id=%s", job_id)
        update_job(job_id, status="error", progress=0, message=str(exc)[:600], source_url=url)
        # Keep the job state for the frontend to display the actual error.
        if not job_dir.exists():
            job_dir.mkdir(parents=True, exist_ok=True)


@app.get("/api/source-info")
def source_info(url: str = "") -> dict[str, Any]:
    value, is_platform = _validate_source_url(url)
    if not is_platform:
        return {
            "kind": "direct",
            "supported": True,
            "label": "Direct public media",
            "message": "Ready for direct-media acquisition.",
        }
    return {
        "kind": "platform",
        "supported": False,
        "label": "Platform link",
        "message": "This platform is not enabled in Phase 5.",
    }


@app.post("/api/import-url")
async def import_url(
    request: Request,
    url: str = Form(...),
    clip_length: str = Form("30"),
    export_format: str = Form("original"),
    export_quality: str = Form("original"),
    export_ratio: str = Form("source"),
    start: str = Form("0"),
    end: str = Form("0"),
    captions: bool = Form(False),
    caption_language: str = Form("auto"),
    caption_style: str = Form("classic"),
    caption_font: str = Form("DejaVu Sans"),
    caption_size: str = Form("large"),
    caption_animation: str = Form("none"),
    caption_position: str = Form("bottom"),
) -> dict[str, Any]:
    rate_limit(request); cleanup_old_jobs()
    url = (url or "").strip()
    length, export, start_v, end_v = validate_options(clip_length, export_format, start, end, export_quality, export_ratio)
    _validate_source_url(url)
    job_id = uuid.uuid4().hex
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    captions_cfg = captions_from_form(
        captions, caption_language, caption_style, caption_font,
        caption_size, caption_animation, caption_position
    )
    update_job(
        job_id,
        status="queued",
        progress=0,
        message="Queued video acquisition.",
        source_url=url,
        clip_length=length,
        export_format=export,
        export_quality=export.split(":",1)[0],
        export_ratio=export.split(":",1)[1],
        start=start_v,
        end=end_v,
        captions=captions_cfg,
    )
    # Acquisition is asynchronous so the UI gets a job immediately and can
    # show acquisition/processing progress through the same status endpoint.
    asyncio.create_task(asyncio.to_thread(
        run_url_import_job,
        job_id, url, length, export, start_v, end_v, captions_cfg
    ))
    return {"job_id": job_id, "status_url": f"/api/jobs/{job_id}"}


@app.post("/api/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    clip_length: str = Form("30"),
    export_format: str = Form("original"),
    export_quality: str = Form("original"),
    export_ratio: str = Form("source"),
    start: str = Form("0"),
    end: str = Form("0"),
    captions: bool = Form(False),
    caption_language: str = Form("auto"),
    caption_style: str = Form("classic"),
    caption_font: str = Form("DejaVu Sans"),
    caption_size: str = Form("large"),
    caption_animation: str = Form("none"),
    caption_position: str = Form("bottom"),
) -> dict[str, Any]:
    if not getattr(request.state, "batch_upload", False):
        rate_limit(request)
    cleanup_old_jobs()
    length, export, start_v, end_v = validate_options(clip_length, export_format, start, end, export_quality, export_ratio)
    filename = safe_filename(file.filename or "video.mp4")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS and not (file.content_type or "").startswith("video/"):
        raise HTTPException(400, "Unsupported video format.")
    if suffix not in SUPPORTED_EXTENSIONS: suffix = ".mp4"
    job_id = uuid.uuid4().hex
    job_dir = JOBS_ROOT / job_id; job_dir.mkdir(parents=True, exist_ok=True)
    source = job_dir / f"input{suffix}"
    total = 0
    with source.open("wb") as target:
        while True:
            chunk = await file.read(4 * 1024 * 1024)
            if not chunk: break
            total += len(chunk)
            if total > MAX_VIDEO_SIZE:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, "Video exceeds the configured upload limit.")
            target.write(chunk)
    await file.close()
    if total == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, "The selected video is empty.")
    captions_cfg = captions_from_form(captions, caption_language, caption_style, caption_font, caption_size, caption_animation, caption_position)
    update_job(job_id, status="queued", progress=0, message="Queued.", source_path=str(source), source_name=filename, clip_length=length, export_format=export, start=start_v, end=end_v, captions=captions_cfg)
    executor.submit(run_job_from_upload, job_id, source, filename, length, export, start_v, end_v, captions_cfg)
    return {"job_id": job_id, "status_url": f"/api/jobs/{job_id}"}


@app.post("/api/upload-chunk")
async def upload_chunk(request: Request) -> dict[str, Any]:
    """Receive an idempotent, resumable upload chunk."""
    job_id = (request.headers.get("X-Upload-Job") or "").strip()
    session_id = (request.headers.get("X-Upload-Session") or "").strip()
    if not job_id and session_id:
        if not re.fullmatch(r"[0-9a-fA-F]{32}", session_id):
            raise HTTPException(400, "Invalid upload session.")
        job_id = session_id
    is_first = not bool(job_id)
    if is_first:
        rate_limit(request); cleanup_old_jobs(); job_id = uuid.uuid4().hex
    try:
        total_size = int(request.headers.get("X-Upload-Total", "0"))
        index = int(request.headers.get("X-Upload-Index", "-1"))
        total_chunks = int(request.headers.get("X-Upload-Chunks", "0"))
        requested_chunk_size = int(request.headers.get("X-Upload-Chunk-Size", str(UPLOAD_CHUNK_SIZE)))
    except ValueError as exc:
        raise HTTPException(400, "Invalid upload chunk metadata.") from exc
    if total_size <= 0 or total_size > MAX_VIDEO_SIZE:
        raise HTTPException(413, "Video exceeds the configured upload limit or has an invalid size.")
    if requested_chunk_size < 1 * 1024 * 1024 or requested_chunk_size > MAX_UPLOAD_CHUNK_SIZE:
        raise HTTPException(400, "Invalid upload chunk size.")
    if index < 0 or total_chunks < 1 or index >= total_chunks:
        raise HTTPException(400, "Invalid upload chunk index.")
    filename = safe_filename(request.headers.get("X-Upload-Name") or "video.mp4")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, "Unsupported video format.")
    job_dir = JOBS_ROOT / job_id; part = job_dir / "upload.part"; meta_path = job_dir / "upload.json"
    job_dir.mkdir(parents=True, exist_ok=True)
    lock = upload_locks.setdefault(job_id, asyncio.Lock())
    async with lock:
        if not meta_path.is_file():
            if total_chunks > math.ceil(MAX_VIDEO_SIZE / requested_chunk_size):
                raise HTTPException(400, "Too many upload chunks.")
            metadata = {
                "filename": filename, "total_size": total_size, "total_chunks": total_chunks, "chunk_size": requested_chunk_size, "received": [],
                "clip_length": request.headers.get("X-Upload-Clip-Length", "30"), "export_format": request.headers.get("X-Upload-Export", "original"), "export_quality": request.headers.get("X-Upload-Quality", "original"), "export_ratio": request.headers.get("X-Upload-Ratio", "source"),
                "start": request.headers.get("X-Upload-Start", "0"), "end": request.headers.get("X-Upload-End", "0"),
                "captions": request.headers.get("X-Upload-Captions", "0") == "1",
                "caption_language": request.headers.get("X-Upload-Caption-Language", "auto"), "caption_style": request.headers.get("X-Upload-Caption-Style", "classic"),
                "caption_font": request.headers.get("X-Upload-Caption-Font", "DejaVu Sans"), "caption_size": request.headers.get("X-Upload-Caption-Size", "large"),
                "caption_animation": request.headers.get("X-Upload-Caption-Animation", "none"), "caption_position": request.headers.get("X-Upload-Caption-Position", "bottom"),
            }
            validate_options(metadata["clip_length"], metadata["export_format"], metadata["start"], metadata["end"], metadata.get("export_quality","original"), metadata.get("export_ratio","source"))
            meta_tmp = meta_path.with_suffix(".json.tmp")
            meta_tmp.write_text(json.dumps(metadata), encoding="utf-8")
            os.replace(meta_tmp, meta_path)
        else:
            # Existing metadata means this is a normal continuation/retry of the
            # same resumable upload. Do not reject it just because the job is not
            # in the in-memory jobs dict yet; the upload is only queued after the
            # final chunk arrives.
            existing = jobs.get(job_id)
            if existing and existing.get("status") in {"queued", "processing", "complete"}:
                return {"job_id": job_id, "status_url": f"/api/jobs/{job_id}", "complete": True, "retry": True}
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(400, "Upload session metadata is invalid.") from exc
        chunk_size = int(metadata.get("chunk_size", UPLOAD_CHUNK_SIZE))
        if chunk_size < 1 * 1024 * 1024 or chunk_size > MAX_UPLOAD_CHUNK_SIZE:
            raise HTTPException(400, "Upload session has an invalid chunk size.")
        if metadata.get("total_size") != total_size or metadata.get("total_chunks") != total_chunks or metadata.get("filename") != filename:
            raise HTTPException(400, "Upload chunk metadata does not match the original upload.")
        if "X-Upload-Chunk-Size" in request.headers and requested_chunk_size != chunk_size:
            raise HTTPException(400, "Upload chunk size does not match the original upload session.")
        offset = index * chunk_size; expected_chunk = min(chunk_size, total_size - offset)
        if offset >= total_size or expected_chunk <= 0:
            raise HTTPException(400, "Upload chunk offset is invalid.")
        received = {int(x) for x in metadata.get("received", [])}
        if index in received:
            return {
                "job_id": job_id,
                "status_url": f"/api/jobs/{job_id}",
                "index": index,
                "complete": len(received) == total_chunks,
                "received": len(received),
                "total_chunks": total_chunks,
            }
        written = 0
        try:
            with part.open("r+b" if part.exists() else "w+b") as target:
                target.seek(offset)
                async for data in request.stream():
                    if not data: continue
                    written += len(data)
                    if written > expected_chunk: raise HTTPException(413, "Upload chunk is too large.")
                    target.write(data)
                target.flush()
                if UPLOAD_FSYNC:
                    os.fsync(target.fileno())
        except HTTPException: raise
        except Exception as exc: raise HTTPException(500, "Failed to store upload chunk.") from exc
        if written != expected_chunk:
            raise HTTPException(400, f"Incomplete upload chunk: received {written} bytes, expected {expected_chunk}.")
        received.add(index); metadata["received"] = sorted(received)
        meta_tmp = meta_path.with_suffix(".json.tmp")
        meta_tmp.write_text(json.dumps(metadata), encoding="utf-8")
        os.replace(meta_tmp, meta_path)
        if len(received) != total_chunks:
            return {
                "job_id": job_id,
                "status_url": f"/api/jobs/{job_id}",
                "index": index,
                "complete": False,
                "received": len(received),
                "total_chunks": total_chunks,
            }
        if part.stat().st_size != total_size:
            raise HTTPException(400, "Upload is incomplete or corrupted.")
        try:
            length, export, start_v, end_v = validate_options(metadata["clip_length"], metadata["export_format"], metadata["start"], metadata["end"], metadata.get("export_quality","original"), metadata.get("export_ratio","source"))
            captions_cfg = captions_from_form(metadata["captions"], metadata["caption_language"], metadata["caption_style"], metadata["caption_font"], metadata["caption_size"], metadata["caption_animation"], metadata["caption_position"])
            source = job_dir / f"input{suffix}"; part.replace(source); meta_path.unlink(missing_ok=True)
            update_job(job_id, status="queued", progress=0, message="Upload complete. Queued.", source_path=str(source), source_name=filename, clip_length=length, export_format=export, start=start_v, end=end_v, captions=captions_cfg)
            executor.submit(run_job_from_upload, job_id, source, filename, length, export, start_v, end_v, captions_cfg)
            return {"job_id": job_id, "status_url": f"/api/jobs/{job_id}", "complete": True, "received": total_chunks}
        except HTTPException: raise
        except Exception:
            log.exception("CHUNK_UPLOAD_FINALIZE_FAILED id=%s", job_id); shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(500, "The upload could not be finalized.")

@app.get("/api/upload-session/{job_id}")
def upload_session(job_id: str) -> dict[str, Any]:
    """Return resumable upload state for a previously started upload."""
    meta_path = JOBS_ROOT / job_id / "upload.json"
    if not meta_path.is_file():
        existing = jobs.get(job_id)
        if existing and existing.get("status") in {"queued", "processing", "complete"}:
            return {"job_id": job_id, "complete": True, "received": [], "total_chunks": 0}
        # A resumable client can legitimately hold a session id across a
        # Render restart/redeploy.  The old session data may be gone because
        # the filesystem is ephemeral.  Return an empty resumable state instead
        # of a 404 so the frontend can resend chunk 0 and recreate the session
        # using the same id.  This directly addresses the retry loop seen in
        # production after a deploy.
        if re.fullmatch(r"[0-9a-fA-F]{32}", job_id):
            return {"job_id": job_id, "complete": False, "received": [], "total_chunks": 0, "total_size": 0, "chunk_size": UPLOAD_CHUNK_SIZE, "received_bytes": 0, "next_missing": 0, "filename": "", "reset": True}
        raise HTTPException(400, "Invalid upload session.")
    try: metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc: raise HTTPException(500, "Upload session metadata is unreadable.") from exc
    received = sorted({int(x) for x in metadata.get("received", [])})
    total_chunks = int(metadata.get("total_chunks", 0) or 0)
    total_size = int(metadata.get("total_size", 0) or 0)
    received_set = set(received)
    chunk_size = int(metadata.get("chunk_size", UPLOAD_CHUNK_SIZE))
    received_bytes = sum(
        min(chunk_size, max(0, total_size - i * chunk_size))
        for i in received
    )
    next_missing = next((i for i in range(total_chunks) if i not in received_set), None)
    return {
        "job_id": job_id,
        "complete": False,
        "received": received,
        "total_chunks": total_chunks,
        "total_size": total_size,
        "chunk_size": metadata.get("chunk_size", UPLOAD_CHUNK_SIZE),
        "received_bytes": received_bytes,
        "next_missing": next_missing,
        "filename": metadata.get("filename", ""),
    }


@app.post("/api/batch")
async def batch(
    request: Request,
    files: list[UploadFile] = File(...),
    clip_length: str = Form("30"),
    export_format: str = Form("original"),
    export_quality: str = Form("original"),
    export_ratio: str = Form("source"),
    captions: bool = Form(False),
    caption_language: str = Form("auto"),
    caption_style: str = Form("classic"),
    caption_font: str = Form("DejaVu Sans"),
    caption_size: str = Form("large"),
    caption_animation: str = Form("none"),
    caption_position: str = Form("bottom"),
) -> dict[str, Any]:
    rate_limit(request); cleanup_old_jobs()
    if not 1 <= len(files) <= MAX_BATCH_FILES:
        raise HTTPException(400, f"Batch size must be between 1 and {MAX_BATCH_FILES} videos.")
    results=[]; request.state.batch_upload=True
    for file in files:
        try:
            result=await upload(
                request=request, file=file, clip_length=clip_length, export_format=export_format,
                export_quality=export_quality, export_ratio=export_ratio, start="0", end="0",
                captions=captions, caption_language=caption_language, caption_style=caption_style,
                caption_font=caption_font, caption_size=caption_size, caption_animation=caption_animation,
                caption_position=caption_position,
            )
            results.append(result)
        except HTTPException as exc:
            results.append({"success":False,"filename":file.filename,"error":exc.detail})
    return {"success":True,"jobs":results}


@app.post("/api/detect")
async def detect(request: Request, file: UploadFile = File(...), clip_length: str = Form("30")) -> dict[str, Any]:
    rate_limit(request); cleanup_old_jobs()
    length, _, _, _ = validate_options(clip_length, "original", "0", "0")
    filename = safe_filename(file.filename or "video.mp4")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        suffix = ".mp4"
    temp_id = uuid.uuid4().hex
    job_dir = JOBS_ROOT / temp_id; job_dir.mkdir()
    src = job_dir / f"input{suffix}"
    try:
        with src.open("wb") as target:
            while True:
                chunk = await file.read(4 * 1024 * 1024)
                if not chunk: break
                target.write(chunk)
                if target.stat().st_size > MAX_VIDEO_SIZE: raise HTTPException(413, "Video exceeds the configured upload limit.")
        info = probe(src)
        # Fast scene scan on a low-FPS stream; candidate windows are later turned into real clips.
        result = run_checked([
            FFMPEG, "-hide_banner", "-i", str(src), "-vf", "fps=2,select='gt(scene,0.35)',showinfo", "-f", "null", "-",
        ], timeout=300)
        times = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", result.stderr or "")]
        candidates = []
        half = length / 2
        for t in times[:30]:
            start_v = max(0.0, min(t - half, max(0.0, info["duration"] - length)))
            candidates.append({"start": round(start_v, 2), "end": round(min(info["duration"], start_v + length), 2), "score": 100})
        # De-duplicate close candidates.
        unique = []
        for c in candidates:
            if not unique or abs(c["start"] - unique[-1]["start"]) >= length * 0.4:
                unique.append(c)
        return {"success": True, "candidates": unique[:10]}
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


@app.get("/api/batch-status")
def batch_status(jobs_param: str = Query("", alias="jobs")) -> dict[str, Any]:
    """Return the state of up to MAX_BATCH_FILES jobs in one request.

    Batch clients use this endpoint instead of polling every job separately.
    That cuts request volume substantially while keeping per-job failures
    isolated and preserving the existing job-status contract.
    """
    ids = [x.strip() for x in jobs_param.split(",") if x.strip()]
    if not ids or len(ids) > MAX_BATCH_FILES:
        raise HTTPException(400, f"Batch status must contain between 1 and {MAX_BATCH_FILES} jobs.")
    if any(not re.fullmatch(r"[0-9a-fA-F]{32}", job_id) for job_id in ids):
        raise HTTPException(400, "Invalid batch job identifier.")

    states: list[dict[str, Any]] = []
    for job_id in ids:
        job = _recover_job_from_disk(job_id)
        if not job:
            job = jobs.get(job_id)
        if job is None:
            states.append({"job_id": job_id, "status": "error", "progress": 0, "message": "Job not found or expired."})
        else:
            states.append({
                "job_id": job_id,
                "status": job.get("status", "queued"),
                "progress": max(0, min(100, int(job.get("progress", 0) or 0))),
                "message": job.get("message", ""),
                "result": job.get("result"),
            })
    return {"success": True, "jobs": states}


def _recover_job_from_disk(job_id: str) -> dict[str, Any] | None:
    """Recover a job snapshot, including completed clips, from its job folder."""
    job_dir = JOBS_ROOT / job_id
    if not job_dir.is_dir():
        return None
    state_path = job_dir / "job.json"
    if state_path.is_file():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                jobs[job_id] = data
                return data
        except Exception:
            pass

    # If a process died between creating the job directory and persisting the
    # first state snapshot, don't tell the frontend that the job vanished when
    # useful artifacts are still present. Reconstruct a minimal state.
    clips = sorted((job_dir / "clips").glob("clip_*.mp4")) if (job_dir / "clips").is_dir() else []
    clips = [p for p in clips if p.is_file() and p.stat().st_size > 0]
    if clips:
        source = next(iter(job_dir.glob("input.*")), None)
        source_name = source.name if source else "video.mp4"
        recovered = {
            "job_id": job_id,
            "status": "complete",
            "progress": 100,
            "message": "Your clips are ready.",
            "source_name": source_name,
            "clip_length": 30,
            "result": {
                "success": True,
                "job_id": job_id,
                "source_name": source_name,
                "clip_count": len(clips),
                "clips": [
                    {
                        "name": p.name,
                        "size": p.stat().st_size,
                        "url": f"/api/jobs/{job_id}/download/{p.name}",
                        "thumbnail": f"/api/jobs/{job_id}/thumbnail/{p.name}",
                    } for p in clips
                ],
            },
        }
        try:
            _persist_job(job_id, recovered)
        except Exception:
            pass
        jobs[job_id] = recovered
        return recovered
    return None

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-fA-F]{32}", job_id):
        raise HTTPException(400, "Invalid job identifier.")
    recovered = _recover_job_from_disk(job_id)
    if recovered is not None:
        return recovered
    job = jobs.get(job_id)
    if job:
        return job
    raise HTTPException(404, "Job not found or expired.")


def _preview_headers() -> dict[str, str]:
    return {
        "Cache-Control": "public, max-age=86400, immutable",
        "X-Content-Type-Options": "nosniff",
    }


@app.get("/api/jobs/{job_id}/source-thumb")
def source_thumb(job_id: str) -> FileResponse:
    path = JOBS_ROOT / job_id / "source-thumb.jpg"
    if not path.is_file():
        job = jobs.get(job_id)
        source = Path(job.get("source_path", "")) if job else None
        if not source or not source.is_file():
            candidates = list((JOBS_ROOT / job_id).glob("input.*"))
            source = candidates[0] if candidates else None
        if not source or not source.is_file():
            raise HTTPException(404, "Source thumbnail not available.")
        make_thumbnail(source, path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise HTTPException(404, "Thumbnail not available.")
    return FileResponse(path, media_type="image/jpeg", headers=_preview_headers())


@app.get("/api/jobs/{job_id}/thumbnail/{name}")
def clip_thumb(job_id: str, name: str) -> FileResponse:
    if "/" in name or "\\" in name: raise HTTPException(400, "Invalid filename.")
    clip = JOBS_ROOT / job_id / "clips" / name
    if not clip.is_file() or clip.suffix.lower() != ".mp4":
        raise HTTPException(404, "Clip not found.")
    thumb = clip.with_suffix(".jpg")
    if not thumb.is_file():
        # Backward-compatible fallback for jobs created before preview hardening.
        make_thumbnail(clip, thumb)
    if not thumb.is_file() or thumb.stat().st_size <= 0:
        raise HTTPException(404, "Preview not available.")
    return FileResponse(thumb, media_type="image/jpeg", headers=_preview_headers())


def _safe_clip_path(job_id: str, name: str) -> Path:
    if not re.fullmatch(r"clip_\d{3}\.mp4", name or ""):
        raise HTTPException(400, "Invalid filename.")
    path = JOBS_ROOT / job_id / "clips" / name
    if not path.is_file() or path.stat().st_size <= 0:
        raise HTTPException(404, "Clip not found.")
    return path

def _range_response(request: Request, path: Path, media_type: str, disposition: str = "inline", download_name: str | None = None):
    """Reliable large-file response with browser seeking and explicit disposition."""
    try:
        size = path.stat().st_size
    except OSError:
        raise HTTPException(404, "File not found.")
    if size <= 0:
        raise HTTPException(404, "File is empty.")
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=3600",
        "X-Content-Type-Options": "nosniff",
    }
    if download_name:
        safe = safe_filename(download_name).replace('"', "")
        headers["Content-Disposition"] = f'{disposition}; filename="{safe}"'

    range_header = request.headers.get("range")
    if not range_header:
        headers["Content-Length"] = str(size)
        return FileResponse(path, media_type=media_type, headers=headers)

    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match:
        headers["Content-Range"] = f"bytes */{size}"
        return StreamingResponse(iter(()), status_code=416, media_type=media_type, headers=headers)
    start_s, end_s = match.groups()
    try:
        if start_s:
            start_byte = int(start_s)
            if start_byte >= size:
                raise ValueError
            end_byte = min(int(end_s) if end_s else size - 1, size - 1)
            if end_byte < start_byte:
                raise ValueError
        else:
            suffix = int(end_s)
            if suffix <= 0:
                raise ValueError
            length = min(suffix, size)
            start_byte = size - length
            end_byte = size - 1
    except (ValueError, TypeError):
        headers["Content-Range"] = f"bytes */{size}"
        return StreamingResponse(iter(()), status_code=416, media_type=media_type, headers=headers)

    length = end_byte - start_byte + 1
    def iterator():
        with path.open("rb") as fh:
            fh.seek(start_byte)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(4 * 1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
    headers["Content-Range"] = f"bytes {start_byte}-{end_byte}/{size}"
    headers["Content-Length"] = str(length)
    return StreamingResponse(iterator(), status_code=206, media_type=media_type, headers=headers)

@app.get("/api/jobs/{job_id}/media/{name}")
def media(job_id: str, name: str, request: Request):
    path = _safe_clip_path(job_id, name)
    return _range_response(request, path, "video/mp4", "inline", path.name)


@app.get("/api/jobs/{job_id}/download/{name}")
def download(job_id: str, name: str, request: Request):
    path = _safe_clip_path(job_id, name)
    return _range_response(request, path, "video/mp4", "attachment", path.name)


@app.get("/api/jobs/{job_id}/downloads")
def list_downloads(job_id: str):
    """Return individual completed clip URLs for a job; never creates an archive."""
    job_dir = JOBS_ROOT / job_id
    clip_dir = job_dir / "clips"
    if not job_dir.is_dir() or not clip_dir.is_dir():
        raise HTTPException(404, "Job not found or expired.")
    clips = sorted(p for p in clip_dir.glob("clip_*.mp4") if p.is_file() and p.stat().st_size > 0)
    if not clips:
        raise HTTPException(404, "No completed clips were found.")
    return {
        "success": True,
        "job_id": job_id,
        "files": [
            {"name": p.name, "size": p.stat().st_size, "url": f"/api/jobs/{job_id}/download/{p.name}"}
            for p in clips
        ],
    }


@app.get("/api/jobs/{job_id}/captions/{name}")
def download_caption(job_id: str, name: str) -> FileResponse:
    if not re.fullmatch(r"clip_\d{3}\.vtt", name or ""):
        raise HTTPException(400, "Invalid filename.")
    path = JOBS_ROOT / job_id / name
    if not path.is_file(): raise HTTPException(404, "Caption sidecar not found.")
    return FileResponse(path, media_type="text/vtt", filename=path.name, headers={"Content-Disposition": f'attachment; filename="{path.name}"'})


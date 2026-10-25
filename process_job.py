import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

CONTROL_URL = os.environ["CONTROL_URL"].rstrip("/")
WORKER_SECRET = os.environ["WORKER_SECRET"]
B2_KEY_ID = os.environ["B2_KEY_ID"].strip()
B2_APPLICATION_KEY = os.environ["B2_APPLICATION_KEY"].strip()
B2_BUCKET = os.environ["B2_BUCKET"].strip()
B2_AUTH_URL = "https://api.backblazeb2.com/b2api/v4/b2_authorize_account"

JOB_ID = os.environ["JOB_ID"]
_raw_payload = json.loads(os.environ["JOB_PAYLOAD"])
PAYLOAD = _raw_payload.get("job", _raw_payload)

ROOT = Path("/tmp/clipshortener-worker")
JOBDIR = ROOT / "job"
JOBDIR.mkdir(parents=True, exist_ok=True)
SOURCE = JOBDIR / PAYLOAD.get("filename", "input.mp4")
UPLOAD_BLOCK_SIZE = 16 * 1024 * 1024


def auth():
    response = requests.get(
        B2_AUTH_URL,
        auth=(B2_KEY_ID, B2_APPLICATION_KEY),
        headers={"User-Agent": "ClipShortener-Processor/4.0"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    storage = data["apiInfo"]["storageApi"]
    allowed = storage.get("allowed", {}).get("buckets", [])
    bucket = next((b for b in allowed if b.get("name") == B2_BUCKET), None)
    if not bucket:
        raise RuntimeError(f"B2 bucket {B2_BUCKET!r} is not allowed to this key")
    return (
        data["authorizationToken"],
        storage["apiUrl"],
        storage["downloadUrl"],
        bucket["id"],
    )


def status(state, progress=None, message=None, clips=None):
    """Report worker progress with retries for transient network/server errors."""
    data = {"status": state}
    if progress is not None:
        data["progress"] = progress
    if message is not None:
        data["message"] = message
    if clips is not None:
        data["clips"] = clips

    url = f"{CONTROL_URL}/api/jobs/{JOB_ID}/worker-status"
    headers = {
        "X-Worker-Secret": WORKER_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    retryable = {408, 429, 500, 502, 503, 504}
    max_attempts = 5
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=30,
            )

            if response.status_code not in retryable:
                response.raise_for_status()
                return

            last_error = requests.HTTPError(
                f"HTTP {response.status_code}: {response.text[:300]}",
                response=response,
            )

        except requests.RequestException as exc:
            last_error = exc
            response = getattr(exc, "response", None)

            # Authentication and other permanent client errors should not
            # be retried repeatedly.
            if (
                response is not None
                and response.status_code not in retryable
            ):
                raise

        if attempt < max_attempts:
            delay = min(2 ** (attempt - 1), 8)
            print(
                f"WORKER_STATUS_RETRY attempt={attempt}/{max_attempts} "
                f"state={state} retry_in={delay}s "
                f"error={str(last_error)[:250]}",
                flush=True,
            )
            time.sleep(delay)

    # Intermediate status updates should not kill otherwise healthy
    # processing when the status server is temporarily unavailable.
    if state not in {"complete", "error"}:
        print(
            f"WORKER_STATUS_WARNING state={state} "
            f"error={str(last_error)[:500]}; continuing processing",
            flush=True,
        )
        return

    # The final status is essential. Do not report success locally if the
    # website cannot be notified that the completed clips are ready.
    raise RuntimeError(
        f"Final worker status '{state}' failed after "
        f"{max_attempts} attempts: {last_error}"
    ) from last_error


def download_source():
    token, _, download_url, _ = auth()
    key = PAYLOAD["source_key"]
    encoded = "/".join(quote(part, safe="") for part in key.split("/"))
    url = f"{download_url}/file/{quote(B2_BUCKET, safe='')}/{encoded}"

    temp = SOURCE.with_suffix(SOURCE.suffix + ".part")
    with requests.get(
        url,
        headers={"Authorization": token},
        stream=True,
        timeout=(30, 1800),
    ) as response:
        response.raise_for_status()
        with temp.open("wb") as out:
            for block in response.iter_content(chunk_size=UPLOAD_BLOCK_SIZE):
                if block:
                    out.write(block)
    temp.replace(SOURCE)


def b2_upload_file(path: Path, key: str, content_type: str):
    """Upload a file to B2 with explicit length and SHA-1 verification."""
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"Refusing to upload empty file: {path.name}")

    upload_started = time.monotonic()

    for attempt in range(1, 5):
        token, api_url, _, bucket_id = auth()
        upload_info_response = requests.get(
            f"{api_url}/b2api/v4/b2_get_upload_url",
            params={"bucketId": bucket_id},
            headers={"Authorization": token},
            timeout=30,
        )
        if upload_info_response.status_code == 401 and attempt < 4:
            continue
        upload_info_response.raise_for_status()
        info = upload_info_response.json()

        digest = hashlib.sha1()
        with path.open("rb") as source:
            for block in iter(
                lambda: source.read(UPLOAD_BLOCK_SIZE),
                b"",
            ):
                digest.update(block)
        expected_sha1 = digest.hexdigest().lower()

        try:
            with path.open("rb") as source:
                response = requests.post(
                    info["uploadUrl"],
                    headers={
                        "Authorization": info["authorizationToken"],
                        "X-Bz-File-Name": quote(key, safe=""),
                        "Content-Type": content_type,
                        "Content-Length": str(size),
                        "X-Bz-Content-Sha1": expected_sha1,
                    },
                    data=source,
                    timeout=(30, 1800),
                )
        except requests.RequestException:
            if attempt < 4:
                time.sleep(min(attempt, 3))
                continue
            raise

        if response.ok:
            result = response.json()
            remote_sha1 = str(
                result.get("contentSha1")
                or result.get("content_sha1")
                or ""
            ).lower()

            if remote_sha1 and remote_sha1 != expected_sha1:
                raise RuntimeError(
                    f"B2 checksum mismatch for {path.name}: "
                    f"expected={expected_sha1} remote={remote_sha1}"
                )

            elapsed = time.monotonic() - upload_started
            print(
                f"B2_OUTPUT_UPLOAD_OK file={path.name} bytes={size} "
                f"seconds={elapsed:.2f} "
                f"sha1_verified={remote_sha1 == expected_sha1}",
                flush=True,
            )
            return result

        if response.status_code in (401,) or response.status_code >= 500:
            if attempt < 4:
                time.sleep(min(attempt, 3))
                continue

        raise RuntimeError(
            f"B2 output upload failed: HTTP {response.status_code}; "
            f"content_type={response.headers.get('Content-Type', '')}; "
            f"response={response.text[:1000]}"
        )

    raise RuntimeError(f"B2 output upload failed after retries: {key}")


def upload_one(path: Path, job_dir: Path):
    key = f"jobs/{JOB_ID}/clips/{path.name}"
    b2_upload_file(path, key, "video/mp4")

    import app
    info = app.probe(path)
    vtt_path = job_dir / f"{path.stem}.vtt"

    return {
        "name": path.name,
        "duration": round(float(info["duration"]), 2),
        "size": path.stat().st_size,
        "url": f"{CONTROL_URL}/api/jobs/{JOB_ID}/download/{path.name}",
        "vtt": (
            f"{CONTROL_URL}/api/jobs/{JOB_ID}/captions/{vtt_path.name}"
            if vtt_path.is_file()
            else ""
        ),
        "has_vtt": vtt_path.is_file(),
    }


def upload_outputs():
    import app

    job_dir = Path(app.JOBS_ROOT) / JOB_ID
    outdir = job_dir / "clips"
    clips = sorted(outdir.glob("clip_*.mp4"))
    vtts = sorted(job_dir.glob("*.vtt"))

    if not clips:
        raise RuntimeError("No clips were available for output upload.")

    results = []
    workers = min(4, max(1, len(clips)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(upload_one, path, job_dir)
            for path in clips
        ]
        for future in as_completed(futures):
            results.append(future.result())

    for path in vtts:
        b2_upload_file(
            path,
            f"jobs/{JOB_ID}/captions/{path.name}",
            "text/vtt; charset=utf-8",
        )

    results.sort(key=lambda item: item["name"])
    return results


def main():
    job_started = time.monotonic()

    status("processing", 1, "Processing worker started.")
    status("processing", 5, "Downloading source from B2...")

    download_started = time.monotonic()
    download_source()
    download_seconds = time.monotonic() - download_started

    print(
        f"PHASE_DOWNLOAD seconds={download_seconds:.2f} "
        f"bytes={SOURCE.stat().st_size}",
        flush=True,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import app

    info = app.probe(SOURCE)

    print(
        "SOURCE_PROFILE "
        f"bytes={SOURCE.stat().st_size} duration={info['duration']:.3f} "
        f"width={info['width']} height={info['height']} "
        f"avg_fps={info['avg_frame_rate']} r_fps={info['r_frame_rate']}",
        flush=True,
    )

    status("processing", 16, "Creating clips...")

    encode_started = time.monotonic()
    clips = app.create_clips(
        JOB_ID,
        SOURCE,
        float(PAYLOAD.get("clip_length", 30)),
        PAYLOAD.get("export_format", "original:source"),
        float(PAYLOAD.get("start", 0)),
        PAYLOAD.get("end"),
        PAYLOAD.get("captions"),
        info=info,
    )
    encode_seconds = time.monotonic() - encode_started

    if not clips:
        raise RuntimeError("No clips were produced.")

    output_bytes = sum(
        path.stat().st_size for path in clips if path.is_file()
    )

    print(
        f"PHASE_CLIP_PROCESS seconds={encode_seconds:.2f} "
        f"clips={len(clips)} output_bytes={output_bytes}",
        flush=True,
    )

    status(
        "processing",
        80,
        f"Uploading {len(clips)} finished clips...",
    )

    upload_started = time.monotonic()
    result = upload_outputs()
    upload_seconds = time.monotonic() - upload_started

    print(
        f"PHASE_OUTPUT_UPLOAD seconds={upload_seconds:.2f} "
        f"clips={len(result)} "
        f"output_bytes={sum(int(item.get('size') or 0) for item in result)}",
        flush=True,
    )

    # Include the clip list in the terminal status so the website can
    # display the individual downloadable output files.
    status("complete", 100, "Your clips are ready.", result)

    print(
        f"PHASE_JOB_TOTAL seconds={time.monotonic() - job_started:.2f} "
        f"download_seconds={download_seconds:.2f} "
        f"encode_seconds={encode_seconds:.2f} "
        f"output_upload_seconds={upload_seconds:.2f} "
        f"input_bytes={SOURCE.stat().st_size} "
        f"output_bytes={sum(int(item.get('size') or 0) for item in result)} "
        f"clips={len(result)}",
        flush=True,
    )

    SOURCE.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"WORKER FAILED: {exc}", file=sys.stderr)
        try:
            status("error", 0, str(exc)[:700])
        except Exception as status_exc:
            print(
                f"WORKER_ERROR_STATUS_FAILED: {status_exc}",
                file=sys.stderr,
            )
        raise

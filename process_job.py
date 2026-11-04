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
SOURCE_RANGE_SIZE = 64 * 1024 * 1024
SOURCE_RANGE_ATTEMPTS = 5
SOURCE_RANGE_RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}


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
        raise RuntimeError(f"B2 bucket {B2_BUCKET!r} is not allowed to this application key")
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
    """Download the private source through Cloudflare using verified byte ranges.

    A single multi-gigabyte HTTP response proved vulnerable to mid-stream
    connection resets. The Cloudflare Worker already forwards Range requests
    to B2 and streams each response, so use bounded range requests with
    per-range retries. Failed ranges are discarded and retried without losing
    previously completed ranges during this processor run.
    """
    url = f"{CONTROL_URL}/api/jobs/{quote(JOB_ID, safe='')}/source"
    headers = {
        "X-Worker-Secret": WORKER_SECRET,
        "Accept": "application/octet-stream",
        "Accept-Encoding": "identity",
    }
    temp = SOURCE.with_suffix(SOURCE.suffix + ".part")

    expected_size = int(PAYLOAD.get("total_size") or 0)
    if expected_size <= 0:
        try:
            head = requests.head(
                url,
                headers=headers,
                timeout=(30, 60),
                allow_redirects=False,
            )
            head.raise_for_status()
            expected_size = int(head.headers.get("Content-Length") or 0)
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                "Could not determine source size from the Cloudflare source proxy."
            ) from exc

    if expected_size <= 0:
        raise RuntimeError("The source payload did not contain a valid video size.")

    temp.unlink(missing_ok=True)
    started_at = time.monotonic()
    completed_bytes = 0
    last_status_bucket = 0

    try:
        with temp.open("w+b") as out:
            offset = 0
            while offset < expected_size:
                end = min(offset + SOURCE_RANGE_SIZE, expected_size) - 1
                expected_range_bytes = end - offset + 1
                last_error = None
                range_started = time.monotonic()

                for attempt in range(1, SOURCE_RANGE_ATTEMPTS + 1):
                    received_this_attempt = 0
                    try:
                        range_headers = {
                            **headers,
                            "Range": f"bytes={offset}-{end}",
                        }

                        with requests.get(
                            url,
                            headers=range_headers,
                            stream=True,
                            timeout=(30, 180),
                        ) as response:
                            if response.status_code != 206:
                                # Do not read response.text here: if a proxy
                                # ignores Range and returns HTTP 200, the body
                                # could be the entire multi-gigabyte source.
                                if response.status_code in SOURCE_RANGE_RETRYABLE_STATUSES:
                                    raise requests.HTTPError(
                                        f"Retryable source range HTTP "
                                        f"{response.status_code}",
                                        response=response,
                                    )

                                raise RuntimeError(
                                    f"Cloudflare source range request expected HTTP 206 "
                                    f"but received HTTP {response.status_code}."
                                )

                            content_range = response.headers.get("Content-Range", "")
                            expected_content_range = (
                                f"bytes {offset}-{end}/{expected_size}"
                            )
                            if content_range.strip() != expected_content_range:
                                raise RuntimeError(
                                    "Cloudflare returned an incorrect Content-Range: "
                                    f"expected={expected_content_range!r} "
                                    f"received={content_range!r}"
                                )

                            declared_length = response.headers.get("Content-Length")
                            if declared_length:
                                try:
                                    declared_length = int(declared_length)
                                except ValueError as exc:
                                    raise RuntimeError(
                                        "Cloudflare source range returned an invalid "
                                        f"Content-Length: {declared_length!r}"
                                    ) from exc
                                if declared_length != expected_range_bytes:
                                    raise RuntimeError(
                                        "Cloudflare source range length mismatch: "
                                        f"expected={expected_range_bytes} "
                                        f"declared={declared_length}"
                                    )

                            out.seek(offset)
                            for block in response.iter_content(
                                chunk_size=UPLOAD_BLOCK_SIZE
                            ):
                                if not block:
                                    continue
                                received_this_attempt += len(block)
                                if received_this_attempt > expected_range_bytes:
                                    raise RuntimeError(
                                        "Cloudflare sent more bytes than requested "
                                        f"for range {offset}-{end}."
                                    )
                                out.write(block)

                        if received_this_attempt != expected_range_bytes:
                            raise requests.exceptions.ChunkedEncodingError(
                                "Incomplete source range: "
                                f"expected={expected_range_bytes} "
                                f"received={received_this_attempt}"
                            )

                        last_error = None
                        break

                    except requests.RequestException as exc:
                        last_error = exc
                    except RuntimeError as exc:
                        # Retry only explicitly transient HTTP responses. A
                        # malformed Content-Range or other permanent response
                        # must fail clearly rather than risk corrupting a file.
                        last_error = exc
                        if "Retryable source range HTTP" not in str(exc):
                            raise

                    # Discard partial bytes from this attempt before retrying.
                    out.seek(offset)
                    out.truncate(offset)

                    if attempt < SOURCE_RANGE_ATTEMPTS:
                        delay = min(2 ** (attempt - 1), 16)
                        print(
                            f"SOURCE_RANGE_RETRY range={offset}-{end} "
                            f"attempt={attempt}/{SOURCE_RANGE_ATTEMPTS} "
                            f"retry_in={delay}s error={str(last_error)[:300]}",
                            flush=True,
                        )
                        time.sleep(delay)

                if last_error is not None:
                    raise RuntimeError(
                        f"Source range {offset}-{end} failed after "
                        f"{SOURCE_RANGE_ATTEMPTS} attempts: {last_error}"
                    ) from last_error

                completed_bytes += expected_range_bytes
                range_seconds = time.monotonic() - range_started
                percent = completed_bytes * 100.0 / expected_size
                print(
                    f"SOURCE_RANGE_OK start={offset} end={end} "
                    f"bytes={expected_range_bytes} seconds={range_seconds:.2f} "
                    f"progress={percent:.1f}%",
                    flush=True,
                )

                status_bucket = completed_bytes // (512 * 1024 * 1024)
                if status_bucket > last_status_bucket:
                    last_status_bucket = status_bucket
                    status(
                        "processing",
                        5,
                        f"Downloading source through Cloudflare... "
                        f"{completed_bytes * 100 // expected_size}%",
                    )

                offset = end + 1

            out.flush()
            os.fsync(out.fileno())

        actual_size = temp.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Source proxy download size mismatch: "
                f"expected={expected_size} received={actual_size}"
            )

        temp.replace(SOURCE)
        print(
            f"SOURCE_PROXY_DOWNLOAD_OK bytes={actual_size} "
            f"expected_bytes={expected_size} "
            f"range_size={SOURCE_RANGE_SIZE} "
            f"seconds={time.monotonic() - started_at:.2f}",
            flush=True,
        )
    except Exception:
        temp.unlink(missing_ok=True)
        raise


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
    clip_workers = min(4, max(1, len(clips)))
    vtt_workers = min(4, len(vtts))

    def upload_vtt(path: Path):
        b2_upload_file(
            path,
            f"jobs/{JOB_ID}/captions/{path.name}",
            "text/vtt; charset=utf-8",
        )

    # Upload videos and caption files concurrently.
    with ThreadPoolExecutor(
        max_workers=max(1, clip_workers + vtt_workers)
    ) as pool:
        clip_futures = [
            pool.submit(upload_one, path, job_dir)
            for path in clips
        ]

        vtt_futures = [
            pool.submit(upload_vtt, path)
            for path in vtts
        ]

        for future in as_completed(clip_futures):
            results.append(future.result())

        for future in as_completed(vtt_futures):
            future.result()

    print(
        f"OUTPUT_UPLOAD_CONCURRENCY clips={len(clips)} "
        f"clip_workers={clip_workers} "
        f"captions={len(vtts)} caption_workers={vtt_workers}",
        flush=True,
    )

    results.sort(key=lambda item: item["name"])
    return results


def main():
    job_started = time.monotonic()

    status("processing", 1, "Processing worker started.")
    status("processing", 5, "Downloading source through Cloudflare...")

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

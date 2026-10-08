import hashlib
import json
import os
import shutil
import sys
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
    return data["authorizationToken"], storage["apiUrl"], storage["downloadUrl"], bucket["id"]


def status(state, progress=None, message=None, clips=None):
    data = {"status": state}
    if progress is not None:
        data["progress"] = progress
    if message is not None:
        data["message"] = message
    if clips is not None:
        data["clips"] = clips

    response = requests.post(
        f"{CONTROL_URL}/api/jobs/{JOB_ID}/worker-status",
        headers={
            "X-Worker-Secret": WORKER_SECRET,
            "content-type": "application/json",
        },
        json=data,
        timeout=30,
    )
    response.raise_for_status()


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
            for block in response.iter_content(chunk_size=16 * 1024 * 1024):
                if block:
                    out.write(block)
    temp.replace(SOURCE)


def b2_upload_file(path: Path, key: str, content_type: str):
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
            for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
                digest.update(block)

        size = path.stat().st_size
        try:
            with path.open("rb") as source:
                response = requests.post(
                    info["uploadUrl"],
                    headers={
                        "Authorization": info["authorizationToken"],
                        "X-Bz-File-Name": quote(key, safe=""),
                        "Content-Type": content_type,
                        "Content-Length": str(size),
                        "X-Bz-Content-Sha1": digest.hexdigest(),
                    },
                    data=source,
                    timeout=(30, 1800),
                )
        except requests.RequestException:
            if attempt < 4:
                continue
            raise

        if response.ok:
            return response.json()
        if response.status_code in (401,) or response.status_code >= 500:
            if attempt < 4:
                continue

        raise RuntimeError(
            f"B2 output upload failed: {response.status_code} {response.text[:1200]}"
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
        "vtt": f"{CONTROL_URL}/api/jobs/{JOB_ID}/captions/{vtt_path.name}" if vtt_path.is_file() else "",
        "has_vtt": vtt_path.is_file(),
    }


def upload_outputs():
    import app

    job_dir = Path(app.JOBS_ROOT) / JOB_ID
    outdir = job_dir / "clips"
    clips = sorted(outdir.glob("clip_*.mp4"))
    vtts = sorted(job_dir.glob("*.vtt"))

    results = []
    workers = min(4, max(1, len(clips)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(upload_one, path, job_dir) for path in clips]
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
    status("processing", 1, "Processing worker started.")
    status("processing", 5, "Downloading source from B2...")
    download_source()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import app

    info = app.probe(SOURCE)
    status("processing", 16, "Creating clips...")

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

    if not clips:
        raise RuntimeError("No clips were produced.")

    status("processing", 80, f"Uploading {len(clips)} finished clips...")
    result = upload_outputs()
    status("complete", 100, "Your clips are ready.", result)

    SOURCE.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"WORKER FAILED: {exc}", file=sys.stderr)
        try:
            status("error", 0, str(exc)[:700])
        finally:
            raise

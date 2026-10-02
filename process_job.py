import base64
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

CONTROL_URL = os.environ["CONTROL_URL"].rstrip("/")
WORKER_SECRET = os.environ["WORKER_SECRET"]
B2_KEY_ID = os.environ["B2_KEY_ID"]
B2_APPLICATION_KEY = os.environ["B2_APPLICATION_KEY"]
B2_BUCKET = os.environ["B2_BUCKET"]

job_id = os.environ["JOB_ID"]
payload = json.loads(os.environ["JOB_PAYLOAD"])

ROOT = Path("/tmp/clipshortener-worker")
CHUNKS = ROOT / "chunks"
JOBDIR = ROOT / "job"
SOURCE = ROOT / payload.get("filename", "input.mp4")

CHUNKS.mkdir(parents=True, exist_ok=True)
JOBDIR.mkdir(parents=True, exist_ok=True)

http = requests.Session()
b2 = None


def authorize_b2():
    global b2

    basic = base64.b64encode(
        f"{B2_KEY_ID}:{B2_APPLICATION_KEY}".encode("utf-8")
    ).decode("ascii")

    response = http.get(
        "https://api.backblazeb2.com/b2api/v4/b2_authorize_account",
        headers={"Authorization": f"Basic {basic}"},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    storage = data["apiInfo"]["storageApi"]

    allowed = (
        storage.get("allowed", {}).get("buckets")
        or data.get("allowed", {}).get("buckets", [])
    )

    bucket = next(
        (item for item in allowed if item.get("name") == B2_BUCKET),
        None,
    )

    if not bucket or not bucket.get("id"):
        raise RuntimeError(
            f'B2 bucket "{B2_BUCKET}" is unavailable to this key.'
        )

    b2 = {
        "api_url": storage["apiUrl"],
        "token": data["authorizationToken"],
        "bucket_id": bucket["id"],
    }


def b2_get_upload_url():
    if b2 is None:
        authorize_b2()

    response = http.get(
        f'{b2["api_url"]}/b2api/v4/b2_get_upload_url',
        params={"bucketId": b2["bucket_id"]},
        headers={"Authorization": b2["token"]},
        timeout=30,
    )

    if response.status_code == 401:
        authorize_b2()
        response = http.get(
            f'{b2["api_url"]}/b2api/v4/b2_get_upload_url',
            params={"bucketId": b2["bucket_id"]},
            headers={"Authorization": b2["token"]},
            timeout=30,
        )

    response.raise_for_status()
    return response.json()


def upload_b2_file(path: Path, key: str, content_type: str):
    sha1 = hashlib.sha1()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha1.update(block)

    for attempt in range(5):
        info = b2_get_upload_url()

        headers = {
            "Authorization": info["authorizationToken"],
            "X-Bz-File-Name": quote(key, safe="/"),
            "Content-Type": content_type,
            "X-Bz-Content-Sha1": sha1.hexdigest(),
            "Content-Length": str(path.stat().st_size),
        }

        try:
            with path.open("rb") as file:
                response = http.post(
                    info["uploadUrl"],
                    headers=headers,
                    data=file,
                    timeout=(30, 900),
                )

            if response.ok:
                return

            if response.status_code < 500 or attempt == 4:
                response.raise_for_status()

        except requests.RequestException:
            if attempt == 4:
                raise

        time.sleep(min(2 ** attempt, 15))

    raise RuntimeError(f"Upload failed after retries: {key}")


def status(state, progress=None, message=None, clips=None):
    data = {"status": state}

    if progress is not None:
        data["progress"] = progress

    if message is not None:
        data["message"] = message

    if clips is not None:
        data["clips"] = clips

    response = http.post(
        f"{CONTROL_URL}/api/jobs/{job_id}/worker-status",
        headers={
            "X-Worker-Secret": WORKER_SECRET,
            "content-type": "application/json",
        },
        json=data,
        timeout=30,
    )

    response.raise_for_status()


def download_source():
    total = int(payload["total_chunks"])

    with SOURCE.open("wb") as out:
        for i in range(total):
            response = http.get(
                f"{CONTROL_URL}/api/jobs/{job_id}/chunks/{i}",
                timeout=(30, 180),
            )
            response.raise_for_status()

            out.write(response.content)

            status(
                "processing",
                min(15, int((i + 1) / total * 15)),
                f"Preparing source ({i + 1}/{total})",
            )


def upload_outputs():
    clips = []
    outdir = JOBDIR / "clips"

    for path in sorted(outdir.glob("clip_*.mp4")):
        key = f"jobs/{job_id}/clips/{path.name}"

        upload_b2_file(
            path,
            key,
            "video/mp4",
        )

        clips.append({
            "name": path.name,
            "url": (
                f"{CONTROL_URL}/api/jobs/"
                f"{job_id}/download/{path.name}"
            ),
            "size": path.stat().st_size,
        })

    for path in sorted(outdir.glob("*.vtt")):
        key = f"jobs/{job_id}/captions/{path.name}"

        upload_b2_file(
            path,
            key,
            "text/vtt; charset=utf-8",
        )

    return clips


def main():
    authorize_b2()

    status(
        "processing",
        1,
        "Processing worker started.",
    )

    download_source()

    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent),
    )

    import app

    info = app.probe(SOURCE)

    status(
        "processing",
        18,
        "Creating clips...",
    )

    clips = app.create_clips(
        job_id,
        SOURCE,
        float(payload.get("clip_length", 30)),
        payload.get("export_format", "original"),
        float(payload.get("start", 0)),
        payload.get("end"),
        payload.get("captions"),
        info=info,
    )

    source_job_dir = Path(app.JOBS_ROOT) / job_id
    source_clip_dir = source_job_dir / "clips"

    outdir = JOBDIR / "clips"
    outdir.mkdir(parents=True, exist_ok=True)

    for path in source_clip_dir.glob("clip_*.mp4"):
        shutil.copy2(path, outdir / path.name)

    for path in source_job_dir.glob("*.vtt"):
        shutil.copy2(path, outdir / path.name)

    status(
        "processing",
        80,
        f"Uploading {len(clips)} finished clips...",
    )

    result = upload_outputs()

    status(
        "complete",
        100,
        "Your clips are ready.",
        result,
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            f"WORKER FAILED: {exc}",
            file=sys.stderr,
        )

        try:
            status(
                "error",
                0,
                str(exc)[:700],
            )
        finally:
            raise

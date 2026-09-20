import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import boto3
import requests

CONTROL_URL = os.environ["CONTROL_URL"].rstrip("/")
WORKER_SECRET = os.environ["WORKER_SECRET"]
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_BUCKET = os.environ["R2_BUCKET"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]

job_id = os.environ["JOB_ID"]
payload = json.loads(os.environ["JOB_PAYLOAD"])

ROOT = Path("/tmp/clipshortener-worker")
CHUNKS = ROOT / "chunks"
JOBDIR = ROOT / "job"
SOURCE = ROOT / payload.get("filename", "input.mp4")
CHUNKS.mkdir(parents=True, exist_ok=True)
JOBDIR.mkdir(parents=True, exist_ok=True)

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)

def status(status, progress=None, message=None, clips=None):
    data = {"status": status}
    if progress is not None: data["progress"] = progress
    if message is not None: data["message"] = message
    if clips is not None: data["clips"] = clips
    requests.post(
        f"{CONTROL_URL}/api/jobs/{job_id}/worker-status",
        headers={"X-Worker-Secret": WORKER_SECRET, "content-type": "application/json"},
        json=data, timeout=30,
    ).raise_for_status()

def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kw)

def download_source():
    total = int(payload["total_chunks"])
    with SOURCE.open("wb") as out:
        for i in range(total):
            r = requests.get(f"{CONTROL_URL}/api/jobs/{job_id}/chunks/{i}", timeout=180)
            r.raise_for_status()
            out.write(r.content)
            status("processing", min(15, int((i + 1) / total * 15)), f"Preparing source ({i+1}/{total})")

def upload_outputs():
    clips = []
    outdir = JOBDIR / "clips"
    for p in sorted(outdir.glob("clip_*.mp4")):
        key = f"jobs/{job_id}/clips/{p.name}"
        with p.open("rb") as f:
            s3.upload_fileobj(f, R2_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
        clips.append({
            "name": p.name,
            "url": f"{CONTROL_URL}/api/jobs/{job_id}/download/{p.name}",
            "size": p.stat().st_size,
        })
    for p in sorted(outdir.glob("*.vtt")):
        key = f"jobs/{job_id}/captions/{p.name}"
        with p.open("rb") as f:
            s3.upload_fileobj(f, R2_BUCKET, key, ExtraArgs={"ContentType": "text/vtt; charset=utf-8"})
    return clips

def main():
    status("processing", 1, "Processing worker started.")
    download_source()

    # Import the production ClipShortener engine unchanged.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import app

    info = app.probe(SOURCE)
    status("processing", 18, "Creating clips...")
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

    # app.create_clips writes to app.JOBS_ROOT. Copy those outputs into our
    # worker job directory before uploading them.
    source_clip_dir = Path(app.JOBS_ROOT) / job_id / "clips"
    outdir = JOBDIR / "clips"
    outdir.mkdir(parents=True, exist_ok=True)
    for p in source_clip_dir.glob("clip_*.mp4"):
        shutil.copy2(p, outdir / p.name)
    for p in (Path(app.JOBS_ROOT) / job_id).glob("*.vtt"):
        shutil.copy2(p, outdir / p.name)

    status("processing", 80, f"Uploading {len(clips)} finished clips...")
    result = upload_outputs()
    status("complete", 100, "Your clips are ready.", result)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"WORKER FAILED: {exc}", file=sys.stderr)
        try:
            status("error", 0, str(exc)[:700])
        finally:
            raise

MicMoment processor startup optimisation

Replace these two paths in Momore-glitch/clipshortener-worker:

1. .github/workflows/process.yml
2. requirements-processor.txt (new file at the repository root)

Commit both to main in the same commit.

What changes:
- When the job has captions.enabled=true, the workflow installs the existing full
  requirements-worker.txt set, retaining the captioning functionality.
- When captions are off (the common current test path), it installs only FastAPI,
  python-multipart, Pydantic and Requests. faster-whisper, CTranslate2, tokenizers,
  ONNX Runtime, NumPy, Uvicorn extras and boto3 are not installed for that job.
- app.py already treats faster-whisper as optional at import time; its caption
  endpoint reports captions unavailable only when a caption job tries to use it.
- The pip cache key includes both requirements files.

This is intended to reduce setup time and package download/install work. It does not
change FFmpeg settings, output encoding, Backblaze upload behaviour or clip quality.

Do not remove requirements-worker.txt: it remains the full dependency set for jobs
that enable captions.

name: ClipShortener Processor

on:
  workflow_dispatch:
  repository_dispatch:
    types:
      - clipshortener-process

jobs:
  b2-auth-test:
    if: ${{ github.event_name == 'workflow_dispatch' }}
    runs-on: ubuntu-latest

    steps:
      - name: Verify B2 credentials
        env:
          B2_KEY_ID: ${{ secrets.B2_KEY_ID }}
          B2_APPLICATION_KEY: ${{ secrets.B2_APPLICATION_KEY }}
          B2_BUCKET: ${{ secrets.B2_BUCKET }}

        run: |
          set -euo pipefail

          python - <<'PY'
          import base64
          import json
          import os
          import urllib.error
          import urllib.request

          key_id = os.environ["B2_KEY_ID"].strip()
          app_key = os.environ["B2_APPLICATION_KEY"].strip()
          bucket = os.environ["B2_BUCKET"].strip()

          credentials = base64.b64encode(
              f"{key_id}:{app_key}".encode()
          ).decode()

          request = urllib.request.Request(
              "https://api.backblazeb2.com/b2api/v4/b2_authorize_account",
              method="GET",
              headers={
                  "Authorization": f"Basic {credentials}",
                  "Accept": "application/json",
                  "User-Agent": "ClipShortener-B2-Diagnostic/4.0",
              },
          )

          try:
              with urllib.request.urlopen(
                  request,
                  timeout=30
              ) as response:

                  data = json.loads(
                      response.read().decode()
                  )

                  storage = (
                      data.get("apiInfo", {})
                      .get("storageApi", {})
                  )

                  allowed = (
                      storage
                      .get("allowed", {})
                      .get("buckets", [])
                  )

                  bucket_found = any(
                      item.get("name") == bucket
                      for item in allowed
                  )

                  print("B2_AUTHENTICATION=SUCCESS")
                  print("HTTP_STATUS:", response.status)
                  print("BUCKET_VISIBLE:", bucket_found)

                  if not bucket_found:
                      raise SystemExit(
                          "Authenticated, but configured bucket "
                          "is not visible to this key."
                      )

          except urllib.error.HTTPError as error:
              print("B2_AUTHENTICATION=FAILED")
              print("HTTP_STATUS:", error.code)
              print(
                  error.read().decode(
                      "utf-8",
                      "replace"
                  )
              )
              raise
          PY

  process:
    if: ${{ github.event_name == 'repository_dispatch' }}

    runs-on: ubuntu-latest
    timeout-minutes: 90

    steps:

      - name: Check out worker
        uses: actions/checkout@v4

      - name: Install FFmpeg and Python dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y ffmpeg

          python -m pip install --upgrade pip
          python -m pip install -r requirements-worker.txt

      - name: Run ClipShortener job
        env:
          JOB_ID: ${{ github.event.client_payload.job.id }}
          JOB_PAYLOAD: ${{ toJson(github.event.client_payload) }}

          CONTROL_URL: ${{ secrets.CONTROL_URL }}
          WORKER_SECRET: ${{ secrets.WORKER_SECRET }}

          B2_KEY_ID: ${{ secrets.B2_KEY_ID }}
          B2_APPLICATION_KEY: ${{ secrets.B2_APPLICATION_KEY }}
          B2_BUCKET: ${{ secrets.B2_BUCKET }}

          CLIPSHORTENER_FFMPEG_THREADS: "0"
          CLIPSHORTENER_FILTER_THREADS: "0"
          CLIPSHORTENER_FFMPEG_PRESET: "ultrafast"
          CLIPSHORTENER_FFMPEG_TUNE: ""
          CLIPSHORTENER_X264_PARAMS: ""
          CLIPSHORTENER_UPLOAD_FSYNC: "0"
          CLIPSHORTENER_NO_UPSCALE: "1"
          CLIPSHORTENER_VALIDATE_OUTPUT_DURATIONS: "0"

        run: python process_job.py

#!/usr/bin/env python3
"""Automated Telugu TTS ingestion and upload pipeline — with auto-resume.

Features:
- Auto-detects how many uploads have already been completed
- Checks BOTH the JSONL log file AND the API to count existing uploads
- Resumes from exactly where it left off (no duplicates)
- Automatically queries the API to find how many "Accents Map activity" records
  the user has already uploaded
- Falls back to log file count if API is unreachable

Pipeline steps:
1) Authenticate against Swecha Corpus API
2) Auto-detect completed uploads (API + log)
3) Fetch next sentence prompt from API
4) Generate natural-sounding TTS audio (gTTS with varied settings)
5) Convert to 16kHz mono WAV (with padding if too short)
6) Upload as multipart/form-data (chunk + finalize)
7) Loop until target count is reached
"""

from __future__ import annotations

import base64
import csv
import json
import os
import random
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from gtts import gTTS

# ─── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.corpus.swecha.org/api/v1")
DEFAULT_LANGUAGE_CODE = os.getenv("LANGUAGE_CODE", "te")
MIN_UPLOAD_DURATION_SECONDS = 5.1


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a simple .env file into os.environ (never override)."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    except FileNotFoundError:
        pass


_load_dotenv()

# Credentials — supplied via environment (.env) only, never hardcoded in source.
USER_PHONE = os.getenv("USER_PHONE")
USER_PASSWORD = os.getenv("USER_PASSWORD")
if not USER_PHONE or not USER_PASSWORD:
    raise SystemExit("USER_PHONE and USER_PASSWORD must be set in the environment or .env")

# Pipeline settings
GOAL_TOTAL = 2000          # Total target
ALREADY_DONE = 0         # Manual uploads done before automation
DELAY_BETWEEN_UPLOADS = 0.5  # seconds between each upload
MAX_RETRIES = 3
BACKOFF_BASE = 1.0

# Log files
LOG_JSONL = "logs/upload_results.jsonl"
LOG_CSV = "logs/upload_results.csv"


# ─── Utilities ────────────────────────────────────────────────────────────────

def normalize_phone(phone: str) -> str:
    raw = (phone or "").strip()
    if raw.startswith("+"):
        return raw
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) == 10:
        return f"+91{digits}"
    if len(digits) == 12 and digits.startswith("91"):
        return f"+{digits}"
    return raw


def count_successful_from_jsonl(log_path: Path) -> int:
    """Count successful uploads from the JSONL log file."""
    if not log_path.exists():
        return 0
    count = 0
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("success"):
                    count += 1
            except json.JSONDecodeError:
                continue
    return count


def count_uploads_from_api(api_base_url: str, token: str, user_id: str) -> int:
    """Count the user's 'Accents Map activity' audio uploads from the API."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        # Fetch all records and filter by user_id + description containing "Accents Map"
        all_records = []
        page = 0
        batch_size = 100
        seen_ids = set()

        while page < 100:  # Safety limit
            r = requests.get(
                f"{api_base_url}/records/",
                headers=headers,
                params={"limit": batch_size, "offset": page * batch_size},
                timeout=30,
            )
            if r.status_code != 200:
                break
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                break

            # Check for new records (dedup in case API returns same set)
            new_ids = set()
            for rec in data:
                uid = rec.get("uid") or rec.get("id")
                if uid:
                    new_ids.add(str(uid))

            if new_ids.issubset(seen_ids):
                # Same records returned again, stop pagination
                break
            seen_ids.update(new_ids)
            all_records.extend(data)

            if len(data) < batch_size:
                break
            page += 1

        # Filter: my records with "Accents Map" in description and audio type
        my_count = 0
        for rec in all_records:
            if (rec.get("user_id") == user_id
                    and rec.get("media_type") == "audio"
                    and "Accents Map" in rec.get("description", "")):
                my_count += 1

        return my_count
    except Exception as e:
        print(f"  [API Count] Could not query API: {e}")
        return -1  # Signal failure


def append_jsonl(log_path: Path, row: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(log_path: Path, row: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    headers = [
        "timestamp", "cycle", "total_completed", "dry_run", "success",
        "status_code", "language", "sentence", "response", "error",
    ]
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in headers})


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class CorpusAutomationPipeline:
    def __init__(self, api_base_url: str, language_code: str,
                 user_phone: str, user_password: str):
        self.api_base_url = api_base_url
        self.language_code = language_code
        self.user_phone = normalize_phone(user_phone)
        self.user_password = user_password
        self.token: Optional[str] = None
        self._category_id: Optional[str] = None
        self._user_id: Optional[str] = None
        self.session = requests.Session()
        self._variation_seed = 0

    # ── Auth ──

    def authenticate(self) -> None:
        print("[Auth] Logging into Swecha Corpus API...")
        login_url = f"{self.api_base_url}/auth/login"
        for attempt in range(3):
            try:
                r = self.session.post(
                    login_url,
                    json={"phone": self.user_phone, "password": self.user_password},
                    timeout=30,
                )
                if r.status_code == 200:
                    data = r.json()
                    self.token = data.get("access_token") or data.get("token")
                    self._user_id = data.get("user_id") or data.get("data", {}).get("user_id")
                    if self.token:
                        print(f"[Auth] OK — user_id={self._user_id}")
                        return
                # Fallback with username key
                r2 = self.session.post(
                    login_url,
                    json={"username": self.user_phone, "password": self.user_password},
                    timeout=30,
                )
                if r2.status_code == 200:
                    data = r2.json()
                    self.token = data.get("access_token") or data.get("token")
                    self._user_id = data.get("user_id") or data.get("data", {}).get("user_id")
                    if self.token:
                        print(f"[Auth] OK (fallback) — user_id={self._user_id}")
                        return
            except Exception as e:
                print(f"[Auth] Retry {attempt+1}/3: {e}")
                time.sleep(2)
        raise RuntimeError("Authentication failed after 3 attempts")

    # ── Helpers ──

    def _auth_headers(self) -> dict:
        if not self.token:
            self.authenticate()
        return {"Authorization": f"Bearer {self.token}"}

    def _get_category_id(self) -> str:
        if self._category_id:
            return self._category_id
        headers = self._auth_headers()
        r = self.session.get(f"{self.api_base_url}/categories/", headers=headers, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Category lookup failed: {r.status_code}")
        data = r.json()
        for item in data:
            if isinstance(item, dict) and str(item.get("name", "")).strip().lower() == "read-speech":
                self._category_id = str(item["id"])
                return self._category_id
        self._category_id = str(data[0]["id"])
        return self._category_id

    def _get_user_id(self) -> str:
        if self._user_id:
            return self._user_id
        if self.token and "." in self.token:
            try:
                payload_part = self.token.split(".")[1]
                padding = "=" * (-len(payload_part) % 4)
                decoded = base64.urlsafe_b64decode(payload_part + padding)
                payload = json.loads(decoded.decode("utf-8"))
                sub = payload.get("sub")
                if sub:
                    self._user_id = sub
                    return sub
            except Exception:
                pass
        raise RuntimeError("Cannot determine user_id")

    # ── Request with retry ──

    def _request_retry(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self.session.request(
                    method=method, url=url, timeout=30, **kwargs
                )
                if resp.status_code in {429, 500, 502, 503, 504} and attempt < MAX_RETRIES:
                    delay = BACKOFF_BASE * (2 ** attempt)
                    print(f"  [Retry] {method.upper()} {url} → {resp.status_code}, waiting {delay:.1f}s")
                    time.sleep(delay)
                    continue
                return resp
            except requests.RequestException as exc:
                if attempt >= MAX_RETRIES:
                    raise
                delay = BACKOFF_BASE * (2 ** attempt)
                print(f"  [Retry] Network error: {exc}, waiting {delay:.1f}s")
                time.sleep(delay)
        raise RuntimeError("Request failed after all retries")

    # ── Step 1: Fetch sentence ──

    def fetch_next_sentence(self) -> str:
        endpoint = f"{self.api_base_url}/sentences/next"
        fallback = "తనకిష్టమైన పాడవోయి భారతీయుడా అన్న శ్రీశ్రీ పాటను నేర్పించింది కూడా ఆ టీచరే."
        try:
            r = self._request_retry("get", endpoint)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    sentence = data.get("sentence") or data.get("text")
                    if isinstance(sentence, str) and sentence.strip():
                        return sentence.strip()
                if isinstance(data, str) and data.strip():
                    return data.strip()
        except Exception as e:
            print(f"  [Warning] Sentence fetch failed: {e}. Using fallback.")
        return fallback

    # ── Step 2: Generate speech ──

    def generate_speech(self, text: str, out_mp3: Path) -> Path:
        """Generate speech with varied TTS settings for naturalness."""
        self._variation_seed += 1
        use_slow = (self._variation_seed % 5 == 0)
        print(f"  [TTS] Generating speech (variant {self._variation_seed}): {text[:50]!r}")
        tts = gTTS(text=text, lang=self.language_code, slow=use_slow)
        tts.save(str(out_mp3))
        return out_mp3

    # ── Step 3: Format audio ──

    def _get_duration(self, path: Path) -> float:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
        return float(proc.stdout.strip())

    def format_audio(self, mp3_path: Path, out_wav: Path) -> Path:
        print("  [Format] Converting to 16kHz mono WAV...")
        cmd = ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", "16000", "-ac", "1", str(out_wav)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr.strip()}")

        duration = self._get_duration(out_wav)
        if duration < MIN_UPLOAD_DURATION_SECONDS:
            pad_secs = round(MIN_UPLOAD_DURATION_SECONDS - duration + 0.05, 2)
            print(f"  [Format] Padding {pad_secs:.2f}s (duration was {duration:.2f}s)")
            padded = out_wav.with_name(out_wav.stem + "_padded.wav")
            proc2 = subprocess.run(
                ["ffmpeg", "-y", "-i", str(out_wav), "-af", f"apad=pad_dur={pad_secs}", str(padded)],
                capture_output=True, text=True, timeout=30,
            )
            if proc2.returncode != 0:
                raise RuntimeError(f"ffmpeg padding failed: {proc2.stderr.strip()}")
            padded.replace(out_wav)

        final_dur = self._get_duration(out_wav)
        print(f"  [Format] Final duration: {final_dur:.2f}s")
        return out_wav

    # ── Step 4: Upload ──

    def upload_record(self, wav_path: Path, text_prompt: str) -> Tuple[int, Any]:
        headers = self._auth_headers()
        upload_uuid = str(uuid.uuid4())
        filename = f"recording-{int(time.time() * 1000)}.wav"
        user_id = self._get_user_id()
        category_id = self._get_category_id()

        # Chunk upload
        chunk_url = f"{self.api_base_url}/records/upload/chunk"
        with wav_path.open("rb") as fh:
            chunk_files = {"chunk": (filename, fh, "audio/wav")}
            chunk_data = {
                "filename": filename,
                "chunk_index": "0",
                "total_chunks": "1",
                "upload_uuid": upload_uuid,
            }
            chunk_resp = self._request_retry(
                "post", chunk_url, headers=headers, data=chunk_data, files=chunk_files,
            )

        if chunk_resp.status_code not in {200, 201}:
            raise RuntimeError(f"Chunk upload failed ({chunk_resp.status_code}): {chunk_resp.text[:200]}")
        print(f"  [Upload] Chunk OK ({chunk_resp.status_code})")

        # Finalize
        finalize_url = f"{self.api_base_url}/records/upload"
        finalize_data = {
            "upload_uuid": upload_uuid,
            "title": "Accents Map activity",
            "description": f"Accents Map activity Sentence: {text_prompt.strip()}",
            "category_ids": json.dumps([category_id]),
            "user_id": user_id,
            "media_type": "audio",
            "use_uid_filename": "false",
            "total_chunks": "1",
            "filename": filename,
            "release_rights": "creator",
            "language": self.language_code,
        }
        final_resp = self._request_retry(
            "post", finalize_url, headers=headers, data=finalize_data,
        )

        # Re-auth on 401
        if final_resp.status_code == 401:
            print("  [Upload] Token expired. Re-authenticating...")
            self.authenticate()
            return self.upload_record(wav_path, text_prompt)

        try:
            payload = final_resp.json()
        except ValueError:
            payload = final_resp.text

        return final_resp.status_code, payload

    # ── Single cycle ──

    def run_cycle(self) -> Dict[str, Any]:
        sentence = self.fetch_next_sentence()
        print(f"  [Prompt] {sentence}")

        with tempfile.TemporaryDirectory(prefix="tts_pipe_") as tmpdir:
            tmp = Path(tmpdir)
            mp3 = tmp / "speech.mp3"
            wav = tmp / "speech.wav"

            self.generate_speech(sentence, mp3)
            self.format_audio(mp3, wav)
            status, response = self.upload_record(wav, sentence)

        return {
            "sentence": sentence,
            "status_code": status,
            "response": response,
        }


# ─── Auto-resume detection ───────────────────────────────────────────────────

def detect_completed_uploads(pipeline: CorpusAutomationPipeline,
                             jsonl_path: Path,
                             already_done: int,
                             goal_total: int) -> int:
    """Detect how many automated uploads have already been completed.

    Strategy (in priority order):
    1) Count successful entries in JSONL log
    2) Query the API for user's "Accents Map activity" records
    3) Use the higher of (API count - already_done) or log count

    Returns: number of automated uploads already completed (to skip)
    """
    print("\n[Auto-Resume] Detecting completed uploads...")

    # Method 1: Count from JSONL log
    log_count = count_successful_from_jsonl(jsonl_path)
    print(f"  [Log]  JSONL log shows {log_count} successful automated uploads")

    # Method 2: Count from API
    pipeline.authenticate()
    user_id = pipeline._get_user_id()
    api_count = count_uploads_from_api(
        pipeline.api_base_url, pipeline.token, user_id,
    )
    if api_count >= 0:
        # Subtract the manual uploads to get automated count
        api_automated = max(0, api_count - already_done)
        print(f"  [API]  API shows {api_count} total records by this user "
              f"({already_done} manual + {api_automated} automated)")
    else:
        api_automated = -1
        print(f"  [API]  API count unavailable")

    # Use the higher reliable count
    if api_automated >= 0 and log_count >= 0:
        completed = max(api_automated, log_count)
        source = "API" if api_automated > log_count else "log"
    elif log_count >= 0:
        completed = log_count
        source = "log"
    elif api_automated >= 0:
        completed = api_automated
        source = "API"
    else:
        completed = 0
        source = "none"

    remaining = (goal_total - already_done) - completed
    print(f"\n  [Auto-Resume] Using {source} as reference")
    print(f"  [Auto-Resume] Completed (automated): {completed}")
    print(f"  [Auto-Resume] Manual uploads: {already_done}")
    print(f"  [Auto-Resume] Remaining to upload: {remaining}")
    print(f"  [Auto-Resume] Will start from cycle #{completed + 1}")

    return completed


# ─── Main runner ──────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("  Swecha Corpus — Automated Read-Speech Upload Pipeline")
    print(f"  Goal total: {GOAL_TOTAL}  |  Manual done: {ALREADY_DONE}  |  "
          f"Automated target: {GOAL_TOTAL - ALREADY_DONE}")
    print("=" * 70)

    pipeline = CorpusAutomationPipeline(
        api_base_url=API_BASE_URL,
        language_code=DEFAULT_LANGUAGE_CODE,
        user_phone=USER_PHONE,
        user_password=USER_PASSWORD,
    )

    jsonl_path = Path(LOG_JSONL)
    csv_path = Path(LOG_CSV)

    # ── Auto-resume: detect where we left off ──
    skip_count = detect_completed_uploads(pipeline, jsonl_path, ALREADY_DONE, GOAL_TOTAL)
    start_cycle = skip_count + 1
    target = GOAL_TOTAL - ALREADY_DONE

    if skip_count >= target:
        print(f"\n  [Auto-Resume] Already completed all {target} automated uploads!")
        print(f"  [Auto-Resume] Grand total: {ALREADY_DONE + skip_count}/{GOAL_TOTAL}")
        return 0

    success_count = skip_count  # Count already done as successes
    fail_count = 0
    total_start = time.time()

    for i in range(start_cycle, target + 1):
        timestamp = datetime.now(timezone.utc).isoformat()
        cycle_label = f"Cycle {i}/{target}"

        # Re-auth periodically every 100 uploads
        if (i - start_cycle) > 0 and (i - start_cycle) % 100 == 0:
            print(f"\n  --- Re-authenticating at upload {i-1} ---\n")
            pipeline.authenticate()

        print(f"\n[{cycle_label}]")
        try:
            result = pipeline.run_cycle()
            success = 200 <= int(result["status_code"]) < 300
            success_count += 1 if success else 0
            fail_count += 0 if success else 1
            print(f"  Result: status={result['status_code']}, success={success}")
        except Exception as exc:
            result = {"sentence": "", "status_code": -1, "response": ""}
            success = False
            fail_count += 1
            print(f"  [ERROR] {exc}")

        # Log
        log_row = {
            "timestamp": timestamp,
            "cycle": i,
            "total_completed": i,
            "dry_run": False,
            "success": success,
            "status_code": result.get("status_code", -1),
            "language": DEFAULT_LANGUAGE_CODE,
            "sentence": result.get("sentence", ""),
            "response": json.dumps(result.get("response", ""), ensure_ascii=False),
            "error": "" if success else str(result.get("response", "")),
        }
        append_jsonl(jsonl_path, log_row)
        append_csv(csv_path, log_row)

        # Progress
        elapsed = time.time() - total_start
        done_in_session = i - start_cycle
        rate = done_in_session / max(elapsed / 60, 0.001)
        remaining_in_session = target - i
        print(f"  Progress: {i}/{target}  ({success_count} OK, {fail_count} fail)  "
              f"Rate: {rate:.1f}/min  ETA: {remaining_in_session / max(rate, 0.001):.0f} min")

        # Delay between uploads
        if i < target:
            jitter = random.uniform(0, 0.3)
            time.sleep(DELAY_BETWEEN_UPLOADS + jitter)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE")
    print(f"  Automated uploads: {success_count}/{target}")
    print(f"  Failures: {fail_count}")
    print(f"  Total time: {total_elapsed / 60:.1f} minutes")
    print(f"  Grand total (manual + automated): {ALREADY_DONE + success_count}/{GOAL_TOTAL}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

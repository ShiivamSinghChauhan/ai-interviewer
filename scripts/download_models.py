"""Download the local models into ./models (resumable).

  python scripts/download_models.py                 # STT, VAD, turn detector, voice match (~680 MB on disk)
  python scripts/download_models.py --kokoro fp32   # + the offline fallback voice (~380 MB on disk)
  python scripts/download_models.py --stt 160ms     # a lower-latency STT variant

All models run on CPU via onnxruntime / sherpa-onnx.
Interrupted downloads resume from the .part file when you re-run the script.
"""
import argparse
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
HF = "https://huggingface.co"

FILES = {
    "silero_vad.onnx": f"{SHERPA}/asr-models/silero_vad_v5.onnx",
    "smart-turn-v3.2-cpu.onnx": f"{HF}/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx",
    # TitaNet-small separated voices far better than CAM++ / WeSpeaker in our tests (same >= 0.76, other <= 0.39)
    "speaker_titanet_small_en.onnx": f"{SHERPA}/speaker-recongition-models/nemo_en_titanet_small.onnx",
}
KOKORO = {
    "fp32": ("kokoro-multi-lang-v1_0", f"{SHERPA}/tts-models/kokoro-multi-lang-v1_0.tar.bz2"),
    "int8": ("kokoro-int8-multi-lang-v1_0", f"{SHERPA}/tts-models/kokoro-int8-multi-lang-v1_0.tar.bz2"),
}


def stt_archive(chunk: str) -> tuple[str, str]:
    name = f"sherpa-onnx-nemotron-speech-streaming-en-0.6b-{chunk}-int8-2026-04-25"
    return name, f"{SHERPA}/asr-models/{name}.tar.bz2"


def total_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        cr = r.headers.get("Content-Range", "")  # "bytes 0-0/12345"
        return int(cr.split("/")[-1]) if "/" in cr else int(r.headers.get("Content-Length") or 0)


def fetch(url: str, dest: Path, conns: int = 8) -> None:
    """Parallel ranged download (slow links often cap speed per connection), resumable."""
    part = dest.with_name(dest.name + ".part")
    try:
        total = total_size(url)
    except Exception:  # no size: fall back to a plain download
        total = 0
    have = part.stat().st_size if part.exists() else 0
    remaining = total - have
    if total <= 0 or remaining < 8 << 20 or conns <= 1:
        return fetch_single(url, dest)
    print(f"-> {url}  ({conns} connections, {remaining / 1e6:.0f} MB to go)")
    seg = -(-remaining // conns)
    ranges = [(have + i * seg, min(total, have + (i + 1) * seg) - 1) for i in range(conns)]
    segs = [dest.with_name(f"{dest.name}.seg{i}") for i in range(conns)]
    progress = [0] * conns
    errors: list[str] = []

    def worker(i: int) -> None:
        start, end = ranges[i]
        for attempt in range(30):
            got = segs[i].stat().st_size if segs[i].exists() else 0
            progress[i] = got
            if start + got > end:
                return
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + got}-{end}"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r, open(segs[i], "ab") as f:
                    while chunk := r.read(1 << 17):
                        f.write(chunk)
                        progress[i] += len(chunk)
            except Exception:  # retry this segment
                time.sleep(min(20, 2 + attempt))
        errors.append(f"segment {i} failed")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(conns)]
    for t in threads:
        t.start()
    t0 = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        done = sum(progress)
        sys.stdout.write(f"\r   {(have + done) / 1e6:7.1f} / {total / 1e6:.1f} MB  "
                         f"{done / max(time.time() - t0, 1) / 1e6:5.2f} MB/s ")
        sys.stdout.flush()
    print()
    if errors:
        raise SystemExit("; ".join(errors) + " - re-run the script to resume")
    with open(part, "ab") as out:
        for s in segs:
            out.write(s.read_bytes())
            s.unlink()
    if part.stat().st_size != total:
        raise SystemExit(f"size mismatch for {dest.name}; delete {part.name} and re-run")
    part.replace(dest)


def fetch_single(url: str, dest: Path, attempts: int = 20) -> None:
    part = dest.with_name(dest.name + ".part")
    print(f"-> {url}")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                if have and r.status != 206:  # server ignored Range: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                with open(part, "ab" if have else "wb") as f:
                    done, t0 = have, time.time()
                    while chunk := r.read(1 << 18):
                        f.write(chunk)
                        done += len(chunk)
                        rate = (done - have) / max(time.time() - t0, 1e-3) / 1e6
                        sys.stdout.write(f"\r   {done / 1e6:7.1f} / {total / 1e6:.1f} MB  {rate:5.2f} MB/s ")
                        sys.stdout.flush()
            print()
            part.replace(dest)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:  # already complete
                part.replace(dest)
                return
            print(f"\n   HTTP {e.code}, retry {attempt}/{attempts}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"\n   {type(e).__name__}: {e} - retry {attempt}/{attempts}")
        time.sleep(min(30, 2 * attempt))
    raise SystemExit(f"Failed to download {url}")


def extract(archive: Path) -> None:
    print(f"   extracting {archive.name}")
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(MODELS, filter="data")
    archive.unlink()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt", default="560ms", choices=["80ms", "160ms", "560ms", "1120ms"],
                    help="Nemotron chunk size: smaller = faster partials, slightly lower accuracy")
    ap.add_argument("--kokoro", choices=["none", "fp32", "int8"], default="none",
                    help="optional offline TTS. The default edge-tts voice needs no download; int8 Kokoro "
                         "measured RTF ~3.5 on a Ryzen 5 5500U (too slow), so prefer fp32 if you want it")
    args = ap.parse_args(argv)

    MODELS.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dest = MODELS / name
        if dest.exists():
            print(f"ok {name}")
        else:
            fetch(url, dest)

    archives = [stt_archive(args.stt)] + ([] if args.kokoro == "none" else [KOKORO[args.kokoro]])
    for name, url in archives:
        if (MODELS / name).is_dir():
            print(f"ok {name}/")
            continue
        archive = MODELS / (name + ".tar.bz2")
        if not archive.exists():
            fetch(url, archive)
        extract(archive)

    print("\nAll models ready in", MODELS)


if __name__ == "__main__":
    main()

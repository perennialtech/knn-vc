"""
### Usage

```bash
export CARTESIA_API_KEY="sk_car_..."

python cartesia_tts_sets.py --voice-id "YOUR_VOICE_ID"
```

With a grouped prompt file:

```text
[set_1]
The birch canoe slid on the smooth planks.

[set_2]
The boy was there when the sun rose.
```

```bash
python cartesia_tts_sets.py \
  --voice-id "YOUR_VOICE_ID" \
  --input prompts.txt \
  --out-dir cartesia_tts_out
```

Output:

```text
cartesia_tts_out/
  voice-YOUR_VOICE_ID/
    model-sonic-3.5/
      default/    # or named sets from [set] headers
        001_the-birch-canoe-slid-on-the-smooth-planks_<hash>.wav
      combined.wav
      manifest.json
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
import wave
from pathlib import Path

API_URL = "https://api.cartesia.ai/tts/bytes"
SAMPLE_RATE = 48000
SILENCE_MS = 100
CARTESIA_VERSION = "2026-03-01"

DEFAULT_TEXT = """
[set_1]
The birch canoe slid on the smooth planks.
Glue the sheet to the dark blue background.
It's easy to tell the depth of a well.
These days a chicken leg is a rare dish.
Rice is often served in round bowls.
The juice of lemons makes fine punch.
The box was thrown beside the parked truck.
The hogs were fed chopped corn and garbage.
Four hours of steady work faced us.
A large size in stockings is hard to sell.

[set_2]
The boy was there when the sun rose.
A rod is used to catch pink salmon.
The source of the huge river is the clear spring.
Kick the ball straight and follow through.
Help the woman get back to her feet.
A pot of tea helps to pass the evening.
Smoky fires lack flame and heat.
The soft cushion broke the man's fall.
The salt breeze came across from the sea.
The girl at the booth sold fifty bonds.
""".strip()


def slugify(text: str) -> str:
    return (
        re.sub(r"[^a-z0-9]+", "-", text.lower().replace("'", "").replace("’", ""))[
            :56
        ].strip("-")
        or "item"
    )


def safe_dir(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.")
    if not name or name in {".", ".."}:
        return "unknown"
    return name


def combine_wavs(wav_paths: list[Path], out_path: Path, silence_ms: int):
    with wave.open(str(wav_paths[0]), "rb") as f:
        params = f.getparams()

    silence = (
        b"\x00"
        * int(params.framerate * silence_ms / 1000)
        * params.nchannels
        * params.sampwidth
    )

    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        for i, p in enumerate(wav_paths):
            with wave.open(str(p), "rb") as src:
                out.writeframes(src.readframes(src.getnframes()))
            if i < len(wav_paths) - 1:
                out.writeframes(silence)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate grouped Cartesia TTS WAVs and a combined WAV."
    )
    parser.add_argument("--voice-id", required=True, help="Cartesia voice ID to use.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional grouped text file. Defaults to bundled sample prompts.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("cartesia_tts_out"))
    parser.add_argument("--model-id", default="sonic-3.5")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate files even when matching outputs exist.",
    )
    args = parser.parse_args()

    if not (api_key := os.environ.get("CARTESIA_API_KEY")):
        raise SystemExit("Missing CARTESIA_API_KEY")

    text = args.input.read_text("utf-8") if args.input else DEFAULT_TEXT

    sets: dict[str, list[str]] = {}
    current_set = "default"
    for line in filter(None, (l.strip() for l in text.splitlines())):
        if line.startswith("[") and line.endswith("]"):
            current_set = line[1:-1].strip() or "default"
        else:
            sets.setdefault(current_set, []).append(line)

    if not sets:
        raise SystemExit("No utterances found.")

    run_dir = (
        args.out_dir
        / f"voice-{safe_dir(args.voice_id)}"
        / f"model-{safe_dir(args.model_id)}"
    )
    wav_paths, manifest_sets = [], []

    for set_name, utterances in sets.items():
        set_dir = run_dir / safe_dir(set_name)
        set_dir.mkdir(parents=True, exist_ok=True)
        items = []

        for i, transcript in enumerate(utterances, start=1):
            payload = json.dumps(
                {
                    "model_id": args.model_id,
                    "transcript": transcript,
                    "voice": {"mode": "id", "id": args.voice_id},
                    "language": args.language,
                    "output_format": {
                        "container": "wav",
                        "encoding": "pcm_s16le",
                        "sample_rate": SAMPLE_RATE,
                    },
                },
                separators=(",", ":"),
            ).encode()

            req_sha = hashlib.sha256(payload).hexdigest()
            wav_path = set_dir / f"{i:03d}_{slugify(transcript)}_{req_sha[:12]}.wav"
            rel_wav = wav_path.relative_to(run_dir).as_posix()

            reused = not args.force and wav_path.exists()
            print(f"{'skip' if reused else 'tts '} {rel_wav}")

            if not reused:
                req = urllib.request.Request(
                    API_URL,
                    data=payload,
                    method="POST",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Cartesia-Version": CARTESIA_VERSION,
                        "Content-Type": "application/json",
                        "Accept": "audio/wav",
                    },
                )
                try:
                    with urllib.request.urlopen(req, timeout=60.0) as resp:
                        wav_path.write_bytes(resp.read())
                except urllib.error.HTTPError as e:
                    raise RuntimeError(
                        f"Cartesia HTTP {e.code}: {e.read().decode(errors='replace')}"
                    ) from e

            wav_paths.append(wav_path)
            items.append({"transcript": transcript, "wav": rel_wav, "reused": reused})

        manifest_sets.append({"set": set_name, "items": items})

    combined_path = run_dir / "combined.wav"
    print(f"join {combined_path.relative_to(run_dir).as_posix()}")
    combine_wavs(wav_paths, combined_path, SILENCE_MS)

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {"settings": vars(args), "sets": manifest_sets}, default=str, indent=2
        )
        + "\n",
        "utf-8",
    )

    print(
        f"\nwrote {len(wav_paths)} individual WAVs\ncombined WAV: {combined_path}\nmanifest: {run_dir / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()

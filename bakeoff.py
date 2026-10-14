"""Compare speech models on filler-word recall and speed. Nothing is written to disk.

Usage:
  .venv/bin/python bakeoff.py path/to/memo.m4a
  .venv/bin/python bakeoff.py --record 30        # record 30s from the mic, in memory only
  .venv/bin/python bakeoff.py memo.m4a --models small.en
"""

import argparse
import re
import time

import numpy as np
from faster_whisper import WhisperModel, decode_audio

SR = 16000
CHUNK_S = 3.0

# A prompt full of fillers nudges Whisper into transcribing them verbatim.
FILLER_PROMPT = "Umm, so, uh, I was like, you know, basically, um, thinking about it, uh."

# medium.en (~5s/chunk) and CrisperWhisper (~8s/chunk) are too slow on this Intel CPU.
MODELS = ["tiny.en", "base.en", "small.en"]

FILLER_RE = re.compile(
    r"\[(?:UM|UH)\]|\b(?:u+m+|u+h+|e+r+m*|a+h+|hmm+|mm+|like|so|basically|you know)\b",
    re.IGNORECASE,
)


def load_audio(args):
    if args.record:
        import sounddevice as sd

        print(f"Recording {args.record}s from the mic... talk naturally, fillers welcome.")
        audio = sd.rec(int(args.record * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        return audio[:, 0]
    return decode_audio(args.path, sampling_rate=SR)


def transcribe(model, audio, prompt):
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=1,
        temperature=0.0,
        initial_prompt=prompt,
        condition_on_previous_text=False,
        vad_filter=False,
        without_timestamps=True,
    )
    return " ".join(s.text.strip() for s in segments)


def mark(text):
    return FILLER_RE.sub(lambda m: f"\033[1;31m{m.group(0)}\033[0m", text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?")
    p.add_argument("--record", type=float)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    args = p.parse_args()
    if not args.path and not args.record:
        p.error("give an audio path or --record SECONDS")

    audio = load_audio(args)
    dur = len(audio) / SR
    chunk = audio[: int(CHUNK_S * SR)]
    print(f"Audio: {dur:.1f}s\n")

    results = []
    for name in args.models:
        print(f"=== {name} ===")
        t = time.perf_counter()
        model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=6)
        print(f"load: {time.perf_counter() - t:.1f}s")

        prompt = FILLER_PROMPT
        for _ in range(2):  # warm-up
            transcribe(model, chunk, prompt)

        t = time.perf_counter()
        transcribe(model, chunk, prompt)
        chunk_s = time.perf_counter() - t

        t = time.perf_counter()
        text = transcribe(model, audio, prompt)
        full_s = time.perf_counter() - t

        fillers = FILLER_RE.findall(text)
        print(mark(text))
        print(f"\nfillers found: {len(fillers)}  {fillers}")
        print(f"3s chunk: {chunk_s:.2f}s   full clip: {full_s:.1f}s ({dur / full_s:.1f}x real time)\n")
        results.append((name, len(fillers), chunk_s))
        del model

    print("=== summary ===")
    print(f"{'model':40} {'fillers':>8} {'3s chunk':>9}")
    for name, n, c in results:
        flag = "  ok for live" if c < 1.0 else "  too slow for live"
        print(f"{name:40} {n:>8} {c:>8.2f}s{flag}")


if __name__ == "__main__":
    main()

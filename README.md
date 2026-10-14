# um0

A macOS menu-bar app that flashes your screen when you say a filler word on a call.

It listens to your mic, transcribes short chunks of speech **locally** with Whisper
(via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)), and flashes when it hears
*um, uh, like, so, basically* or *you know*. The menu bar shows a running count.

## Privacy

- All processing happens on your Mac. Nothing is sent anywhere after the one-time model download.
- No transcripts are kept. Each chunk's audio and text are discarded right after checking.
- The only thing saved is your flash-brightness setting.

## Setup

Requires Python 3.12 (`brew install python@3.12`).

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install faster-whisper "av<16" sounddevice numpy rumps pyobjc-framework-Cocoa
./run.sh
```

On first run, macOS asks for microphone access, and the model (~150 MB) downloads once.
After that, it runs offline.

## Using it

- **Menu bar:** live filler count, pause/resume (turns the mic off), reset count, flash brightness.
- **Use headphones** on calls, or the mic will catch other people's fillers too.
- `./run.sh --show-text` prints transcripts to the terminal (not saved) to help tune the filler rules.

Settings like the model, loudness gate and filler patterns are at the top of `um0.py`.

## How it was built

I first tried Cohere Transcribe, but it drops filler words from its output, so it can't be used here.
`bakeoff.py` compares Whisper models on your own voice for filler recall and speed.
On an Intel i7 CPU, `base.en` processes a chunk in about 0.5s and was the best fit for live use.
`small.en` catches more fillers but is about 3x slower.

```sh
.venv/bin/python bakeoff.py --record 30
```

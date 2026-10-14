# um0

A macOS menu-bar app that flashes your screen when you say a filler word on a call.

It listens to your mic, transcribes short chunks of speech **locally** with Whisper
(via [faster-whisper](https://pypi.org/project/faster-whisper/)), and flashes when it hears
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
- **Use headphones** on calls. Through speakers, the mic also hears the other people on the call, so their fillers set off the flash too.
- `./run.sh --show-text` prints transcripts to the terminal (not saved) to help tune the filler rules.

Settings like the model, loudness gate and filler patterns are at the top of `um0.py`.

## How it was built

Cohere Transcribe was the first choice, but it drops filler words, as most speech-to-text models
do. Whisper keeps more of them when prompted with text that already contains fillers, so um0 uses
Whisper with that kind of prompt.

`bakeoff.py` compares Whisper models on a recording of your own voice:

```sh
.venv/bin/python bakeoff.py --record 30
```

### Speed vs. catching fillers

Results on an Intel i7 Mac (CPU only), 30 seconds of natural speech:

| Model | Time per chunk | Fillers caught |
|---|---|---|
| tiny.en | ~0.3–0.8s | 5 |
| **base.en** (default) | **~0.5s** | **5** |
| small.en | ~1.7s | 8 |

Bigger models catch more fillers but take longer, which delays the flash. With `base.en`, the
flash comes about 1–2 seconds after the filler. With `small.en`, it's about 2–4 seconds. um0
defaults to the faster option so the flash still feels tied to the word. Change `MODEL` in `um0.py`
to switch.

### Matching the model to the hardware

Faster hardware can run a bigger model live and catch more fillers. Only the Intel setup has been
tested. The rest are suggested starting points.

- **Intel Mac or CPU only:** `base.en` (this repo's default).
- **Apple Silicon Mac:** `small.en` or `medium.en`.
- **NVIDIA GPU:** [CrisperWhisper](https://huggingface.co/nyrahealth/faster_CrisperWhisper), which is
  trained to keep fillers (non-commercial license).
- **Cloud speech APIs:** some offer a filler-word option, but audio leaves the machine, so they
  don't fit um0's local-only design.

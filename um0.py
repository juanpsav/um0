"""um0: flashes the screen when you say a filler word.

Listens to the mic, transcribes short chunks locally with Whisper, and throws each
chunk's audio and text away right after checking it. Nothing is written to disk.

Run: ./run.sh            (add --show-text to print transcripts to the terminal for tuning)
"""

import queue
import re
import sys
import threading

import numpy as np
import rumps
import sounddevice as sd
from AppKit import (
    NSAnimationContext,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSColor,
    NSScreen,
    NSScreenSaverWindowLevel,
    NSUserDefaults,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
)
from faster_whisper import WhisperModel

# --- Settings ---------------------------------------------------------------

MODEL = "base.en"  # "small.en" catches more fillers but is ~3x slower
SR = 16000
MIN_CHUNK_S = 1.5  # wait for at least this much audio before transcribing
MAX_CHUNK_S = 4.0  # if we fall behind, keep only the most recent audio
SPEECH_RMS = 0.005  # loudness gate; raise it if background noise triggers flashes
MIN_SPEECH_S = 0.3  # chunk needs this much audio above the gate to count as speech
NO_SPEECH_PROB = 0.6  # Whisper's own "this was silence" score; skip segments above it

FLASH_RGB = (1.0, 0.25, 0.1)
FLASH_PEAK = 0.7  # default opacity at the start of the flash; adjustable from the menu
FLASH_LEVELS = [0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
FLASH_FADE_S = 0.6

# A prompt full of fillers nudges Whisper into transcribing them instead of dropping them.
FILLER_PROMPT = "Umm, so, uh, I was like, you know, basically, um, thinking about it, uh."

# "like", "so" and "you know" are only fillers in some positions, so they need a
# comma or sentence edge next to them. "I like it" and "so much" don't count.
FILLERS = [
    re.compile(r"\b(?:u+m+|u+h+|e+r+m+|hmm+)\b", re.I),
    re.compile(r"\bbasically\b", re.I),
    re.compile(r"(?:^|[,.?!]\s*)like\s*,|,\s*like\b", re.I),
    re.compile(r"(?:^|[.?!]\s*)so\s*,", re.I),
    re.compile(r"(?<!\bdo )(?<!\bif )\byou know\s*(?:[,.?!]|$)", re.I),
]

SHOW_TEXT = "--show-text" in sys.argv


def count_fillers(text):
    return sum(len(p.findall(text)) for p in FILLERS)


def looks_like_speech(audio):
    frame = int(0.03 * SR)
    n = len(audio) // frame
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    return (rms > SPEECH_RMS).sum() * 0.03 >= MIN_SPEECH_S


# --- Listening and transcribing (background thread) -------------------------


class Listener:
    def __init__(self, on_fillers):
        self.on_fillers = on_fillers
        self.audio_q = queue.Queue()
        self.stream = None
        try:  # use the downloaded copy so we never touch the network after the first run
            self.model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=6,
                                      local_files_only=True)
        except Exception:
            self.model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=6)
        threading.Thread(target=self._run, daemon=True).start()

    def start(self):
        self.stream = sd.InputStream(
            samplerate=SR, channels=1, dtype="float32", callback=self._on_audio
        )
        self.stream.start()

    def stop(self):
        # Closing the stream releases the mic, so macOS's mic indicator turns off.
        if self.stream:
            self.stream.close()
            self.stream = None
        with self.audio_q.mutex:
            self.audio_q.queue.clear()

    def _on_audio(self, indata, frames, time, status):
        self.audio_q.put(indata[:, 0].copy())

    def _next_chunk(self):
        parts, n = [], 0
        while n < MIN_CHUNK_S * SR:
            part = self.audio_q.get()
            parts.append(part)
            n += len(part)
        while not self.audio_q.empty():
            parts.append(self.audio_q.get_nowait())
        return np.concatenate(parts)[-int(MAX_CHUNK_S * SR):]

    def _run(self):
        while True:
            audio = self._next_chunk()
            if not looks_like_speech(audio):
                continue
            segments, _ = self.model.transcribe(
                audio,
                language="en",
                beam_size=1,
                temperature=0.0,
                initial_prompt=FILLER_PROMPT,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < NO_SPEECH_PROB)
            n = count_fillers(text)
            if SHOW_TEXT:
                print(f"[{n}] {text}", flush=True)
            del audio, text
            if n:
                self.on_fillers(n)


# --- Screen flash (main thread) ---------------------------------------------


class Flasher:
    def __init__(self, peak):
        self.peak = peak
        self.windows, self.frames = [], None

    def _build(self):
        frames = [tuple(map(tuple, s.frame())) for s in NSScreen.screens()]
        if frames == self.frames:
            return
        for w in self.windows:
            w.close()
        self.windows, self.frames = [], frames
        for screen in NSScreen.screens():
            w = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                screen.frame(), NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
            )
            w.setReleasedWhenClosed_(False)
            w.setOpaque_(False)
            w.setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(*FLASH_RGB, 1.0))
            w.setAlphaValue_(0.0)
            w.setIgnoresMouseEvents_(True)
            w.setLevel_(NSScreenSaverWindowLevel)
            w.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
                | NSWindowCollectionBehaviorFullScreenAuxiliary
                | NSWindowCollectionBehaviorIgnoresCycle
            )
            w.orderFrontRegardless()
            self.windows.append(w)

    def flash(self):
        self._build()
        for w in self.windows:
            w.setAlphaValue_(self.peak)
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(FLASH_FADE_S)
        for w in self.windows:
            w.animator().setAlphaValue_(0.0)
        NSAnimationContext.endGrouping()


# --- Menu bar app -----------------------------------------------------------


class App(rumps.App):
    def __init__(self):
        super().__init__("um0", title="um 0", quit_button=None)
        self.count = 0
        self.pending = queue.Queue()
        # Only the brightness setting is saved (~/Library/Preferences/com.um0.plist).
        self.prefs = NSUserDefaults.alloc().initWithSuiteName_("com.um0")
        self.flasher = Flasher(self.prefs.floatForKey_("flashPeak") or FLASH_PEAK)
        self.toggle = rumps.MenuItem("Pause listening", callback=self.on_toggle)
        brightness = rumps.MenuItem("Flash brightness")
        for level in FLASH_LEVELS:
            item = rumps.MenuItem(f"{round(level * 100)}%", callback=self.on_brightness)
            item.level = level
            item.state = abs(level - self.flasher.peak) < 0.01
            brightness.add(item)
        self.brightness = brightness
        self.menu = [self.toggle, rumps.MenuItem("Reset count", callback=self.on_reset),
                     brightness, None, rumps.MenuItem("Quit", callback=rumps.quit_application)]
        self.listener = Listener(on_fillers=self.pending.put)
        self.listener.start()
        rumps.Timer(self.drain, 0.05).start()

    def drain(self, _):
        # Fillers are found on a background thread; flashing must happen on the main thread.
        hits = 0
        while not self.pending.empty():
            hits += self.pending.get_nowait()
        if hits:
            self.count += hits
            self.title = f"um {self.count}"
            self.flasher.flash()

    def on_toggle(self, item):
        if self.listener.stream:
            self.listener.stop()
            item.title = "Resume listening"
            self.title = f"⏸ {self.count}"
        else:
            self.listener.start()
            item.title = "Pause listening"
            self.title = f"um {self.count}"

    def on_brightness(self, item):
        for other in self.brightness.values():
            other.state = other is item
        self.flasher.peak = item.level
        self.prefs.setFloat_forKey_(item.level, "flashPeak")
        self.flasher.flash()  # preview

    def on_reset(self, _):
        self.count = 0
        self.title = "um 0" if self.listener.stream else "⏸ 0"


if __name__ == "__main__":
    app = App()
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon
    app.run()

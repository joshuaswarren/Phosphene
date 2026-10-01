#!/usr/bin/env python3
"""Separate the sung vocal out of a song, for a2v "Listen to the voice only".

Runs in the ENGINE venv (the one the helper uses): demucs + torch live there,
installed by scripts/pinokio/a2v_stems_deps.sh from install.js and from every
Update (scripts/post_update.sh). The panel calls this as a tracked subprocess
so Stop reaches it.

WHY A RUNNER AND NOT THE `demucs` CLI. The CLI reads and writes audio through
torchaudio, and torchaudio 2.9+ moved load/save onto torchcodec (a native
package with its own FFmpeg ABI rules). demucs 4.0.1's `save_audio` therefore
dies on a current torchaudio after it has spent the whole separation. This
runner uses demucs only for the MODEL: audio comes in through the panel's own
ffmpeg and goes out through the stdlib `wave` module, so no audio I/O library
can break it.

    a2v_separate.py --in SONG --out VOCALS.wav [--ffmpeg PATH] [--model NAME]
    a2v_separate.py --prefetch [--model NAME]      # download the weights, exit

Model weights (~80 MB for htdemucs) are cached by torch.hub under $TORCH_HOME;
the panel points TORCH_HOME at mlx_models/demucs so they live with the rest of
Phosphene's weights. Exit 0 = the vocal file is written and non-empty.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import wave

SAMPLE_RATE = 44100          # htdemucs' own rate; the input is resampled to it
DEFAULT_MODEL = "htdemucs"


def _scrub_env() -> None:
    # The engine venv is shared with MLX work; nothing here needs the MPS
    # fallback Pinokio injects, and demucs runs on the CPU on purpose (the GPU
    # belongs to the render this separation is the first step of).
    os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    os.environ.pop("PYTORCH_MPS_FAST_MATH", None)


def load_model(name: str):
    from demucs.pretrained import get_model          # noqa: PLC0415
    model = get_model(name)
    model.eval()
    return model


def read_audio(path: str, ffmpeg: str):
    import numpy as np                               # noqa: PLC0415
    import torch                                     # noqa: PLC0415
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-i", path, "-f", "f32le",
         "-ac", "2", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-1:]
        raise RuntimeError(f"could not read the song ({tail[0] if tail else 'ffmpeg wrote nothing'})")
    pcm = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2).T.copy()
    return torch.from_numpy(pcm)


def write_wav(path: str, audio) -> None:
    import numpy as np                               # noqa: PLC0415
    pcm = np.clip(audio.numpy().T, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    tmp = path + ".part"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(pcm16.shape[1])
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())
    os.replace(tmp, path)


def separate(src: str, dst: str, ffmpeg: str, model_name: str) -> None:
    import torch                                     # noqa: PLC0415
    from demucs.apply import apply_model             # noqa: PLC0415
    model = load_model(model_name)
    wav = read_audio(src, ffmpeg)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    std = std if float(std) > 1e-8 else torch.tensor(1.0)
    with torch.no_grad():
        sources = apply_model(model, ((wav - mean) / std)[None], device="cpu",
                              shifts=1, split=True, overlap=0.25, progress=False)[0]
    vocals = sources[model.sources.index("vocals")] * std + mean
    write_wav(dst, vocals)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src")
    ap.add_argument("--out", dest="dst")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prefetch", action="store_true")
    a = ap.parse_args(argv)
    _scrub_env()
    if a.prefetch:
        load_model(a.model)
        print(f"vocal separator ready ({a.model})")
        return 0
    if not a.src or not a.dst:
        ap.error("--in and --out are required (or --prefetch)")
    separate(a.src, a.dst, a.ffmpeg, a.model)
    if not os.path.isfile(a.dst) or os.path.getsize(a.dst) <= 44:
        print("the separator wrote no audio", file=sys.stderr)
        return 1
    print(f"vocals written: {a.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

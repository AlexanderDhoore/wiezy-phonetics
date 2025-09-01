#!/usr/bin/env python3
import argparse
import json
import os
import tempfile

import numpy as np
import soundfile as sf  # pip install soundfile

def mm_ss_min(t: float) -> str:
    """Format seconds -> 'X:YY' (minutes + zero-padded seconds)."""
    if t < 0:
        t = 0
    m = int(t // 60)
    s = int(t - 60 * m)
    return f"{m:02d}:{s:02d}"

def slice_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Return audio[start:end] as float32 mono."""
    s = max(0, int(round(start * sr)))
    e = min(len(audio), int(round(end * sr)))
    if e <= s:
        return np.zeros(0, dtype=np.float32)
    return audio[s:e].astype(np.float32, copy=False)

def main():
    ap = argparse.ArgumentParser(
        description="Transcribe each merged segment with Whisper (Dutch) and Allosaurus (audio→IPA)."
    )
    ap.add_argument("--wav", default="input.wav", help="Mono 16 kHz WAV.")
    ap.add_argument("--segments", default="segments.json", help="Merged segments JSON.")
    ap.add_argument("--out", default="transcript_both.txt", help="Output text file.")

    # Whisper settings
    ap.add_argument("--whisper_model", default="large-v3",
                    help="Whisper model size (tiny/base/small/medium/large-v3). Default: large-v3")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="Force device (default: auto).")

    # Allosaurus settings
    ap.add_argument("--allosaurus_lang", default="ipa",
                    help="Allosaurus language/inventory id (e.g., ipa, nld). Default: ipa")
    ap.add_argument("--allosaurus_model", default="latest",
                    help="Allosaurus model name. Default: latest")

    ap.add_argument("--print_empty", action="store_true",
                    help="Include lines even if the transcription is empty.")
    args = ap.parse_args()

    # Load audio
    audio, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    if sr != 16000:
        raise SystemExit(f"Expected 16 kHz audio, got {sr} Hz. (Your file: {args.wav})")

    # Load segments
    with open(args.segments, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # --- Whisper setup ---
    try:
        import torch
        import whisper  # pip install openai-whisper
    except Exception as e:
        raise SystemExit(
            "Missing dependency. Install with:\n  pip install openai-whisper soundfile\n"
            f"Details: {e}"
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    wmodel = whisper.load_model(args.whisper_model, device=device)

    # --- Allosaurus setup ---
    try:
        from allosaurus.app import read_recognizer  # pip install allosaurus
    except Exception as e:
        raise SystemExit(
            "Missing dependency. Install with:\n  pip install allosaurus soundfile\n"
            f"Details: {e}"
        )
    arec = read_recognizer(args.allosaurus_model)

    # Process each segment
    lines = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        spk = seg.get("speaker", "unknown")
        tstamp = mm_ss_min(start)

        clip = slice_audio(audio, sr, start, end)
        if clip.size == 0:
            if args.print_empty:
                lines.append(f"{tstamp} | {spk} | ")
                lines.append(f"{tstamp} | {spk} | ")
            continue

        # Whisper (Dutch text)
        wres = wmodel.transcribe(
            clip,
            language="nl",
            task="transcribe",
            fp16=(device == "cuda"),
            temperature=0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            verbose=False,
        )
        wtext = (wres.get("text") or "").strip()
        if wtext or args.print_empty:
            lines.append(f"{tstamp} | {spk} | {wtext}")

        # Allosaurus (IPA) — expects a file path, so write temp wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, clip, sr, subtype="PCM_16")
            ipa = arec.recognize(tmp_path, args.allosaurus_lang) or ""
            ipa = ipa.strip()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if ipa or args.print_empty:
            lines.append(f"{tstamp} | {spk} | {ipa}")

    # Write all lines
    with open(args.out, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Wrote {len(lines)} lines to {args.out}")

if __name__ == "__main__":
    main()


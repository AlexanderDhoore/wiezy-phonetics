#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict

import numpy as np
import soundfile as sf  # pip install soundfile

def parse_rttm(rttm_path):
    """
    Returns dict: {speaker_label: [(start_sec, dur_sec), ...]} sorted by start.
    RTTM columns (NIST): TYPE FILE CHAN START DUR ORT STY NAME CONF SLAT
    We need START (idx 3), DUR (idx 4), NAME (idx 7).
    """
    segments = defaultdict(list)
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue
            try:
                start = float(parts[3])
                dur = float(parts[4])
                spk = parts[7]
            except (ValueError, IndexError):
                continue
            if dur <= 0:
                continue
            segments[spk].append((start, dur))
    # sort by start time per speaker
    for spk in segments:
        segments[spk].sort(key=lambda x: x[0])
    return segments

def sec_to_samples(t, sr):
    return int(round(t * sr))

def concat_segments(audio, sr, segs, padding_ms=0.0):
    """
    Extract and concatenate segments from `audio` (numpy float array).
    Optional symmetric padding per segment (clipped to audio bounds).
    """
    pad = int(round((padding_ms / 1000.0) * sr))
    chunks = []
    n = len(audio)
    for start_sec, dur_sec in segs:
        s = max(0, sec_to_samples(start_sec, sr) - pad)
        e = min(n, s + sec_to_samples(dur_sec, sr) + 2 * pad)
        if e > s:
            chunks.append(audio[s:e])
    if not chunks:
        return np.zeros((0,), dtype=audio.dtype)
    return np.concatenate(chunks, axis=0)

def sanitize_filename(name):
    # safe-ish filename from label
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)

def main():
    p = argparse.ArgumentParser(description="Create per-speaker stitched WAVs from RTTM.")
    p.add_argument("--wav", default="input.wav", help="Path to input mono WAV (16kHz).")
    p.add_argument("--rttm", default="outputs/pred_rttms/input.rttm", help="Path to RTTM file.")
    p.add_argument("--outdir", default=".", help="Directory to write output WAVs.")
    p.add_argument("--padding_ms", type=float, default=0.0,
                   help="Optional padding added before/after each segment (ms).")
    args = p.parse_args()

    # Load audio
    audio, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if audio.ndim != 1:
        # Make mono if needed
        audio = audio.mean(axis=1)
    if sr <= 0:
        raise RuntimeError("Invalid sample rate.")
    # Parse RTTM
    segs_by_spk = parse_rttm(args.rttm)
    if not segs_by_spk:
        raise RuntimeError(f"No usable segments found in RTTM: {args.rttm}")

    # Prefer NeMo labels 'speaker_0' and 'speaker_1'
    labels = list(segs_by_spk.keys())
    # Stable order: if labels match speaker_\d+, sort by the number; else alphabetically
    def sort_key(lbl):
        m = re.match(r"speaker_(\d+)$", lbl)
        return (0, int(m.group(1))) if m else (1, lbl)
    labels.sort(key=sort_key)

    # Build mapping to desired filenames
    out_map = {}
    for lbl in labels:
        if lbl == "speaker_0":
            out_map[lbl] = "speaker0.wav"
        elif lbl == "speaker_1":
            out_map[lbl] = "speaker1.wav"
        else:
            out_map[lbl] = f"{sanitize_filename(lbl)}.wav"

    os.makedirs(args.outdir, exist_ok=True)

    for lbl in labels:
        stitched = concat_segments(audio, sr, segs_by_spk[lbl], padding_ms=args.padding_ms)
        out_path = os.path.join(args.outdir, out_map[lbl])
        # Write PCM 16-bit
        sf.write(out_path, stitched, sr, subtype="PCM_16")
        total_sec = len(stitched) / float(sr) if len(stitched) else 0.0
        print(f"Wrote {out_path}  ({total_sec:.2f}s, {len(segs_by_spk[lbl])} segments)")

    print("Done.")

if __name__ == "__main__":
    main()


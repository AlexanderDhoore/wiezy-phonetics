#!/usr/bin/env python3
import argparse
import json
import numpy as np
import soundfile as sf
import torch
import whisper
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def mm_ss(t: float) -> str:
    """Format seconds -> 'MM:SS' (zero-padded)."""
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
        description="Per-segment transcription: Whisper (text) + Phoneme model (IPA)."
    )
    ap.add_argument("--wav", default="input.wav", help="Mono 16 kHz WAV.")
    ap.add_argument("--segments", default="segments.json", help="Merged segments JSON.")
    ap.add_argument("--out", default="transcript.txt", help="Output text file.")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="Force device (default: auto).")
    ap.add_argument("--whisper_model", default="large-v3",
                    help="Whisper model (tiny/base/small/medium/large-v3). Default: large-v3")
    ap.add_argument("--phoneme_model",
                    default="facebook/wav2vec2-lv-60-espeak-cv-ft",
                    help="Phoneme CTC model. For Dutch-only, try "
                         "'Clementapa/wav2vec2-base-960h-phoneme-reco-dutch'.")
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

    # Whisper setup
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    wmodel = whisper.load_model(args.whisper_model, device=device)

    # Phoneme model setup
    ph_processor = Wav2Vec2Processor.from_pretrained(args.phoneme_model)
    ph_model = Wav2Vec2ForCTC.from_pretrained(args.phoneme_model).to(device)
    ph_model.eval()

    # Process each segment, write lines to file
    with open(args.out, "w", encoding="utf-8") as f:
        for seg in segments:
            start = float(seg["start"])
            end = float(seg["end"])
            spk = seg["speaker"]
            tstamp = mm_ss(start)

            clip = slice_audio(audio, sr, start, end)
            if clip.size == 0:
                continue

            # Whisper model (text)
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
            f.write(f"{tstamp} | {spk} | {wtext}\n")

            # Phoneme model (IPA)
            with torch.no_grad():
                inputs = ph_processor(
                    clip,
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding="longest"
                )
                input_values = inputs.input_values.to(device)
                attention_mask = getattr(inputs, "attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                logits = ph_model(input_values, attention_mask=attention_mask).logits
                pred_ids = logits.argmax(dim=-1)
                decoded = ph_processor.batch_decode(pred_ids)[0]  # string of symbols (often space-separated)
                ipa = (decoded or "").strip()

            f.write(f"{tstamp} | {spk} | {ipa}\n")
            f.flush()

    print(f"DONE")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json

def parse_rttm(rttm_path):
    """
    Parse RTTM and return a list of dicts:
    { "file": <recording_id>, "start": float, "end": float, "speaker": <label> }
    """
    segs = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # RTTM: TYPE FILE CHAN START DUR ORT STY NAME CONF SLAT
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue
            rec_id = parts[1]
            try:
                start = float(parts[3]); dur = float(parts[4])
            except ValueError:
                continue
            if dur <= 0:
                continue
            speaker = parts[7]
            segs.append({
                "file": rec_id,
                "start": start,
                "end": start + dur,
                "speaker": speaker,
            })
    segs.sort(key=lambda s: (s["file"], s["start"], s["end"]))
    return segs

def merge_runs_always(segs):
    """
    Merge *consecutive* segments from the same speaker within the same file,
    regardless of any silent gap between them. (Silence is absorbed.)
    """
    merged = []
    cur = None
    for s in segs:
        if cur is None:
            cur = s.copy()
            continue
        if s["file"] == cur["file"] and s["speaker"] == cur["speaker"]:
            # extend to the latest end (absorbs any gap or overlap)
            if s["end"] > cur["end"]:
                cur["end"] = s["end"]
        else:
            merged.append(cur)
            cur = s.copy()
    if cur is not None:
        merged.append(cur)
    return merged

def drop_file_if_single_recording(segments):
    recs = {s["file"] for s in segments}
    if len(recs) == 1:
        for s in segments:
            s.pop("file", None)
    return segments

def main():
    ap = argparse.ArgumentParser(description="RTTM → merged speaker segments (always merge consecutive same-speaker runs).")
    ap.add_argument("--rttm", default="outputs/pred_rttms/input.rttm", help="Path to RTTM file.")
    ap.add_argument("--out", default="segments.json", help="Path to output JSON.")
    args = ap.parse_args()

    segs = parse_rttm(args.rttm)
    if not segs:
        raise SystemExit(f"No SPEAKER lines found in {args.rttm}")

    merged = merge_runs_always(segs)
    merged = drop_file_if_single_recording(merged)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    total = sum(s["end"] - s["start"] for s in merged)
    print(f"Wrote {args.out} with {len(merged)} merged segments (total span {total:.2f}s).")

if __name__ == "__main__":
    main()


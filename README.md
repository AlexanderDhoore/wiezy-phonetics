
# Wiezy Phonetics

Transcribe logopedic sessions into IPA for testing schoolchildren.

## Dump of commands (TODO cleanup)

```bash
docker run --gpus all -it -v .:/workspace -w /workspace \
    --shm-size=4g nvcr.io/nvidia/nemo:25.07

apt update && apt install ffmpeg -y

# Convert audio to 16 kHz mono WAV
ffmpeg -i input.m4a -ac 1 -ar 16000 input.wav

# Split up the speakers (diarization)
python diarize.py
# -> outputs/pred_rttms/input.rttm

# Turn the rttm file into a json, which has merged segments per speaker
python rttm_to_segments.py

# Transcribe the speaker segments into text, plus IPA phonetics
python transcribe_segments.py --phoneme_model Clementapa/wav2vec2-base-960h-phoneme-reco-dutch
```

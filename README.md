
# Wiezy Phonetics

Transcribe logopedic sessions into IPA for testing schoolchildren.

## Dump of commands (TODO cleanup)

```bash
docker run --gpus all -it -v .:/workspace -w /workspace \
    --shm-size=4g nvcr.io/nvidia/nemo:25.07

apt update && apt install ffmpeg -y

# convert audio to 16 kHz mono WAV
ffmpeg -i input.m4a -ac 1 -ar 16000 input.wav

python diarize.py
```

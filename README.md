
# Wiezy Phonetics

Transcribe logopedic sessions into the IPA (International Phonetic Alphabet) for testing with schoolchildren.

* `input.wav`: an audio recording of a therapist and a child having a conversation.
* `diarize.py`: turns the audio into segments indicating who spoke when (RTTM file).
* `rttm_to_segments.py`: reads the RTTM file, merges same-speaker segments, and outputs `segments.json`.
* `transcribe_segments.py`: transcribes the audio in both natural language and IPA.

## Dump of commands

```bash
docker run --gpus all -it -v .:/workspace -w /workspace \
    --shm-size=4g nvcr.io/nvidia/nemo:25.07

apt update && apt install ffmpeg -y
pip install openai-whisper phonemizer

# Convert audio to 16 kHz mono WAV
ffmpeg -i input.m4a -ac 1 -ar 16000 input.wav

# Split up the speakers (diarization)
python diarize.py
# -> outputs/pred_rttms/input.rttm

# Turn the rttm file into a json, which has merged segments per speaker
python rttm_to_segments.py

# Transcribe the speaker segments into text, plus IPA phonetics
python transcribe_segments.py

# Or for a Dutch-only model (although I'm not sure if it's actually better):
python transcribe_segments.py --phoneme_model Clementapa/wav2vec2-base-960h-phoneme-reco-dutch
```

# Voice Trainer Pipeline

This repository builds a reproducible single-sample voice-training pipeline around three local reference repositories:

- `./.ref/faster-qwen3-tts`: voice cloning candidate generation
- `./.ref/speechbrain`: speaker embedding extraction and ranking
- `./.ref/vits`: small TTS model preprocessing and training

## Pipeline

1. Read one reference waveform and its transcript.
2. Generate a large Korean sentence set.
3. Synthesize candidate utterances with voice cloning.
4. Compute speaker embeddings for the reference and candidates.
5. Keep the top-N candidates by cosine similarity.
6. Export the selected subset as a VITS dataset.
7. Preprocess and train VITS with reproducible config files.

## Current Assumptions

- Input is a single Korean reference clip plus one Korean transcript.
- Candidate prompts are generated locally from templates in this repository.
- Candidate ranking uses SpeechBrain ECAPA embeddings and cosine similarity.
- VITS is trained as a single-speaker model from the selected synthetic set.

## Usage

The main entry point will be:

```bash
python3 scripts/run_pipeline.py --config configs/kim_haru_pipeline.json
```

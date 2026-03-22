# Voice Trainer Pipeline

This repository builds a reproducible single-sample voice-training pipeline guided by three local reference repositories:

- `./.ref/faster-qwen3-tts`: design reference for voice cloning candidate generation
- `./.ref/speechbrain`: design reference for speaker embedding extraction and ranking
- `./.ref/vits`: design reference for small TTS preprocessing and training

The runtime pipeline in this repository does not import or execute code from `./.ref`.

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

By default the repository trains its own compact VITS-style model locally.
`vits.training_command` is optional and only needed if you want to replace that final trainer with a different command.

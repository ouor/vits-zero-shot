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
6. Export the selected subset as a reusable training corpus.
7. Let the selected training backend prepare its own assets and train from reproducible config files.

## Current Assumptions

- Input is a single Korean reference clip plus one Korean transcript.
- Candidate prompts are generated locally from templates in this repository.
- Candidate ranking uses SpeechBrain ECAPA embeddings and cosine similarity.
- The default backend is the vendored local full VITS implementation.
- The current VITS path uses the multi-speaker model structure while assigning a single selected speaker id in the exported corpus.

## Setup

```bash
uv sync
```

The project targets Python 3.11 because the current `faster-qwen3-tts` dependency chain pulls `onnxruntime`, which does not provide Python 3.10 wheels for this environment.

## Usage

The main entry point is:

```bash
uv run voice-trainer-run --config configs/kim_haru_pipeline.json
```

## Config Shape

Top-level pipeline settings are separated from backend-specific trainer settings:

```json
{
  "training_backend": "vits",
  "reference": {},
  "generation": {},
  "speaker_ranking": {},
  "backends": {
    "vits": {}
  }
}
```

The pipeline still accepts the legacy top-level `vits` block, but new configs should use `backends.vits`.

## Backends

By default the repository trains the vendored local full VITS implementation through the `vits` backend.
`backends.vits.training_command` is optional and only needed if you want to replace that final trainer with a different command.

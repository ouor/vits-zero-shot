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
- The default backend is the vendored local VITS implementation.
- The current VITS path uses the multi-speaker model structure while assigning a single selected speaker id in the exported corpus.

## Setup

```bash
uv sync
```

The project targets Python 3.11 because the current `faster-qwen3-tts` dependency chain pulls `onnxruntime`, which does not provide Python 3.10 wheels for this environment.

## Usage

The main entry point is:

```bash
uv run voice-trainer-run --config configs/korean_cleaners.json
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

`generation.prompt_languages` controls which corpora are sampled for candidate prompts.
The `cjke_cleaners2` configs mix Korean, English, Japanese, and Chinese prompts, while the
`korean_cleaners` configs explicitly stay Korean-only.

## Backends

By default the repository trains the vendored local VITS implementation through the `vits` backend.
`backends.vits.training_command` is optional and only needed if you want to replace that final trainer with a different command.

### VITS Override Shape

The VITS backend still accepts the existing shortcut keys such as `batch_size`, `epochs`,
`pretrained_generator`, and `pretrained_discriminator`. If you want to control fields that
normally appear in the generated `training/config.json`, add nested `train`, `data`, or `model`
override blocks under `backends.vits`.

```json
{
  "backends": {
    "vits": {
      "target_sample_rate": 22050,
      "train_split_ratio": 0.8,
      "batch_size": 2,
      "epochs": 50,
      "pretrained_generator": "sample/pretrained/cjke_cleaners2/G_0.pth",
      "train": {
        "log_interval": 50,
        "eval_interval": 200,
        "num_gpus": 1,
        "fp16_run": true
      },
      "data": {
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024
      },
      "model": {
        "gin_channels": 256
      }
    }
  }
}
```

Pipeline-generated file paths such as `training_files` and `validation_files` are still managed
by the backend and are not expected to be overridden from the top-level pipeline config.

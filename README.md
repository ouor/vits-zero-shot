# Voice Trainer Pipeline

A reproducible single-sample voice cloning and fine-tuning pipeline. Given one reference audio clip and its transcript, it synthesizes a large set of voice-cloned candidate utterances, filters them by speaker similarity, and uses the best ones to fine-tune a local VITS TTS model.

## How it works

```
Reference audio + transcript
         │
         ▼  1. Prompt generation   — sample Korean / multilingual sentences
         ▼  2. Candidate generation — voice-clone each sentence (FasterQwen3TTS)
         ▼  3. Speaker ranking      — ECAPA-TDNN cosine similarity → keep top-N
         ▼  4. Data preparation     — export to VITS dataset format
         ▼  5. Training             — fine-tune vendored VITS model
         ▼  6. Summary              — events.jsonl + result JSON
```

## Setup

```bash
uv sync
```

> Requires Python 3.11–3.12. The `faster-qwen3-tts` dependency chain pulls `onnxruntime`, which does not provide Python 3.10 wheels for this environment.

To use the Gradio demo, also install the optional dependency:

```bash
uv pip install ".[demo]"
```

## Training pipeline

```bash
uv run voice-trainer-run --config configs/korean_cleaners.json
```

## Gradio demo

Run a web UI against any trained VITS generator checkpoint:

```bash
uv run demo-vits-run \
  --model sample/pretrained/korean_cleaners/G_0.pth \
  --config sample/pretrained/korean_cleaners/config.json
```

| Option | Description |
|--------|-------------|
| `--model PATH` | Generator checkpoint (`G_*.pth`) |
| `--config PATH` | Matching `config.json` |
| `--device DEVICE` | `cuda` / `cpu` — auto-detected if omitted |
| `--port PORT` | Local port (default: 7860) |
| `--share` | Create a public Gradio share link |

For multi-speaker models the UI shows a speaker dropdown derived from the `speakers` list in `config.json`.

## Python inference API

```python
from voice_trainer.vits.inference import VitsInference
import soundfile as sf

tts = VitsInference(
    generator_path="sample/pretrained/korean_cleaners/G_0.pth",
    config_path="sample/pretrained/korean_cleaners/config.json",
)
audio = tts.synthesize("안녕하세요", speaker_id=0)
sf.write("output.wav", audio, tts.sampling_rate)
```

`synthesize()` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `text` | — | Raw input text; cleaners applied internally |
| `speaker_id` | `0` | Speaker index for multi-speaker models |
| `noise_scale` | `0.667` | Latent flow variation |
| `noise_scale_w` | `0.8` | Duration predictor variation |
| `length_scale` | `1.0` | Phoneme duration multiplier (>1 = slower) |

The cleaner and symbol set are loaded automatically from `config.json`.

## Config shape

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
The `cjke_cleaners2` configs mix Korean, English, Japanese, and Chinese prompts; the
`korean_cleaners` configs stay Korean-only.

## Backends

By default the repository trains the vendored local VITS implementation through the `vits` backend.
`backends.vits.training_command` is optional and only needed to replace the final trainer with a custom command.

### VITS override shape

The VITS backend accepts shortcut keys (`batch_size`, `epochs`, `pretrained_generator`,
`pretrained_discriminator`) as well as nested `train`, `data`, and `model` blocks for
fine-grained control over the generated `training/config.json`.

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

Pipeline-generated paths (`training_files`, `validation_files`) are managed by the backend and should not be overridden from the top-level config.

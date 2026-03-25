from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="demo-vits-run",
        description="Launch a Gradio TTS demo using a trained VITS generator.",
    )
    parser.add_argument("--model", required=True, metavar="PATH", help="Path to the generator checkpoint (G_*.pth)")
    parser.add_argument("--config", required=True, metavar="PATH", help="Path to the config.json for the checkpoint")
    parser.add_argument("--device", default=None, metavar="DEVICE", help="Torch device (e.g. 'cuda', 'cpu'). Auto-detected if omitted.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7860, metavar="PORT", help="Local port to serve on (default: 7860)")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    if not model_path.is_file():
        sys.exit(f"[demo-vits-run] Model file not found: {model_path}")
    if not config_path.is_file():
        sys.exit(f"[demo-vits-run] Config file not found: {config_path}")

    try:
        import gradio as gr
    except ImportError:
        sys.exit(
            "[demo-vits-run] 'gradio' is not installed. Run: pip install gradio"
        )

    print(f"[demo-vits-run] Loading model from {model_path} ...")
    from voice_trainer.vits.inference import VitsInference

    tts = VitsInference(
        generator_path=model_path,
        config_path=config_path,
        device=args.device,
    )

    n_speakers = tts.net_g.n_speakers
    speakers = getattr(tts.hps, "speakers", None)

    # Build speaker choices: use named list when available, otherwise ints.
    if speakers and len(speakers) > 0:
        speaker_choices = list(speakers)
    elif n_speakers > 1:
        speaker_choices = [str(i) for i in range(n_speakers)]
    else:
        speaker_choices = None

    def _synthesize(text: str, speaker: str, noise_scale: float, noise_scale_w: float, length_scale: float):
        if not text.strip():
            return None

        sid = 0
        if speaker_choices is not None:
            if speaker in speaker_choices:
                sid = speaker_choices.index(speaker)
            else:
                try:
                    sid = int(speaker)
                except ValueError:
                    sid = 0

        audio = tts.synthesize(
            text,
            speaker_id=sid,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )
        return tts.sampling_rate, audio

    with gr.Blocks(title="VITS TTS Demo") as demo:
        gr.Markdown("## VITS TTS Demo")
        gr.Markdown(f"**Model:** `{model_path.name}`  |  **Config:** `{config_path.name}`")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Input text",
                    placeholder="Enter text to synthesise...",
                    lines=4,
                )

                if speaker_choices is not None:
                    speaker_input = gr.Dropdown(
                        choices=speaker_choices,
                        value=speaker_choices[0],
                        label="Speaker",
                    )
                else:
                    speaker_input = gr.Textbox(value="0", visible=False)

                with gr.Accordion("Advanced", open=False):
                    noise_scale = gr.Slider(0.0, 2.0, value=0.667, step=0.01, label="Noise scale (variation)")
                    noise_scale_w = gr.Slider(0.0, 2.0, value=0.8, step=0.01, label="Noise scale W (duration)")
                    length_scale = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Length scale (speed)")

                run_btn = gr.Button("Synthesise", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(label="Output audio", type="numpy")

        run_btn.click(
            fn=_synthesize,
            inputs=[text_input, speaker_input, noise_scale, noise_scale_w, length_scale],
            outputs=audio_output,
        )
        text_input.submit(
            fn=_synthesize,
            inputs=[text_input, speaker_input, noise_scale, noise_scale_w, length_scale],
            outputs=audio_output,
        )

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

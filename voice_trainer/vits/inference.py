from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .models import SynthesizerTrn
from .utils import get_hparams_from_file, load_model_checkpoint
from .runtime import ensure_monotonic_align_built
from . import text as text_module


# Symbol sets per cleaner name, matching the commented blocks in text/symbols.py.
# Used as fallback when config.json does not include an explicit "symbols" list.
_CLEANER_SYMBOLS: dict[str, list[str]] = {
    "japanese_cleaners": (
        ["_"] + list(",.!?-") + list("AEINOQUabdefghijkmnoprstuvwyzʃʧ↓↑ ")
    ),
    "japanese_cleaners2": (
        ["_"] + list(",.!?-~…") + list("AEINOQUabdefghijkmnoprstuvwyzʃʧʦ↓↑ ")
    ),
    "korean_cleaners": (
        ["_"] + list(",.!?…~") + list("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅓㅗㅜㅡㅣㅐㅔ ")
    ),
    "chinese_cleaners": (
        ["_"] + list("，。！？—…") + list("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩˉˊˇˋ˙ ")
    ),
    "zh_ja_mixture_cleaners": (
        ["_"] + list(",.!?-~…") + list("AEINOQUabdefghijklmnoprstuvwyzʃʧʦɯɹəɥ⁼ʰ`→↓↑ ")
    ),
    "cjks_cleaners": (
        ["_"] + list(",.!?-~…") + list("NQabdefghijklmnopstuvwxyzʃʧʥʦɯɹəɥçɸɾβŋɦː⁼ʰ`^#*=→↓↑ ")
    ),
    "cjke_cleaners": (
        ["_"] + list(",.!?-~…") + list("NQabdefghijklmnopstuvwxyzɑæʃʑçɯɪɔɛɹðəɫɥɸʊɾʒθβŋɦ⁼ʰ`^#*=→↓↑ ")
    ),
    "cjke_cleaners2": (
        ["_"] + list(",.!?-~…") + list("NQabdefghijklmnopstuvwxyzɑæʃʑçɯɪɔɛɹðəɫɥɸʊɾʒθβŋɦ⁼ʰ`^#*=ˈˌ→↓↑ ")
    ),
    "thai_cleaners": (
        ["_"] + list(".!? ") + list("กขฃคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์")
    ),
    "shanghainese_cleaners": (
        ["_"] + list(",.!?…") + list("abdfghiklmnopstuvyzøŋȵɑɔɕəɤɦɪɿʑʔʰ̩̃ᴀᴇ15678 ")
    ),
}


class VitsInference:
    """Inference wrapper for a trained VITS generator.

    Parameters
    ----------
    generator_path:
        Path to the generator checkpoint (``G_*.pth``).
    config_path:
        Path to the matching ``config.json``.
    device:
        Torch device string, ``torch.device``, or ``None`` (auto-selects CUDA
        when available, otherwise CPU).
    """

    def __init__(
        self,
        generator_path: Union[str, Path],
        config_path: Union[str, Path],
        device: Union[str, torch.device, None] = None,
    ) -> None:
        ensure_monotonic_align_built()

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.hps = get_hparams_from_file(str(config_path))

        symbols = self._resolve_symbols()
        text_module.set_symbols(symbols)

        self.cleaner_names: list[str] = list(self.hps.data.text_cleaners)
        self.sampling_rate: int = self.hps.data.sampling_rate
        self.add_blank: bool = bool(getattr(self.hps.data, "add_blank", False))

        n_speakers = getattr(self.hps.data, "n_speakers", 0)

        self.net_g = SynthesizerTrn(
            n_vocab=len(symbols),
            spec_channels=self.hps.data.filter_length // 2 + 1,
            segment_size=self.hps.train.segment_size // self.hps.data.hop_length,
            inter_channels=self.hps.model.inter_channels,
            hidden_channels=self.hps.model.hidden_channels,
            filter_channels=self.hps.model.filter_channels,
            n_heads=self.hps.model.n_heads,
            n_layers=self.hps.model.n_layers,
            kernel_size=self.hps.model.kernel_size,
            p_dropout=self.hps.model.p_dropout,
            resblock=self.hps.model.resblock,
            resblock_kernel_sizes=self.hps.model.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hps.model.resblock_dilation_sizes,
            upsample_rates=self.hps.model.upsample_rates,
            upsample_initial_channel=self.hps.model.upsample_initial_channel,
            upsample_kernel_sizes=self.hps.model.upsample_kernel_sizes,
            n_speakers=n_speakers,
            gin_channels=self.hps.model.gin_channels,
        ).to(self.device)

        load_model_checkpoint(str(generator_path), self.net_g)
        self.net_g.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_symbols(self) -> list[str]:
        """Return the symbol list to use for this model.

        Priority:
        1. Explicit ``symbols`` key in ``config.json`` (most reliable)
        2. Lookup in ``_CLEANER_SYMBOLS`` by the first recognised cleaner name
        3. Module default (``text/symbols.py``)
        """
        if hasattr(self.hps, "symbols") and self.hps.symbols:
            return list(self.hps.symbols)

        cleaners: list[str] = getattr(self.hps.data, "text_cleaners", [])
        for name in cleaners:
            if name in _CLEANER_SYMBOLS:
                return list(_CLEANER_SYMBOLS[name])

        from .text.symbols import symbols as _default
        return list(_default)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
    ) -> np.ndarray:
        """Convert *text* to a waveform.

        Parameters
        ----------
        text:
            Input text (raw; cleaners are applied internally).
        speaker_id:
            Speaker embedding index for multi-speaker models. Ignored for
            single-speaker models.
        noise_scale:
            Controls variation of the latent flow (default 0.667).
        noise_scale_w:
            Controls variation of the duration predictor (default 0.8).
        length_scale:
            Multiplier for predicted phoneme durations (>1 = slower).

        Returns
        -------
        np.ndarray
            Float32 audio samples, shape ``(T,)``, at ``self.sampling_rate``.
        """
        sequence = text_module.text_to_sequence(text, self.cleaner_names)
        if self.add_blank:
            from .commons import intersperse
            sequence = intersperse(sequence, 0)

        x = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([x.size(1)]).to(self.device)

        sid: torch.Tensor | None = None
        if self.net_g.n_speakers > 0:
            sid = torch.LongTensor([speaker_id]).to(self.device)

        audio, *_ = self.net_g.infer(
            x,
            x_lengths,
            sid=sid,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
        )
        return audio[0, 0].cpu().numpy()

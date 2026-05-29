from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch import Tensor
from torchaudio.models import Wav2Vec2Model

from .hifigan.models import Generator as HiFiGAN
from .hifigan.utils import AttrDict

SPEAKER_INFORMATION_LAYER = 6


def fast_cosine_dist(
    source_feats: Tensor,
    matching_pool: Tensor,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Compute cosine distance between two feature matrices."""

    source = F.normalize(source_feats.to(device).float(), p=2, dim=-1, eps=1e-12)
    pool = F.normalize(matching_pool.to(device).float(), p=2, dim=-1, eps=1e-12)
    return 1.0 - source @ pool.T


class KNeighborsVC(nn.Module):
    """kNN-VC matcher and vocoder wrapper."""

    def __init__(
        self,
        wavlm: Wav2Vec2Model,
        hifigan: HiFiGAN,
        hifigan_cfg: AttrDict,
        device="cuda",
    ) -> None:
        """kNN-VC matcher.
        Arguments:
            - `wavlm` : trained WavLM model
            - `hifigan`: trained hifigan model
            - `hifigan_cfg`: hifigan config to use for vocoding.
        """
        super().__init__()
        # load hifigan
        self.hifigan = hifigan.eval()
        self.h = hifigan_cfg
        # store wavlm
        self.wavlm = wavlm.eval()
        self.device = torch.device(device)
        self.sr = self.h.sampling_rate
        self.hop_length = 320

    def get_matching_set(
        self,
        wavs: Sequence[str | Path | Tensor],
        layer: int = SPEAKER_INFORMATION_LAYER,
        vad_trigger_level: float = 7,
    ) -> Tensor:
        """Return concatenated WavLM features for the reference utterances."""

        feats = [
            self.get_features(
                wav,
                layer=layer,
                vad_trigger_level=vad_trigger_level,
            )
            for wav in wavs
        ]

        return torch.cat(feats, dim=0).cpu()

    @torch.inference_mode()
    def vocode(self, c: Tensor) -> Tensor:
        """Vocode features with hifigan. `c` is of shape (bs, seq_len, c_dim)"""
        y_g_hat = self.hifigan(c)
        y_g_hat = y_g_hat.squeeze(1)
        return y_g_hat

    @torch.inference_mode()
    def get_features(
        self, path, layer: int = SPEAKER_INFORMATION_LAYER, vad_trigger_level=0
    ):
        """Returns features of `path` waveform as a tensor of shape (seq_len, dim), optionally perform VAD trimming
        on start/end with `vad_trigger_level`.
        """
        # load audio
        if type(path) in [str, Path]:
            x, sr = torchaudio.load(path, normalize=True)
        else:
            x: Tensor = path
            sr = self.sr
            if x.dim() == 1:
                x = x[None]

        if not sr == self.sr:
            print(f"resample {sr} to {self.sr} in {path}")
            x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=self.sr)
            sr = self.sr

        # trim silence from front and back
        if vad_trigger_level > 1e-3:
            transform = T.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
            x_front_trim = transform(x)
            # original way, disabled because it lacks windows support
            # waveform_reversed, sr = apply_effects_tensor(x_front_trim, sr, [["reverse"]])
            waveform_reversed = torch.flip(x_front_trim, (-1,))
            waveform_reversed_front_trim = transform(waveform_reversed)
            waveform_end_trim = torch.flip(waveform_reversed_front_trim, (-1,))
            # waveform_end_trim, sr = apply_effects_tensor(
            #    waveform_reversed_front_trim, sr, [["reverse"]]
            # )
            x = waveform_end_trim

        # extract the representation of each layer
        wav_input_16khz = x.to(self.device)

        if layer == 0:
            c, _ = self.wavlm.feature_extractor(wav_input_16khz, None)
            projected = self.wavlm.encoder.feature_projection(c)

            if isinstance(projected, tuple):
                projected = projected[0]

            return projected.squeeze(0)

        features_list, _ = self.wavlm.extract_features(
            wav_input_16khz, num_layers=layer
        )
        features = features_list[-1].squeeze(0)

        return features

    @torch.inference_mode()
    def match(
        self,
        query_seq: Tensor,
        matching_set: Tensor,
        synth_set: Tensor = None,
        topk: int = 4,
        tgt_loudness_db: float | None = -16,
        target_duration: float | None = None,
        device: str | None = None,
    ) -> Tensor:
        """Perform kNN feature matching and vocode the converted waveform."""

        if topk < 1:
            raise ValueError(f"topk must be at least 1, got {topk}")

        resolved_device = torch.device(device) if device is not None else self.device

        if matching_set.shape[0] == 0:
            raise ValueError("matching_set must contain at least one frame")

        matching_set = matching_set.to(resolved_device)
        query_seq = query_seq.to(resolved_device)

        if synth_set is None:
            synth_set = matching_set
        else:
            synth_set = synth_set.to(resolved_device)

        if target_duration is not None:
            target_samples = int(target_duration * self.sr)
            scale_factor = (target_samples / self.hop_length) / query_seq.shape[0]
            query_seq = F.interpolate(
                query_seq.T[None],
                scale_factor=scale_factor,
                mode="linear",
                align_corners=False,
            )[0].T

        dists = fast_cosine_dist(query_seq, matching_set, device=resolved_device)
        k = min(topk, matching_set.shape[0])
        best = dists.topk(k=k, largest=False, dim=-1)
        out_feats = synth_set[best.indices].mean(dim=1)

        prediction = self.vocode(out_feats[None].to(resolved_device)).cpu().squeeze()

        if tgt_loudness_db is None:
            return prediction

        src_loudness = torchaudio.functional.loudness(
            prediction[None],
            self.h.sampling_rate,
        )
        gain_db = float(tgt_loudness_db - src_loudness.item())

        return torchaudio.functional.gain(prediction, gain_db)

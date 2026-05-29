import logging
import math
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch import Tensor
from torchaudio.models import Wav2Vec2Model

from .devices import module_device, resolve_device
from .hifigan.models import Generator as HiFiGAN
from .hifigan.utils import AttrDict
from .wavlm import SPEAKER_INFORMATION_LAYER, extract_wavlm_layers

LOGGER = logging.getLogger(__name__)

DEFAULT_FEATURE_LOUDNESS_CEILING_DB: float | None = None


def _validate_feature_matrix(
    name: str,
    features: Tensor,
    *,
    allow_empty: bool = True,
) -> None:
    if features.dim() != 2:
        raise ValueError(
            f"{name} must have shape (frames, dim), got {tuple(features.shape)}"
        )

    if not allow_empty and features.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")


def _validate_same_feature_dim(
    left_name: str,
    left: Tensor,
    right_name: str,
    right: Tensor,
) -> None:
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"{left_name} and {right_name} feature dimensions must match, got "
            f"{left.shape[1]} and {right.shape[1]}"
        )


def fast_cosine_dist(
    source_feats: Tensor,
    matching_pool: Tensor,
    device: str | torch.device | None = None,
) -> Tensor:
    """Compute cosine distance between two feature matrices."""

    _validate_feature_matrix("source_feats", source_feats)
    _validate_feature_matrix("matching_pool", matching_pool)
    _validate_same_feature_dim(
        "source_feats", source_feats, "matching_pool", matching_pool
    )

    resolved_device = source_feats.device if device is None else resolve_device(device)
    source = F.normalize(
        source_feats.to(resolved_device).float(), p=2, dim=-1, eps=1e-12
    )
    pool = F.normalize(
        matching_pool.to(resolved_device).float(), p=2, dim=-1, eps=1e-12
    )
    return 1.0 - source @ pool.T


def attenuate_loud_waveform(
    waveform: Tensor,
    sample_rate: int,
    loudness_ceiling_db: float | None = DEFAULT_FEATURE_LOUDNESS_CEILING_DB,
) -> Tensor:
    """Optionally attenuate audio that is louder than the feature-extraction ceiling."""

    if loudness_ceiling_db is None:
        return waveform

    loudness = torchaudio.functional.loudness(waveform, sample_rate)
    loudness_db = float(loudness.item())

    if not math.isfinite(loudness_db) or loudness_db <= loudness_ceiling_db:
        return waveform

    return torchaudio.functional.gain(waveform, loudness_ceiling_db - loudness_db)


class KNeighborsVC(nn.Module):
    """kNN-VC matcher and vocoder wrapper."""

    def __init__(
        self,
        wavlm: Wav2Vec2Model,
        hifigan: HiFiGAN,
        hifigan_cfg: AttrDict,
        device: str | torch.device | None = None,
    ) -> None:
        """kNN-VC matcher.
        Arguments:
            - `wavlm` : trained WavLM model
            - `hifigan`: trained hifigan model
            - `hifigan_cfg`: hifigan config to use for vocoding.
        """
        super().__init__()
        resolved_device = resolve_device(device)
        # load hifigan
        self.hifigan = hifigan.to(resolved_device).eval()
        self.h = hifigan_cfg
        # store wavlm
        self.wavlm = wavlm.to(resolved_device).eval()
        self.sr = int(self.h.sampling_rate)
        self.hop_length = int(self.h.hop_size)

    @property
    def device(self) -> torch.device:
        return module_device(self)

    def get_matching_set(
        self,
        wavs: Sequence[str | Path | Tensor],
        layer: int = SPEAKER_INFORMATION_LAYER,
        vad_trigger_level: float = 7.0,
        feature_loudness_ceiling_db: float | None = DEFAULT_FEATURE_LOUDNESS_CEILING_DB,
    ) -> Tensor:
        """Return concatenated WavLM features for the reference utterances."""

        if not wavs:
            raise ValueError("wavs must contain at least one reference utterance")

        feats = [
            self.get_features(
                wav,
                layer=layer,
                vad_trigger_level=vad_trigger_level,
                feature_loudness_ceiling_db=feature_loudness_ceiling_db,
            ).cpu()
            for wav in wavs
        ]

        return torch.cat(feats, dim=0)

    @torch.inference_mode()
    def vocode(self, c: Tensor) -> Tensor:
        """Vocode features with hifigan. `c` is of shape (bs, seq_len, c_dim)"""
        c = c.to(self.device)

        if c.dim() != 3:
            raise ValueError(
                f"c must have shape (batch, frames, dim), got {tuple(c.shape)}"
            )

        y_g_hat = self.hifigan(c)
        y_g_hat = y_g_hat.squeeze(1)
        return y_g_hat

    @torch.inference_mode()
    def get_features(
        self,
        path: str | Path | Tensor,
        *,
        sample_rate: int | None = None,
        layer: int = SPEAKER_INFORMATION_LAYER,
        vad_trigger_level: float = 0.0,
        feature_loudness_ceiling_db: float | None = DEFAULT_FEATURE_LOUDNESS_CEILING_DB,
    ) -> Tensor:
        """Returns features of `path` waveform as a tensor of shape (seq_len, dim), optionally perform VAD trimming
        on start/end with `vad_trigger_level`.
        """
        # load audio
        if isinstance(path, (str, Path)):
            x, sr = torchaudio.load(path, normalize=True)
        else:
            x = path.detach()
            sr = self.sr if sample_rate is None else int(sample_rate)
            if x.dim() == 1:
                x = x[None]

        x = x.cpu()

        if x.dim() != 2:
            raise ValueError(
                f"waveform tensor must have shape (samples,) or (channels, samples), got {tuple(x.shape)}"
            )

        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)

        if not sr == self.sr:
            LOGGER.info("Resampling audio from %s Hz to %s Hz.", sr, self.sr)
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

        x = attenuate_loud_waveform(x, sr, feature_loudness_ceiling_db)

        # extract the representation of each layer
        wav_input_16khz = x.to(self.device)
        features = extract_wavlm_layers(self.wavlm, wav_input_16khz, {layer})

        return features[layer]

    @torch.inference_mode()
    def match(
        self,
        query_seq: Tensor,
        matching_set: Tensor,
        synth_set: Tensor | None = None,
        topk: int = 4,
        tgt_loudness_db: float | None = None,
        target_duration: float | None = None,
        device: str | torch.device | None = None,
    ) -> Tensor:
        """
        Perform kNN feature matching and vocode the converted waveform.

        Loudness normalization is opt-in. Passing `tgt_loudness_db` applies a
        final gain stage, but no limiter, so callers should choose that target
        deliberately.
        """

        if topk < 1:
            raise ValueError(f"topk must be at least 1, got {topk}")

        _validate_feature_matrix("query_seq", query_seq, allow_empty=False)
        _validate_feature_matrix("matching_set", matching_set, allow_empty=False)
        _validate_same_feature_dim("query_seq", query_seq, "matching_set", matching_set)

        if synth_set is None:
            synth_set = matching_set
        else:
            _validate_feature_matrix("synth_set", synth_set, allow_empty=False)

        if synth_set.shape[0] != matching_set.shape[0]:
            raise ValueError(
                "synth_set and matching_set must contain the same number of frames, "
                f"got {synth_set.shape[0]} and {matching_set.shape[0]}"
            )

        expected_synth_dim = int(self.h.hubert_dim)
        if synth_set.shape[1] != expected_synth_dim:
            raise ValueError(
                f"synth_set feature dimension must match the vocoder input dimension "
                f"{expected_synth_dim}, got {synth_set.shape[1]}"
            )

        if target_duration is not None and target_duration <= 0:
            raise ValueError(f"target_duration must be positive, got {target_duration}")

        resolved_device = resolve_device(device) if device is not None else self.device

        if matching_set.shape[0] == 0:
            raise ValueError("matching_set must contain at least one frame")

        matching_set = matching_set.to(resolved_device)
        query_seq = query_seq.to(resolved_device)
        synth_set = synth_set.to(resolved_device)

        target_samples = None
        if target_duration is not None:
            target_samples = max(1, int(round(target_duration * self.sr)))
            target_frames = max(1, round(target_samples / self.hop_length))
            query_seq = (
                F.interpolate(
                    query_seq.T.unsqueeze(0),
                    size=target_frames,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .T
            )

        dists = fast_cosine_dist(query_seq, matching_set, device=resolved_device)
        k = min(topk, matching_set.shape[0])
        best = dists.topk(k=k, largest=False, dim=-1)
        out_feats = synth_set[best.indices].mean(dim=1)

        prediction = self.vocode(out_feats[None].to(resolved_device)).cpu().squeeze(0)

        if target_samples is not None:
            if prediction.numel() > target_samples:
                prediction = prediction[:target_samples]
            elif prediction.numel() < target_samples:
                prediction = F.pad(prediction, (0, target_samples - prediction.numel()))

        if tgt_loudness_db is None:
            return prediction

        src_loudness = torchaudio.functional.loudness(
            prediction[None],
            self.h.sampling_rate,
        )
        src_loudness_db = float(src_loudness.item())

        if not math.isfinite(src_loudness_db):
            return prediction

        gain_db = float(tgt_loudness_db - src_loudness_db)

        return torchaudio.functional.gain(prediction, gain_db)

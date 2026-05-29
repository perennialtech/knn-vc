"""
Extract and optionally pre-match WavLM audio features for a dataset of utterances.

The script expects files whose speaker id can be read from the filename prefix before
the first dash, for example LibriSpeech-style files like:

    1089-134686-0000.flac

Pre-matching is a kNN regression performed within each speaker. For every utterance,
each frame searches for nearest frames from all other utterances by the same speaker.
The nearest neighbors are selected using the matching layer, then averaged using the
synthesis layer. This lets you match in one WavLM layer while saving features from
another.

Layer numbering:

    1-24   one-based WavLM transformer outputs, matching the original kNN-VC/WavLM convention

If torchcodec or ffmpeg is not available, this version uses torchaudio instead.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torchaudio.pipelines import WAVLM_LARGE
from tqdm import tqdm

from knn_vc.devices import resolve_device
from knn_vc.wavlm import extract_wavlm_layers, validate_wavlm_layer

TARGET_SAMPLE_RATE = 16_000
DOWNSAMPLE_FACTOR = 320

MIN_VECTORS_FOR_PREMATCH = 9_000  # roughly 3 minutes
MAX_VECTORS_FOR_PREMATCH = 24_000  # roughly 8 minutes

LOGGER = logging.getLogger("prematch_dataset")


def configure_logging() -> None:
    """Configure file logging once."""

    if LOGGER.handlers:
        return

    handler = logging.FileHandler("prematch_dataset.log")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


@dataclass(frozen=True)
class WorkItem:
    """One source utterance and its optional output path."""

    source: Path
    target: Path | None


@dataclass(frozen=True)
class FeatureExtractor:
    """Thin WavLM wrapper for loading audio and returning selected layers."""

    wavlm: nn.Module
    device: torch.device
    pad_to_multiple: bool

    @torch.inference_mode()
    def __call__(self, path: Path, layers: Iterable[int]) -> dict[int, Tensor]:
        layer_set = set(layers)

        for layer in layer_set:
            validate_wavlm_layer(layer)

        waveform = self.load_waveform(path)
        return extract_wavlm_layers(self.wavlm, waveform, layer_set)

    def load_waveform(self, path: Path) -> Tensor:
        waveform, sample_rate = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                TARGET_SAMPLE_RATE,
            )

        if self.pad_to_multiple:
            waveform = pad_waveform_to_downsample_factor(waveform)

        return waveform.to(self.device)


@dataclass(frozen=True)
class PreMatcher:
    """kNN pre-matcher for one speaker chunk."""

    topk: int
    distance_batch_size: int

    @torch.inference_mode()
    def __call__(
        self,
        match_feats: Sequence[Tensor],
        synth_feats: Sequence[Tensor],
        targets: Sequence[Path | None],
    ) -> list[Tensor | None]:
        if not match_feats:
            return []

        lengths = [int(feat.shape[0]) for feat in match_feats]
        frame_ranges = make_frame_ranges(lengths)

        match_raw = torch.cat(tuple(match_feats), dim=0)
        match_all = F.normalize(match_raw.float(), p=2, dim=-1, eps=1e-12)

        if match_feats is synth_feats:
            synth_all = match_raw
        else:
            synth_all = torch.cat(tuple(synth_feats), dim=0)

        matched: list[Tensor | None] = [None] * len(match_feats)

        for utt_idx, (start, end) in enumerate(frame_ranges):
            target_path = targets[utt_idx]

            if target_path is None:
                continue

            source_match = match_all[start:end]
            source_synth = synth_all[start:end]
            target_count = match_all.shape[0] - source_match.shape[0]

            if source_match.shape[0] == 0:
                matched[utt_idx] = source_synth
                continue

            if target_count < MIN_VECTORS_FOR_PREMATCH:
                LOGGER.warning("Not enough target vectors for %s", target_path)
                matched[utt_idx] = source_synth
                continue

            target_match = without_range(match_all, start, end)
            target_synth = without_range(synth_all, start, end)
            k = min(self.topk, target_match.shape[0])

            batches: list[Tensor] = []

            for source_batch in batches_of_rows(source_match, self.distance_batch_size):
                distances = 1.0 - source_batch @ target_match.T
                nearest = distances.topk(k=k, dim=-1, largest=False).indices
                batches.append(target_synth[nearest].mean(dim=1))

            matched[utt_idx] = torch.cat(batches, dim=0)

        return matched


def make_df(root_path: Path, ext: str = ".flac") -> pd.DataFrame:
    """Build a dataframe with file paths and speaker ids."""

    if not root_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root_path}")

    normalized_ext = ext if ext.startswith(".") else f".{ext}"

    LOGGER.info("Loading %s files from %s", normalized_ext, root_path)

    files = sorted(root_path.rglob(f"*{normalized_ext}"))
    speakers = [path.stem.split("-")[0] for path in files]

    LOGGER.info("Loaded %s files", len(files))

    return pd.DataFrame({"path": files, "speaker": speakers})


def validate_positive(name: str, value: int) -> None:
    """Validate a positive integer CLI argument."""

    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")


def pad_waveform_to_downsample_factor(waveform: Tensor) -> Tensor:
    """Pad the waveform only when it is not already divisible by the stride."""

    n_pad = (-waveform.shape[-1]) % DOWNSAMPLE_FACTOR

    if n_pad == 0:
        return waveform

    return F.pad(waveform, (0, n_pad), value=0.0)


def save_tensor(path: Path, tensor: Tensor) -> None:
    """Save a tensor as CPU float16, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu().half(), path)


def make_frame_ranges(lengths: Sequence[int]) -> list[tuple[int, int]]:
    """Convert per-utterance frame counts into stacked tensor slice ranges."""

    ranges: list[tuple[int, int]] = []
    start = 0

    for length in lengths:
        end = start + int(length)
        ranges.append((start, end))
        start = end

    return ranges


def without_range(tensor: Tensor, start: int, end: int) -> Tensor:
    """Return tensor rows outside [start, end)."""

    if start == 0:
        return tensor[end:]

    if end == tensor.shape[0]:
        return tensor[:start]

    return torch.cat((tensor[:start], tensor[end:]), dim=0)


def batches_of_rows(tensor: Tensor, batch_size: int) -> Iterable[Tensor]:
    """Yield row batches from a 2D tensor."""

    for start in range(0, tensor.shape[0], batch_size):
        yield tensor[start : start + batch_size]


def make_chunk_ranges(feat_lens: Sequence[int]) -> list[tuple[int, int]]:
    """
    Split a speaker into utterance-aligned chunks when there is a lot of data.

    Chunks are balanced by total frame count, but never split an utterance.
    """

    n_utts = len(feat_lens)
    total_feats = int(sum(feat_lens))

    if n_utts == 0:
        return []

    if n_utts == 1 or total_feats <= MAX_VECTORS_FOR_PREMATCH * 2:
        return [(0, n_utts)]

    n_splits = min(n_utts, max(2, math.ceil(total_feats / MAX_VECTORS_FOR_PREMATCH)))
    target_chunk_size = total_feats / n_splits

    ranges: list[tuple[int, int]] = []
    start = 0
    running_feats = 0

    for end, feat_len in enumerate(feat_lens, start=1):
        running_feats += int(feat_len)

        remaining_utts = n_utts - end
        remaining_splits = n_splits - len(ranges) - 1

        if (
            remaining_splits > 0
            and running_feats >= target_chunk_size
            and remaining_utts >= remaining_splits
        ):
            ranges.append((start, end))
            start = end
            running_feats = 0

    if start < n_utts:
        ranges.append((start, n_utts))

    return ranges


def speaker_items(
    group: pd.DataFrame,
    source_root: Path,
    out_path: Path,
    resume: bool,
) -> list[WorkItem]:
    """Build per-speaker work items, preserving resumed files as match candidates."""

    items: list[WorkItem] = []

    for row in group.itertuples(index=False):
        source = Path(getattr(row, "path"))
        rel_path = source.relative_to(source_root)
        target = (out_path / rel_path).with_suffix(".pt")

        if resume and target.is_file():
            LOGGER.warning("Features already exist for %s", rel_path)
            target = None

        items.append(WorkItem(source=source, target=target))

    return items


def save_unmatched_speaker(
    items: Sequence[WorkItem],
    extractor: FeatureExtractor,
    synthesis_layer: int,
) -> None:
    """Extract and save one synthesis layer without pre-matching."""

    for item in items:
        if item.target is None:
            continue

        features = extractor(item.source, {synthesis_layer})
        save_tensor(item.target, features[synthesis_layer])


def load_speaker_features(
    items: Sequence[WorkItem],
    extractor: FeatureExtractor,
    matching_layer: int,
    synthesis_layer: int,
) -> tuple[list[Tensor], list[Tensor], list[Path | None]]:
    """Load matching and synthesis features for one speaker."""

    match_feats: list[Tensor] = []
    synth_feats: list[Tensor] = []
    targets: list[Path | None] = []

    for item in items:
        selected = extractor(item.source, {matching_layer, synthesis_layer})

        match_tensor = selected[matching_layer].detach().cpu()

        if matching_layer == synthesis_layer:
            synth_tensor = match_tensor
        else:
            synth_tensor = selected[synthesis_layer].detach().cpu()

        if match_tensor.shape[0] != synth_tensor.shape[0]:
            raise ValueError(
                "Matching and synthesis layers must have the same number of frames "
                f"for {item.source}: {match_tensor.shape[0]} != {synth_tensor.shape[0]}"
            )

        match_feats.append(match_tensor)
        synth_feats.append(synth_tensor)
        targets.append(item.target)

    return match_feats, synth_feats, targets


def save_prematched_speaker(
    items: Sequence[WorkItem],
    extractor: FeatureExtractor,
    prematcher: PreMatcher,
    matching_layer: int,
    synthesis_layer: int,
) -> None:
    """Extract, pre-match, and save all unfinished utterances for one speaker."""

    match_cpu, synth_cpu, targets = load_speaker_features(
        items=items,
        extractor=extractor,
        matching_layer=matching_layer,
        synthesis_layer=synthesis_layer,
    )

    feat_lens = [int(feat.shape[0]) for feat in match_cpu]

    for start, end in make_chunk_ranges(feat_lens):
        chunk_match = [feat.to(extractor.device) for feat in match_cpu[start:end]]

        if matching_layer == synthesis_layer:
            chunk_synth = chunk_match
        else:
            chunk_synth = [feat.to(extractor.device) for feat in synth_cpu[start:end]]

        chunk_targets = targets[start:end]
        matched = prematcher(chunk_match, chunk_synth, chunk_targets)

        for target, features in zip(chunk_targets, matched):
            if target is not None and features is not None:
                save_tensor(target, features)


def extract(
    df: pd.DataFrame,
    extractor: FeatureExtractor,
    source_root: Path,
    out_path: Path,
    synthesis_layer: int,
    matching_layer: int,
    prematch: bool,
    resume: bool,
    topk: int,
    distance_batch_size: int,
) -> None:
    """Extract and optionally pre-match features for all speakers."""

    prematcher = PreMatcher(topk=topk, distance_batch_size=distance_batch_size)
    grouped = df.groupby("speaker", sort=True)

    for speaker, group in tqdm(grouped, total=int(df["speaker"].nunique())):
        LOGGER.info("Processing speaker %s with %s utterances", speaker, len(group))

        items = speaker_items(
            group=group,
            source_root=source_root,
            out_path=out_path,
            resume=resume,
        )

        if all(item.target is None for item in items):
            LOGGER.info("All outputs already exist for speaker %s", speaker)
            continue

        if prematch:
            save_prematched_speaker(
                items=items,
                extractor=extractor,
                prematcher=prematcher,
                matching_layer=matching_layer,
                synthesis_layer=synthesis_layer,
            )
        else:
            save_unmatched_speaker(
                items=items,
                extractor=extractor,
                synthesis_layer=synthesis_layer,
            )


def main(args: argparse.Namespace) -> None:
    """Run the feature extraction job."""

    configure_logging()

    validate_wavlm_layer(args.synthesis_layer, "synthesis_layer")
    validate_wavlm_layer(args.matching_layer, "matching_layer")
    validate_positive("topk", args.topk)
    validate_positive("distance_batch_size", args.distance_batch_size)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_root = Path(args.path)
    out_path = Path(args.out_path)
    device = torch.device(args.device)
    device = resolve_device(args.device)

    LOGGER.info("Starting run with args %s", args)

    df = make_df(source_root, ext=args.ext)

    LOGGER.info("Loading WavLM large from torchaudio")
    wavlm = WAVLM_LARGE.get_model().to(device)
    wavlm.eval()

    extractor = FeatureExtractor(
        wavlm=wavlm,
        device=device,
        pad_to_multiple=not args.no_pad,
    )

    extract(
        df=df,
        extractor=extractor,
        source_root=source_root,
        out_path=out_path,
        synthesis_layer=args.synthesis_layer,
        matching_layer=args.matching_layer,
        prematch=args.prematch,
        resume=args.resume,
        topk=args.topk,
        distance_batch_size=args.distance_batch_size,
    )

    LOGGER.info("All done!")


def main_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute optionally pre-matched WavLM features for a dataset"
    )

    parser.add_argument("path", type=str)
    parser.add_argument("out_path", type=str)

    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--ext", default=".flac", type=str)

    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--synthesis_layer", type=int, default=6)
    parser.add_argument("--matching_layer", type=int, default=6)

    parser.add_argument("--prematch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no_pad",
        action="store_true",
        help="Do not zero-pad waveforms to a multiple of the WavLM downsample factor",
    )
    parser.add_argument(
        "--distance_batch_size",
        type=int,
        default=1024,
        help="Number of source frames to match at once during pre-matching",
    )

    main(parser.parse_args(argv))


if __name__ == "__main__":
    main_cli()

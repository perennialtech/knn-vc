from __future__ import annotations

import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from torch import Tensor

# Global caches for spectrogram calculation.
# The keys include all parameters that affect the generated object.
mel_basis: Dict[Tuple[Any, ...], Tensor] = {}
hann_window: Dict[Tuple[Any, ...], Tensor] = {}


def load_wav(full_path):
    data, sampling_rate = librosa.load(str(full_path), sr=None)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


def _path_or_current(path):
    return Path(path) if path is not None else Path(".")


def _safe_torch_load(path, map_location="cpu"):
    """
    Keep compatibility with both newer PyTorch versions that support
    weights_only and older versions that do not.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _ensure_2d_waveform(wav):
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    if wav.dim() != 2:
        raise ValueError(
            f"Expected waveform with shape (batch, samples), got {tuple(wav.shape)}"
        )

    return wav


def _pad_waveform(wav, left, right, mode="reflect"):
    if left < 0 or right < 0:
        raise ValueError(
            f"Padding must be non-negative, got left={left}, right={right}"
        )

    if left == 0 and right == 0:
        return wav

    wav = _ensure_2d_waveform(wav)

    if mode == "reflect" and wav.size(1) > max(left, right):
        return F.pad(wav.unsqueeze(1), (left, right), mode="reflect").squeeze(1)

    return F.pad(wav, (left, right), mode="constant")


def _pad_audio_to_length(audio, target_length):
    audio = _ensure_2d_waveform(audio)

    if audio.size(1) >= target_length:
        return audio

    return F.pad(audio, (0, target_length - audio.size(1)), mode="constant")


def _pad_waveform_to_hop_multiple(wav, hop_size):
    if hop_size <= 0:
        raise ValueError(f"hop_size must be positive, got {hop_size}")

    wav = _ensure_2d_waveform(wav)

    remainder = wav.size(1) % hop_size
    right_pad = (hop_size - remainder) % hop_size

    return _pad_waveform(wav, 0, right_pad, mode="reflect")


def _prepare_waveform_for_stft(wav, n_fft, hop_size):
    if n_fft < hop_size:
        raise ValueError(
            f"n_fft must be >= hop_size, got n_fft={n_fft}, hop_size={hop_size}"
        )

    wav = _ensure_2d_waveform(wav)

    # Ensure even extremely short audio can produce at least one valid frame.
    if wav.size(1) < hop_size:
        wav = _pad_audio_to_length(wav, hop_size)

    # Right-pad only when needed. The old hop padding added an extra frame when
    # the length was already divisible by hop_size.
    wav = _pad_waveform_to_hop_multiple(wav, hop_size)

    total_pad = n_fft - hop_size
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad

    return _pad_waveform(wav, left_pad, right_pad, mode="reflect")


def _pad_mel_to_frames(mel, target_frames):
    if mel.size(1) >= target_frames:
        return mel

    return F.pad(mel, (0, 0, 0, target_frames - mel.size(1)), mode="constant")


class LogMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        center=False,
    ):
        super().__init__()

        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_size,
            hop_length=hop_size,
            center=center,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=num_mels,
            mel_scale="slaney",
            f_min=fmin,
            f_max=fmax,
        )

        self.n_fft = n_fft
        self.hop_size = hop_size

    def forward(self, wav):
        wav = _prepare_waveform_for_stft(wav, self.n_fft, self.hop_size)

        self.melspectrogram = self.melspectrogram.to(
            device=wav.device,
            dtype=wav.dtype,
        )

        mel = self.melspectrogram(wav)
        return torch.log(torch.clamp(mel, min=1e-5))


def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False,
):
    y = _ensure_2d_waveform(y)

    min_value = torch.min(y)
    max_value = torch.max(y)

    if min_value < -1.0:
        print(f"min value is {min_value.item()}")
    if max_value > 1.0:
        print(f"max value is {max_value.item()}")

    device_str = str(y.device)
    dtype_str = str(y.dtype)

    basis_key = (
        "mel_basis",
        device_str,
        dtype_str,
        sampling_rate,
        n_fft,
        num_mels,
        fmin,
        fmax,
    )
    window_key = (
        "hann_window",
        device_str,
        dtype_str,
        win_size,
    )

    if basis_key not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        mel_basis[basis_key] = torch.from_numpy(mel).to(
            device=y.device,
            dtype=y.dtype,
        )

    if window_key not in hann_window:
        hann_window[window_key] = torch.hann_window(
            win_size,
            device=y.device,
            dtype=y.dtype,
        )

    y = _prepare_waveform_for_stft(y, n_fft, hop_size)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[window_key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = spec.abs().clamp_min(3e-5)
    spec = torch.matmul(mel_basis[basis_key], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    train_df = pd.read_csv(a.input_training_file)
    valid_df = pd.read_csv(a.input_validation_file)
    return train_df, valid_df


class MelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files,
        segment_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
        fmax_loss=None,
        fine_tuning=False,
        audio_root_path=None,
        feat_root_path=None,
        use_alt_melcalc=False,
    ):
        if segment_size <= 0:
            raise ValueError(f"segment_size must be positive, got {segment_size}")
        if hop_size <= 0:
            raise ValueError(f"hop_size must be positive, got {hop_size}")

        if shuffle:
            self.audio_files = training_files.sample(frac=1, random_state=1234)
        else:
            self.audio_files = training_files

        self.audio_files = self.audio_files.reset_index(drop=True)

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.device = device
        self.fine_tuning = fine_tuning

        self.audio_root_path = _path_or_current(audio_root_path)
        self.feat_root_path = _path_or_current(feat_root_path)

        self.use_alt_melcalc = use_alt_melcalc

        loss_fmax = self.fmax_loss if self.fmax_loss is not None else self.fmax

        self.alt_melspec = LogMelSpectrogram(
            n_fft,
            num_mels,
            sampling_rate,
            hop_size,
            win_size,
            fmin,
            fmax,
        )
        self.alt_melspec_loss = LogMelSpectrogram(
            n_fft,
            num_mels,
            sampling_rate,
            hop_size,
            win_size,
            fmin,
            loss_fmax,
        )

        # Compatibility attributes retained from the original implementation.
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0

        # The original cache reused the previous waveform regardless of the
        # requested index, which could mismatch audio, features, and filenames.
        # This cache is path-keyed instead.
        self._audio_cache = OrderedDict()
        self.max_audio_cache_items = max(0, int(n_cache_reuse or 0))

    def _load_audio(self, row):
        audio_path = Path(row.audio_path)
        full_path = self.audio_root_path / audio_path
        cache_key = full_path.expanduser()

        if cache_key in self._audio_cache:
            audio, sampling_rate = self._audio_cache[cache_key]
            self._audio_cache.move_to_end(cache_key)
        else:
            audio, sampling_rate = load_wav(full_path)

            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"{full_path} has SR {sampling_rate}, expected {self.sampling_rate}"
                )

            if not self.fine_tuning:
                audio = normalize(audio) * 0.95

            audio = np.asarray(audio, dtype=np.float32)

            if self.max_audio_cache_items > 0:
                self._audio_cache[cache_key] = (audio, sampling_rate)
                self._audio_cache.move_to_end(cache_key)

                while len(self._audio_cache) > self.max_audio_cache_items:
                    self._audio_cache.popitem(last=False)

        self.cached_wav = audio
        return audio, sampling_rate

    def _load_features(self, row):
        feat_path = self.feat_root_path / Path(row.feat_path)
        mel = _safe_torch_load(feat_path, map_location="cpu")

        if not torch.is_tensor(mel):
            mel = torch.as_tensor(mel)

        mel = mel.float()

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        elif mel.dim() != 3:
            raise ValueError(
                f"Expected feature tensor with shape (seq_len, dim) or "
                f"(batch, seq_len, dim), got {tuple(mel.shape)} from {feat_path}"
            )

        return mel

    def _random_crop_or_pad_audio(self, audio):
        audio = _ensure_2d_waveform(audio)

        if audio.size(1) > self.segment_size:
            max_audio_start = audio.size(1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            return audio[:, audio_start : audio_start + self.segment_size]

        return _pad_audio_to_length(audio, self.segment_size)

    def _random_crop_or_pad_finetune_pair(self, mel, audio):
        frames_per_seg = math.ceil(self.segment_size / self.hop_size)

        max_start_by_mel = max(0, mel.size(1) - frames_per_seg)
        max_start_by_audio = max(
            0, (audio.size(1) - self.segment_size) // self.hop_size
        )
        max_mel_start = min(max_start_by_mel, max_start_by_audio)

        mel_start = random.randint(0, max_mel_start) if max_mel_start > 0 else 0
        mel_end = mel_start + frames_per_seg

        audio_start = mel_start * self.hop_size
        audio_end = audio_start + self.segment_size

        mel = mel[:, mel_start:mel_end, :]
        audio = audio[:, audio_start:audio_end]

        mel = _pad_mel_to_frames(mel, frames_per_seg)
        audio = _pad_audio_to_length(audio, self.segment_size)

        return mel, audio

    def __getitem__(self, index):
        row = self.audio_files.iloc[index]

        audio, _ = self._load_audio(row)
        audio = torch.from_numpy(audio).float().unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                audio = self._random_crop_or_pad_audio(audio)

            if self.use_alt_melcalc:
                mel = self.alt_melspec(audio)
            else:
                mel = mel_spectrogram(
                    audio,
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )

            # (batch, mel_dim, seq_len) -> (batch, seq_len, mel_dim)
            mel = mel.transpose(1, 2).contiguous()
        else:
            mel = self._load_features(row)

            if self.split:
                mel, audio = self._random_crop_or_pad_finetune_pair(mel, audio)

        if self.use_alt_melcalc:
            mel_loss = self.alt_melspec_loss(audio)
        else:
            mel_loss = mel_spectrogram(
                audio,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax_loss,
                center=False,
            )

        return (
            mel.squeeze(0),
            audio.squeeze(0),
            str(row.audio_path),
            mel_loss.squeeze(0),
        )

    def __len__(self):
        return len(self.audio_files)

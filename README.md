# Voice Conversion With Just Nearest Neighbors (kNN-VC)

This repository contains training and inference code for kNN-VC, an any-to-any voice conversion model from the paper "Voice Conversion With Just k-Nearest Neighbors".

This fork preserves the [original kNN-VC inference code](https://github.com/bshall/knn-vc) while [modernizing the vocoder training code](https://github.com/carlosfranzreb/train_vocoder). The repository was originally forked from kNN-VC, which in turn adapts parts of HiFi-GAN. The training code has been updated for newer package versions, including newer PyTorch versions, and now includes a WebDataset-based training path that can reduce vocoder training time substantially.

Currently, the vocoder training procedure still follows the kNN-VC HiFi-GAN recipe. Over time, this repository may expand to support additional vocoder training procedures.

Links:

- Arxiv paper: https://arxiv.org/abs/2305.18975
- Colab quickstart: <a target="_blank" href="https://colab.research.google.com/github/bshall/knn-vc/blob/master/knnvc_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- Interspeech proceedings: https://www.isca-speech.org/archive/interspeech_2023/baas23_interspeech.html
- Demo page with samples: https://bshall.github.io/knn-vc/

![kNN-VC method](./knn-vc.png)

Figure: kNN-VC setup. The source and reference utterance(s) are encoded into self-supervised features using WavLM. Each source feature is assigned to the mean of the k closest features from the reference. The resulting feature sequence is then vocoded with HiFi-GAN to arrive at the converted waveform output.

Authors:

- [Matthew Baas](https://rf5.github.io/)\*
- [Benjamin van Niekerk](https://scholar.google.com/citations?user=zCokvy8AAAAJ&hl=en&oi=ao)\*
- [Herman Kamper](https://www.kamperh.com/)

\*Equal contribution

## Setup

This repository uses `uv` for local workflows. Install `uv` from the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/), then sync the environment from `pyproject.toml` and `uv.lock`.

Choose exactly one PyTorch backend extra.

For CUDA:

```bash
uv sync --extra cuda
```

For CPU-only:

```bash
uv sync --extra cpu
```

Run repository commands through `uv run`:

```bash
uv run knn-vc-train-hifigan knn_vc/hifigan/config.yaml
```

For development hooks:

```bash
uv sync --extra cuda --group dev
uv run pre-commit install
```

The project metadata, Python requirement, runtime dependencies, backend extras, and development dependency group are maintained in `pyproject.toml`.

## Quickstart

Load the WavLM encoder and HiFi-GAN vocoder from the installed package:

```python
from knn_vc import load_knn_vc

knn_vc = load_knn_vc(
    prematched=True,
    pretrained=True,
    device="cuda",
)
```

Use `device="cpu"` with the CPU-only backend.

To use the vocoder trained without prematched data, set `prematched=False`.

Compute features for input and reference audio:

```python
src_wav_path = "<path to arbitrary 16kHz waveform>.wav"
ref_wav_paths = [
    "<path to arbitrary 16kHz waveform from target speaker>.wav",
    "<path to 2nd utterance from target speaker>.wav",
]

query_seq = knn_vc.get_features(src_wav_path)
matching_set = knn_vc.get_matching_set(ref_wav_paths)
```

By default, feature extraction feeds waveform levels to WavLM unchanged, matching the original kNN-VC inference path. To deliberately attenuate very hot input audio before WavLM, pass a ceiling such as `feature_loudness_ceiling_db=-24` to `get_features` and `get_matching_set`.

Perform kNN matching and vocoding:

```python
out_wav = knn_vc.match(query_seq, matching_set, topk=4)
```

`out_wav` is a `(T,)` tensor containing the converted 16 kHz waveform, using `k=4` for kNN.

By default, kNN-VC preserves the vocoder output amplitude and does not apply an extra output gain stage. To normalize the final waveform deliberately, pass a target loudness:

```python
out_wav = knn_vc.match(query_seq, matching_set, topk=4, tgt_loudness_db=-16)
```

The target speaker from `ref_wav_paths` can be any speaker, but the reference audio should be clean speech from the desired target speaker. Longer cumulative reference duration generally improves quality, though the improvement diminishes beyond roughly 5 minutes of reference speech.

## Checkpoints

The original kNN-VC release provides three checkpoints:

- The WavLM encoder is loaded natively via the original unilm checkpoint.
- The HiFi-GAN vocoder trained on layer 6 of WavLM features.
- The HiFi-GAN vocoder trained on prematched layer 6 of WavLM features, which is the best model in the paper.

For the HiFi-GAN models, both the generator inference checkpoint and full training checkpoint with optimizer states are provided.

Performance on the LibriSpeech dev-clean set:

| Checkpoint                                                                                                        | WER (%) | CER (%) | EER (%) |
| ----------------------------------------------------------------------------------------------------------------- | :-----: | :-----: | :-----: |
| [kNN-VC with prematched HiFi-GAN](https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt) |  6.29   |  2.34   |  35.73  |
| [kNN-VC with regular HiFi-GAN](https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt)             |  6.39   |  2.41   |  32.55  |

## REST API server

A FastAPI server is available to manage reference voices and perform voice conversion over HTTP. You can run it either inside a Docker container or directly on your host machine.

### Running locally with `uv`

To run the server directly on your host machine without Docker, use the built-in FastAPI CLI via `uv`:

```bash
# For development with auto-reload:
uv run fastapi dev knn_vc/server.py

# For production:
uv run fastapi run knn_vc/server.py --port 8000
```

By default, reference voices are saved to a `./data/voices` directory. You can change this by setting the `KNN_VC_DATA_DIR` environment variable.

### Running with Docker Compose

To launch the server using Docker Compose:

```bash
docker compose up --build
```

### Using the API

Once running, navigate to `http://localhost:8000/docs` in your browser to view the interactive API documentation and test the available endpoints.

`POST /convert` accepts optional form fields for `topk`, `feature_loudness_ceiling_db`, and `tgt_loudness_db`. `feature_loudness_ceiling_db` is disabled by default so WavLM sees the same waveform levels as upstream kNN-VC. Set it to a value such as `-24` only when you deliberately want input-side attenuation. Omit `tgt_loudness_db` to return the model output without an extra output gain stage, or set it to a value such as `-16` to apply final loudness normalization.

## Training

We follow the typical encoder-converter-vocoder setup for voice conversion. The kNN matching procedure acts as the converter, while HiFi-GAN is trained as the vocoder over WavLM features.

This fork focuses especially on making vocoder training more convenient and faster on modern systems.

### HiFi-GAN training

Run the training utilities from the synced `uv` environment.

#### 1. Precompute WavLM features

Precompute WavLM features for the vocoder dataset. For LibriSpeech-style data, use the prematching utility.

The original kNN-VC procedure supports generating either regular WavLM features or prematched WavLM features. Prematched features are used for the best model in the paper.

Example:

```bash
uv run knn-vc-prematch \
  /path/to/librispeech/root \
  /path/where/you/want/outputs/to/go \
  --topk 4 \
  --matching_layer 6 \
  --synthesis_layer 6 \
  --prematch
```

Use `--synthesis_layer 6 --matching_layer 6` for the original single-layer recipe. Layer numbers are one-based WavLM transformer outputs, matching the original kNN-VC/WavLM convention. So layer 6 means the sixth transformer block output.

#### 2. Create WebDataset shards

To improve training efficiency, this fork can pack the audio and SSL features into tar shards and train using WebDataset.

Example:

```bash
uv run knn-vc-create-webdataset \
  ls-dev-clean_prematch \
  /cfs/collections/librispeech/LibriSpeech/dev-clean/ \
  librispeech_prematch/dev-clean/ \
  20
```

The WebDataset path reduces filesystem overhead, especially on networked filesystems, and has reduced training time by roughly 25% in observed runs.

#### 3. Train HiFi-GAN

This repository adapts the original [HiFi-GAN](https://github.com/jik876/hifi-gan) training code to work with WavLM features.

The training entrypoint is:

```bash
uv run knn-vc-train-hifigan knn_vc/hifigan/config.yaml
```

Training can be stopped once it reaches around 2.5M updates, or earlier if audio quality begins to degrade.

## Training performance notes

Observed training times:

| Hardware        | Commit      | Setup                                   | Time per epoch |
| --------------- | ----------- | --------------------------------------- | -------------: |
| 1x H200         | `a51930398` | `/cfs`, original dataset access         |           460s |
| 1x H200         | `a51930398` | `/tmp`, workers increased from 10 to 20 |           160s |
| 1x RTX 6000 Ada | `07ccfa0`   | WebDataset, batch size 32, 20 workers   |           120s |
| 1x RTX 6000 Ada | `07ccfa0`   | WebDataset, batch size 32, 50 workers   |           120s |

These numbers are workload- and filesystem-dependent, but they show the main bottleneck clearly: data loading can dominate vocoder training. WebDataset helps reduce that goblin.

## Acknowledgements

Parts of this project are adapted from the following repositories:

- HiFi-GAN: https://github.com/jik876/hifi-gan
- WavLM: https://github.com/microsoft/unilm/tree/master/wavlm
- kNN-VC: https://github.com/bshall/knn-vc

Thank you to the authors of these projects.

## Citation

```bibtex
@inproceedings{baas2023knnvc,
  author={Matthew Baas and Benjamin van Niekerk and Herman Kamper},
  title={Voice Conversion With Just Nearest Neighbors},
  year=2023,
  booktitle={Interspeech},
}
```

# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.15

FROM ${CUDA_IMAGE} AS python-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-venv \
        ffmpeg


FROM ${UV_IMAGE} AS uv


FROM python-base AS deps

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        python3-dev

COPY --link --from=uv /uv /uvx /usr/local/bin/

WORKDIR /srv/app

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --link pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync \
        --python /usr/bin/python3 \
        --frozen \
        --no-dev \
        --no-install-project \
        --no-compile-bytecode


FROM python-base AS runtime

ENV KNN_VC_DATA_DIR=/data/knn_vc \
    TORCH_HOME=/data/torch \
    PYTHONPATH=/srv/app

WORKDIR /srv/app

COPY --link --from=deps /opt/venv /opt/venv

RUN chown 1000:1000 /srv/app \
    && install -d -o 1000 -g 1000 /data/knn_vc /data/torch

COPY --link --chown=1000:1000 knn_vc ./knn_vc

USER 1000:1000

VOLUME /data

CMD ["uvicorn", "knn_vc.server:app", "--host", "0.0.0.0", "--port", "8000"]

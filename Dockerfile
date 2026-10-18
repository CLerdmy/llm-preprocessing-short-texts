FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/share/nltk_data

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml /build/
RUN pip install "/build[viz,dev]" \
    && python -m nltk.downloader -d $NLTK_DATA stopwords \
    && rm -rf /build

RUN useradd -m -u 1000 -s /bin/bash dev
USER dev
WORKDIR /workspace
# --- stage 1: build Telegram's own Bot API server (removes the 20MB download / 50MB upload limits). Takes ~15 min on 4 cores.
FROM python:3.11-slim AS tgapi
RUN apt-get update && apt-get install -y --no-install-recommends make git zlib1g-dev libssl-dev gperf cmake g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --recursive --depth 1 --shallow-submodules https://github.com/tdlib/telegram-bot-api.git /src
WORKDIR /src/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=/opt/tgapi .. \
    && cmake --build . --target install -j4 \
    && strip /opt/tgapi/bin/telegram-bot-api

# --- stage 2: the bot
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    WHISPER_MODEL=tiny \
    MALLOC_ARENA_MAX=2

# ffmpeg + Arabic-capable fonts (Noto Sans Arabic is used for the burned subtitles)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg fontconfig fonts-noto-core fonts-hosny-amiri fonts-dejavu-core fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*
RUN fc-cache -f

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the Whisper model into the image so cold starts don't download it
RUN python -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8')"

COPY --from=tgapi /opt/tgapi/bin/telegram-bot-api /usr/local/bin/telegram-bot-api
COPY bot.py .
CMD ["python", "bot.py"]
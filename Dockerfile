FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    WHISPER_MODEL=tiny \
    MALLOC_ARENA_MAX=2

# ffmpeg + Arabic-capable fonts (Noto Sans Arabic is used for the burned subtitles)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fontconfig fonts-noto-core fonts-hosny-amiri \
    && rm -rf /var/lib/apt/lists/*
RUN fc-cache -f

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the Whisper model into the image so cold starts don't download it
RUN python -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8')"

COPY bot.py .
CMD ["python", "bot.py"]
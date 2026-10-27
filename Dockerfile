# For hosts that run containers (Render, Fly, a VPS, a paid Hugging Face Docker Space). The free Hugging Face
# tier runs space.py instead.
#   docker build -t practice-studio . && docker run -p 7860:7860 --env-file .env practice-studio
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# the speech models are baked into the image, so a cold start doesn't download a gigabyte
COPY --chown=user scripts/download_models.py scripts/
RUN python scripts/download_models.py --kokoro fp32

COPY --chown=user . .

ENV HOST=0.0.0.0 PORT=7860 PUBLIC_DEMO=1 MAX_SESSION_MIN=20 STT_THREADS=2 TTS_THREADS=2
EXPOSE 7860
CMD ["python", "-m", "app.main"]

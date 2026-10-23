# Hugging Face Spaces (SDK: docker). The app listens on 0.0.0.0:7860.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=7860

# Spaces run containers as UID 1000; create that user so file permissions line up.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /home/user/app

# 1) CPU-only PyTorch (~200 MB instead of ~2.5 GB with CUDA libraries). Spaces' free tier is CPU anyway.
RUN pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) Everything else (cached layer: only re-runs when requirements.txt changes)
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

# 3) App code, static files and model artifacts (models/ must contain the .pt and .pkl - see README)
COPY --chown=user . .

EXPOSE 7860
# Single worker on purpose: the model is loaded once and kept in memory.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
